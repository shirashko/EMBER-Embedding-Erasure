"""Concept-detection and logit-lens scores for one frozen SNMF layer."""
from __future__ import annotations

import json
from typing import Iterable, List, Sequence, Set, Tuple

import torch

from ember.utils import _safe_tokens, vector_to_logits


def encode_coefficients(
    activations: torch.Tensor,
    factors: torch.Tensor,
    *,
    n_iter: int,
    reg: float,
) -> torch.Tensor:
    """Nonnegative coefficients of frozen ``F``.

    ``activations`` is ``(d_mlp, n_tokens)`` and ``factors`` is ``(d_mlp, K)``.
    Returns ``y`` of shape ``(n_tokens, K)``, initialized by nonnegative least
    squares and refined with the Semi-NMF multiplicative update for ``G``.
    """
    if activations.shape[0] != factors.shape[0]:
        raise ValueError(
            f"Activation width {activations.shape[0]} does not match F rows {factors.shape[0]}. "
            "The dictionary was trained on MLP-intermediate (down-proj input) activations."
        )
    _width, n_tokens = activations.shape
    rank = factors.shape[1]
    gram = factors.T @ factors
    eye = torch.eye(rank, device=factors.device, dtype=factors.dtype)
    least_squares = torch.linalg.solve(gram + reg * eye, factors.T @ activations)
    coefficients = least_squares.T.clamp_min(1e-6)
    projection = activations.T @ factors
    proj_pos = projection.clamp_min(0)
    proj_neg = (-projection).clamp_min(0)
    gram_pos = gram.clamp_min(0)
    gram_neg = (-gram).clamp_min(0)
    for _ in range(n_iter):
        numer = proj_pos + coefficients @ gram_neg
        denom = proj_neg + coefficients @ gram_pos
        coefficients = coefficients * torch.sqrt(numer / (denom + 1e-6))
    if n_tokens and not torch.isfinite(coefficients).all():
        raise RuntimeError("Coefficient update produced non-finite values. Lower --g-iters or raise --reg.")
    return coefficients


def concept_detection_score(
    coefficients: torch.Tensor,
    n_concept: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``S_CD`` from the max concept coefficient over the max neutral coefficient.

    Concept tokens occupy the leading ``n_concept`` rows of ``coefficients``.
    """
    if n_concept <= 0 or n_concept >= coefficients.shape[0]:
        raise ValueError("Need at least one concept token and one neutral token")
    max_concept = coefficients[:n_concept].max(dim=0).values
    max_neutral = coefficients[n_concept:].max(dim=0).values
    score = torch.log(max_concept / (max_neutral + eps))
    return score, max_concept, max_neutral


def _content_token_ids(tokenizer, sentences: Sequence[str]) -> Set[int]:
    special = {
        int(token_id)
        for token_id in (
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "unk_token_id", None),
        )
        if token_id is not None
    }
    found: Set[int] = set()
    for sentence in sentences:
        for token_id in tokenizer.encode(sentence, add_special_tokens=False):
            token_id = int(token_id)
            if token_id not in special:
                found.add(token_id)
    return found


def exclusive_vocab(
    tokenizer,
    concept_sentences: Sequence[str],
    neutral_sentences: Sequence[str],
) -> Tuple[Set[int], Set[int]]:
    """Token ids that appear in only the concept sentences, and only the neutral ones."""
    concept_ids = _content_token_ids(tokenizer, concept_sentences)
    neutral_ids = _content_token_ids(tokenizer, neutral_sentences)
    return concept_ids - neutral_ids, neutral_ids - concept_ids


def promotion_from_logits(
    logits: torch.Tensor,
    concept_ids: Iterable[int],
    neutral_ids: Iterable[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """How much each feature promotes concept-only tokens over neutral-only tokens.

    Returns ``(promotion, best_concept_id, best_concept_logit, best_neutral_id, best_neutral_logit)``.
    ``best_neutral_id`` is ``-1`` when there are no neutral-only tokens.
    """
    vocab = logits.shape[1]
    concept_index = torch.tensor(
        sorted(token_id for token_id in concept_ids if 0 <= int(token_id) < vocab),
        dtype=torch.long,
    )
    neutral_index = torch.tensor(
        sorted(token_id for token_id in neutral_ids if 0 <= int(token_id) < vocab),
        dtype=torch.long,
    )
    if concept_index.numel() == 0:
        raise ValueError("No concept-specific tokens. The concept sentences only use neutral vocabulary.")
    concept_logits = logits.index_select(1, concept_index)
    best_concept_logit, best_concept_arg = concept_logits.max(dim=1)
    best_concept_id = concept_index[best_concept_arg]
    if neutral_index.numel() == 0:
        best_neutral_logit = torch.zeros(logits.shape[0], dtype=logits.dtype)
        best_neutral_id = torch.full((logits.shape[0],), -1, dtype=torch.long)
    else:
        neutral_logits = logits.index_select(1, neutral_index)
        best_neutral_logit, best_neutral_arg = neutral_logits.max(dim=1)
        best_neutral_id = neutral_index[best_neutral_arg]
    return (
        best_concept_logit - best_neutral_logit,
        best_concept_id,
        best_concept_logit,
        best_neutral_id,
        best_neutral_logit,
    )


def token_strings(model, token_ids: Sequence[int]) -> List[str]:
    if not token_ids:
        return []
    raw = model.tokenizer.convert_ids_to_tokens([int(token_id) for token_id in token_ids])
    return _safe_tokens([str(token) for token in raw])


def feature_logits(model, factors: torch.Tensor, layer: int) -> torch.Tensor:
    """``logits_k = W_U(LN(W_down F_{:,k}))`` for every column. Shape ``(K, vocab)``."""
    weight_out = model.blocks[layer].mlp.W_out
    decoded = factors.to(device=weight_out.device, dtype=weight_out.dtype).T @ weight_out
    return vector_to_logits(model, decoded).float().cpu()


def top_activating_tokens(
    coefficients: torch.Tensor,
    token_ids: torch.Tensor,
    model,
    k: int,
) -> List[List[str]]:
    """Token strings at the largest coefficients of each column of ``y``."""
    if token_ids.shape[0] != coefficients.shape[0]:
        raise ValueError(
            f"Token ids ({token_ids.shape[0]}) do not match coefficients ({coefficients.shape[0]})"
        )
    take = min(k, coefficients.shape[0])
    if take <= 0:
        return [[] for _ in range(coefficients.shape[1])]
    chosen = torch.topk(coefficients, k=take, dim=0).indices
    ids = token_ids.tolist()
    return [
        token_strings(model, [int(ids[int(row)]) for row in chosen[:, feature].tolist()])
        for feature in range(coefficients.shape[1])
    ]


def score_layer(
    *,
    model,
    factors: torch.Tensor,
    concept_acts: torch.Tensor,
    neutral_acts: torch.Tensor,
    token_ids: torch.Tensor,
    concept_ids: Set[int],
    neutral_ids: Set[int],
    layer: int,
    concept_name: str,
    model_name: str,
    eps: float,
    g_iters: int,
    reg: float,
    k_proj: int,
    fit_device: str,
) -> List[dict]:
    """Score every column of one layer and attach activating and promoted tokens.

    ``token_ids`` lines up with ``cat(concept_acts, neutral_acts)`` along tokens.
    ``activation_ratio`` is the linear max ratio that ``S_CD`` takes the log of.
    """
    activations = torch.cat([concept_acts, neutral_acts], dim=0).T.contiguous()
    device_factors = factors.to(fit_device)
    with torch.inference_mode():
        coefficients = encode_coefficients(
            activations.to(fit_device),
            device_factors,
            n_iter=g_iters,
            reg=reg,
        ).cpu()
        score, max_concept, max_neutral = concept_detection_score(coefficients, concept_acts.shape[0], eps)
        logits = feature_logits(model, factors, layer)
    promotion, best_c, best_c_logit, best_n, best_n_logit = promotion_from_logits(
        logits, concept_ids, neutral_ids,
    )
    activation_ratio = max_concept / (max_neutral + eps)
    activating = top_activating_tokens(coefficients, token_ids, model, k_proj)
    top_indices = torch.topk(logits, k=min(k_proj, logits.shape[1]), dim=-1).indices
    rows = []
    rank = int(factors.shape[1])
    for feature in range(rank):
        promoted = token_strings(model, top_indices[feature].tolist())
        neutral_id = int(best_n[feature])
        rows.append({
            "model": model_name,
            "concept": concept_name,
            "rank": rank,
            "layer": layer,
            "feature": feature,
            "s_cd": float(score[feature]),
            "max_y_concept": float(max_concept[feature]),
            "max_y_neutral": float(max_neutral[feature]),
            "activation_ratio": float(activation_ratio[feature]),
            "promotion_score": float(promotion[feature]),
            "best_concept_token": token_strings(model, [int(best_c[feature])])[0],
            "best_concept_logit": float(best_c_logit[feature]),
            "best_neutral_token": "" if neutral_id < 0 else token_strings(model, [neutral_id])[0],
            "best_neutral_logit": float(best_n_logit[feature]),
            "activating_tokens": json.dumps(activating[feature], ensure_ascii=False),
            "projection_top_tokens": json.dumps(promoted, ensure_ascii=False),
        })
    return rows
