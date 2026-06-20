"""Embedding-space erasure primitives.

Subtracts the SparseMatrixFactorization concept contribution from each
eligible concept token's embedding:

    c_i = F'[:, k'] @ G'[i, k']   (concept contribution for token i)
    e_i_new = e_i - delta * c_i

F (shape [d_model, K]) is the dense set of erasure directions;
G (shape [|V'|, K]) is the WTA-sparse signed per-token activation scores.
Only tokens labeled as concept-only are edited (intersection with nonzero G).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ember.utils import _safe_tokens
from ember.erasure import features, log


# ========================================================================== #
# Embedding weight helpers                                                    #
# ========================================================================== #

def _embedding_weight(model: torch.nn.Module) -> torch.Tensor:
    emb = model.get_input_embeddings()
    if emb is None or getattr(emb, "weight", None) is None:
        raise ValueError("HF model has no input embedding weight.")
    return emb.weight


def snapshot(model: torch.nn.Module) -> torch.Tensor:
    """Return a CPU copy of the input-embedding weight tensor."""
    return _embedding_weight(model).data.detach().cpu().clone()


def restore(model: torch.nn.Module, snap: torch.Tensor) -> None:
    """Overwrite the model's input-embedding weight from a saved snapshot."""
    W = _embedding_weight(model)
    with torch.no_grad():
        W.data.copy_(snap.to(device=W.device, dtype=W.dtype))


def _embedding_scale(model: torch.nn.Module, model_name: str) -> float:
    """Read-side scale to bring W_E into the factorization's scaled space.

    HF Gemma stores W_E unscaled; the forward pass applies sqrt(d_model) at
    runtime. We multiply by sqrt(d_model) on read and divide on write so the
    math operates in the same scaled space as the factorization.
    Llama has no embedding scale, so returns 1.0.
    """
    if "gemma" not in model_name.lower():
        return 1.0
    W = _embedding_weight(model)
    return float(np.sqrt(int(W.shape[1])))


# ========================================================================== #
# Concept token set                                                           #
# ========================================================================== #

def _build_concept_token_set(model_name: str, concept_name: str,
                              rank: int, seed: int,
                              vprime_token_ids: List[int]) -> Optional[set]:
    """Set of token-ids labeled exactly as concept_name in the SNMF token CSV.

    Returns None when the token-label CSV is missing.
    """
    str_to_label = features.load_token_label_map(model_name, concept_name, rank, seed)
    if not str_to_label:
        return None

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_strs = tokenizer.convert_ids_to_tokens(vprime_token_ids)
    cleaned = [
        s.replace("▁", " ").replace("Ġ", " ") if isinstance(s, str) else ""
        for s in hf_strs
    ]
    id_to_safe = {tid: s for tid, s in zip(vprime_token_ids, _safe_tokens(cleaned))}
    return {tid for tid in vprime_token_ids
            if str_to_label.get(id_to_safe.get(tid, ""), "") == concept_name}


# ========================================================================== #
# Core erasure                                                                #
# ========================================================================== #

def _eligible_tids_rows(vprime_token_ids: List[int], G_prime: torch.Tensor,
                        concept_token_ids: set) -> Tuple[List[int], List[int]]:
    """Eligible (token_id, row) pairs for the edit.

    A token is eligible when it is a concept token, lives in V', and has a
    nonzero G value on at least one selected feature. ``G_prime`` is G already
    restricted to the selected feature columns (shape ``[|V'|, k']``).
    """
    tid_to_row = {tid: i for i, tid in enumerate(vprime_token_ids)}
    tids: List[int] = []
    rows: List[int] = []
    for tid in concept_token_ids:
        if tid not in tid_to_row:
            continue
        row = tid_to_row[tid]
        if float(G_prime[row, :].abs().max()) > 0.0:
            tids.append(tid)
            rows.append(row)
    return tids, rows


def _erase_factored(
        hf_model: torch.nn.Module,
        model_name: str,
        F_dir: torch.Tensor,
        G_tok: torch.Tensor,
        vprime_token_ids: List[int],
        feature_ids: List[int],
        delta_embed: float,
        concept_token_ids: set,
) -> int:
    """Apply the factored embedding edit. Returns n_edited_tokens.

    For each eligible token i (concept-only ∩ in V' ∩ nonzero G across the
    selected features):

        c_i = F'[:, feature_ids] @ G_tok[row_i, feature_ids]
        e_i ← e_i - delta * c_i
    """
    if not concept_token_ids or not feature_ids:
        return 0

    W_E = _embedding_weight(hf_model)
    device, dtype = W_E.device, W_E.dtype
    scale = _embedding_scale(hf_model, model_name)

    F_prime = torch.as_tensor(F_dir[:, feature_ids]).to(device=device, dtype=dtype)  # [d, k']
    G_prime = torch.as_tensor(G_tok[:, feature_ids])                                  # [|V'|, k'] cpu

    eligible_tids, eligible_rows = _eligible_tids_rows(
        vprime_token_ids, G_prime, concept_token_ids)
    if not eligible_tids:
        return 0

    tids = torch.tensor(eligible_tids, device=device, dtype=torch.long)
    G_rows = G_prime[eligible_rows, :].to(device=device, dtype=dtype)  # [n_tok, k']
    C = (F_prime @ G_rows.T).T                                          # [n_tok, d]

    with torch.no_grad():
        E = W_E.data.index_select(0, tids) * scale
        W_E.data.index_copy_(0, tids, (E - float(delta_embed) * C) / scale)

    return len(eligible_tids)


# ========================================================================== #
# Public API                                                                  #
# ========================================================================== #

def apply_concept_embed_edit_factored(
        hf_model: torch.nn.Module,
        model_name: str,
        concept_name: str,
        delta_embed: float,
        rank: int,
        seed: int,
        ratio_thresh: Optional[float] = 2.0,
) -> Dict[str, Any]:
    """Apply the factored embedding edit for one concept.

    Returns a dict with keys delta_embed, k_features_embed, n_tokens_edited.
    delta_embed=0.0 is a no-op and returns zeros.
    """
    if float(delta_embed) == 0.0:
        return {"delta_embed": 0.0, "k_features_embed": 0, "n_tokens_edited": 0}

    pkl_path, pot_csv = features._embedding_paths(model_name, concept_name, rank, seed)
    F_dir, G_tok, vprime_ids = features.load_embedding_payload(pkl_path)
    feat_ids = features.select_embed_feature_ids(pd.read_csv(pot_csv),
                                                 ratio_thresh=ratio_thresh)

    concept_token_ids = _build_concept_token_set(
        model_name, concept_name, rank, seed, vprime_ids
    )
    if concept_token_ids is None:
        log.warning("embed edit: no token CSV for %r; skipping", concept_name)
        return {"delta_embed": float(delta_embed),
                "k_features_embed": int(len(feat_ids)),
                "n_tokens_edited": 0}
    if not concept_token_ids:
        log.warning("embed edit: empty concept set for %r; skipping", concept_name)
        return {"delta_embed": float(delta_embed),
                "k_features_embed": int(len(feat_ids)),
                "n_tokens_edited": 0}

    n_edited = _erase_factored(
        hf_model=hf_model,
        model_name=model_name,
        F_dir=F_dir,
        G_tok=G_tok,
        vprime_token_ids=vprime_ids,
        feature_ids=feat_ids,
        delta_embed=float(delta_embed),
        concept_token_ids=concept_token_ids,
    )
    return {
        "delta_embed": float(delta_embed),
        "k_features_embed": int(len(feat_ids)),
        "n_tokens_edited": int(n_edited),
    }


def concept_edit_tokens(model_name: str, concept_name: str, rank: int, seed: int,
                        ratio_thresh: Optional[float] = 2.0) -> List[Tuple[str, float]]:
    """``(token, edit_magnitude)`` pairs for the tokens the EMBER edit modifies.

    Same eligibility as the edit itself (concept tokens in V' with nonzero G on a
    selected feature), so this is exactly the set of edited tokens. The magnitude
    is ``||c_i||`` where ``c_i = F'[:, feat] @ G'[i, feat]`` is the vector
    subtracted from token i's embedding (delta and the embedding scale are
    constant across tokens, so this norm ranks how strongly each token is edited).
    Returned sorted from most to least edited. Empty when the token-label CSV or
    concept set is missing.
    """
    pkl_path, pot_csv = features._embedding_paths(model_name, concept_name, rank, seed)
    F_dir, G_tok, vprime_ids = features.load_embedding_payload(pkl_path)
    feat_ids = features.select_embed_feature_ids(pd.read_csv(pot_csv),
                                                 ratio_thresh=ratio_thresh)
    concept_token_ids = _build_concept_token_set(
        model_name, concept_name, rank, seed, vprime_ids)
    if not concept_token_ids or not feat_ids:
        return []

    G_prime = torch.as_tensor(G_tok[:, feat_ids])
    tids, rows = _eligible_tids_rows(vprime_ids, G_prime, concept_token_ids)
    if not tids:
        return []

    F_prime = torch.as_tensor(F_dir[:, feat_ids]).float()       # [d, k']
    G_rows = G_prime[rows, :].float()                            # [n, k']
    mags = (F_prime @ G_rows.T).norm(dim=0)                      # [n] = ||c_i||
    order = torch.argsort(mags, descending=True).tolist()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    toks = tokenizer.convert_ids_to_tokens([tids[i] for i in order])
    cleaned = [t.replace("▁", " ").replace("Ġ", " ") if isinstance(t, str) else ""
               for t in toks]
    return [(tok, float(mags[i])) for tok, i in zip(cleaned, order)]


__all__ = [
    "snapshot", "restore",
    "apply_concept_embed_edit_factored",
    "concept_edit_tokens",
]
