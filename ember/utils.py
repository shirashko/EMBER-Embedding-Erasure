"""Shared utilities for the MF feature-extraction pipeline.

Holds the :class:`SparseMatrixFactorization` embedding factorizer plus the
path/seed helpers, device/seed setup, and the per-feature stat builders used
by ``train_mf_features.py`` and ``interpret_features.py``.
"""
import json
import os
import re
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

from factorization.seminmf import (
    NMFSemiNMF,
    init_knn, init_svd,
    wta_features, wta_cols, fix_hoyer_scale,
)


# ---------------------------------------------------------------------------
# Embedding-specific Semi-NMF subclass
# ---------------------------------------------------------------------------

class SparseMatrixFactorization(NMFSemiNMF):
    """Sparse Matrix Factorization for the token embedding matrix.

    Factorizes A = (d_model, |V'|) into:
      F_ : (d_model, K)  - dense erasure directions in model space
      G_ : (|V'|, K)     - WTA-sparse, signed per-token activation scores

    Both F_ and G_ are unconstrained (no non-negativity). sparsity=1.0
    is passed to the parent so wta_features(F_) is a no-op (F_ stays
    dense). wta_cols(G_) is applied each iteration to enforce per-feature
    token sparsity during fitting.
    """

    def __init__(self, rank: int, fitting_device: str = "cpu", g_sparsity: float = 0.01):
        super().__init__(rank, fitting_device=fitting_device, sparsity=1.0)
        self.g_sparsity = g_sparsity

    def fit(self, A, max_iter=500, tol=1e-4, reg=1e-6, patience=100, verbose=True, init="random"):
        A = A.to(self.fitting_device)
        d_model, n_tokens = A.shape
        K = self.rank

        if init == "knn":
            F0, G0 = init_knn(A, K)
        elif init == "svd":
            F0, G0 = init_svd(A, K)
        elif init == "random":
            # randn (not rand) so G starts signed - no spurious bias toward
            # the non-negative orthant that would slow the descent into a
            # signed SVD-like solution.
            G0 = torch.randn(n_tokens, K, device=self.fitting_device)
            F0 = torch.randn(d_model, K, device=self.fitting_device)
        else:
            raise ValueError(f"Unknown init '{init}'")

        self.G_ = nn.Parameter(G0.to(self.fitting_device))
        self.F_ = nn.Parameter(F0.to(self.fitting_device))

        best_loss = float("inf")
        best_F = best_G = None
        num_no_improve = 0
        eye_K = torch.eye(K, device=self.fitting_device)

        with torch.no_grad():
            for it in range(max_iter):
                # F update (closed-form least squares):
                # F = A @ G @ (G.T @ G + reg*I)^{-1}
                GtG = self.G_.T @ self.G_
                inv_G = torch.linalg.inv(GtG + eye_K * reg)
                self.F_.data.copy_(A @ self.G_ @ inv_G)
                wta_features(self.F_, pct_keep=self.sparsity)  # no-op (sparsity=1.0)
                fix_hoyer_scale(self.F_, self.G_)

                # G update (closed-form least squares):
                # G = A.T @ F @ (F.T @ F + reg*I)^{-1}
                FtF = self.F_.T @ self.F_
                inv_F = torch.linalg.inv(FtF + eye_K * reg)
                self.G_.data.copy_(A.T @ self.F_ @ inv_F)

                # WTA on |G| - top g_sparsity fraction of |V'| tokens per
                # feature column, zeroing the rest. Works for signed G.
                wta_cols(self.G_, pct_keep=self.g_sparsity)

                loss = torch.norm(A - self.F_ @ self.G_.T, p="fro") ** 2

                if loss.item() < best_loss - tol:
                    best_loss = loss.item()
                    best_F = self.F_.data.clone()
                    best_G = self.G_.data.clone()
                    num_no_improve = 0
                else:
                    num_no_improve += 1

                if verbose and (it % 50 == 0 or num_no_improve == 1):
                    print(f"Iter {it:4d}: loss={loss.item():.6f}  "
                          f"(best={best_loss:.6f}, no_improve={num_no_improve})")

                if num_no_improve >= patience:
                    if verbose:
                        print(f"Stopping early at iter {it} (no improvement in {patience} iters)")
                    break

        if best_F is not None:
            with torch.no_grad():
                self.F_.data.copy_(best_F)
                self.G_.data.copy_(best_G)

        self.W = self.G_.detach().clone()
        self.H = self.F_.T
        return self


# ---------------------------------------------------------------------------
# Basic Helpers
# ---------------------------------------------------------------------------

def update_timing(timing_path: Path, key: str, elapsed: float) -> None:
    data = {}
    if timing_path.exists():
        try:
            content = timing_path.read_text().strip()
            if content:
                data = json.loads(content)
        except (json.JSONDecodeError, OSError):
            pass
    data[key] = round(elapsed, 2)
    timing_path.write_text(json.dumps(data, indent=2))


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(spec: str) -> str:
    spec = spec.lower()
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def _safe_concept(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^\w\-]+", "_", name.strip().replace(" ", "_"))).strip("_")


def _safe_tokens(tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    for t in tokens:
        if t is None: continue
        if not isinstance(t, str): t = str(t)
        out.append(t.replace("\r", "\\r").replace("\n", "\\n"))
    return out


# ---------------------------------------------------------------------------
# Path & IO Management
# ---------------------------------------------------------------------------

def get_pipeline_path(
        outdir: str, model_safe: str, file_type: str, rank: int,
        concept_safe: str, track: str, filename: str, seed: int = 42,
) -> str:
    """Build a structured output path for the MF feature-extraction pipeline.

    Layout:
        ``outdir / model_safe / {pickles,csvs} / rank<R> / seed<S> / concept_safe / {embedding,mlp} / filename``
    """
    d = (Path(outdir) / model_safe / file_type
         / f"rank{rank}"
         / f"seed{seed}"
         / concept_safe / track)
    d.mkdir(parents=True, exist_ok=True)
    return str(d / filename)


def save_df_to_csv(df: pd.DataFrame, path: str, dedupe_cols: List[str] = None) -> None:
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df], ignore_index=True)

    if not df.empty and dedupe_cols and set(dedupe_cols).issubset(df.columns):
        df = df.drop_duplicates(subset=dedupe_cols, keep="last")
        df = df.sort_values(dedupe_cols, ascending=[True] * len(dedupe_cols))

    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Model / Activation Extraction
# ---------------------------------------------------------------------------

def get_embedding_matrix(model) -> torch.Tensor:
    if hasattr(model, "W_E") and torch.is_tensor(model.W_E): return model.W_E
    if hasattr(model, "embed") and hasattr(model.embed, "W_E") and torch.is_tensor(
        model.embed.W_E): return model.embed.W_E
    raise RuntimeError("Could not locate token embedding matrix.")


def get_special_token_ids(model) -> set:
    tok = model.tokenizer
    return {int(getattr(tok, attr)) for attr in ["pad_token_id", "bos_token_id", "eos_token_id"] if
            getattr(tok, attr, None) is not None}


def vector_to_logits(model: Any, v: Tensor, use_ln_final: bool = True) -> Tensor:
    v = v.to(device=model.W_U.device, dtype=model.W_U.dtype)
    if use_ln_final and hasattr(model, "ln_final") and model.ln_final is not None:
        v = model.ln_final(v)
    return model.unembed(v)


def generate_token_contexts(tokens: Sequence[int], sample_ids: Sequence[int], act_generator: Any,
                            context_window: int = 15) -> List[Tuple[str, str]]:
    assert len(tokens) == len(sample_ids)
    token_ds = []
    for i in range(len(tokens)):
        current_sample_id = sample_ids[i]
        token_str = act_generator.model.to_str_tokens([tokens[i]])[0][0]
        start = max(0, i - context_window)
        end = min(len(tokens), i + context_window + 1)
        context_tokens = [act_generator.model.to_str_tokens([tokens[j]])[0][0] for j in range(start, end) if
                          sample_ids[j] == current_sample_id]
        token_ds.append((token_str, "".join(context_tokens)))
    return token_ds


def get_top_activating_indices_magnitude(G: Tensor, feature_idx: int, num_samples: int = 20) -> List[int]:
    s = G[:, feature_idx]
    k = min(num_samples, s.shape[0])
    if k == 0:
        return []
    _, local_idx = torch.topk(s.abs(), k=k, largest=True)
    return local_idx.tolist()


def collect_feature_rows_for_layer(G: Tensor, rank: int, token_ds: Sequence[Tuple[str, str]], labels: Sequence[str],
                                   layer: int, model_name: str, concept_name: str, num_samples: int = 30,
                                   threshold: float = 0.3) -> List[Dict[str, Any]]:
    rows = []
    rank_actual = min(rank, G.shape[1])
    for k in range(rank_actual):
        idxs = get_top_activating_indices_magnitude(G, k, num_samples=num_samples)
        raw_tokens = [token_ds[i][0] for i in idxs]
        labels_list = [labels[i] for i in idxs]
        num_concept_related = sum(1 for lbl in labels_list if lbl == concept_name)
        frac = num_concept_related / max(len(labels_list), 1)
        rows.append({
            "model": model_name, "concept": concept_name, "rank": rank_actual,
            "layer": layer, "feature": k, "activating_tokens": _safe_tokens(raw_tokens),
            "labels": labels_list, "num_concept_related": int(num_concept_related),
            "is_concept_related": bool(frac >= threshold),
            "projection_top_tokens": [], "projection_bottom_tokens": [], "projection_abs_top_tokens": []
        })
    return rows


def fit_with_ridge(nmf, A: torch.Tensor, max_iter: int, patience: int = 500, base_reg: float = 1e-4,
                   tries: int = 4) -> float:
    reg = float(base_reg)
    last_err = None
    for t in range(tries):
        try:
            nmf.fit(A, max_iter, patience=patience, reg=reg)
            return reg
        except Exception as e:
            last_err = e
            reg *= 10.0
    raise RuntimeError(f"NMF failed after {tries} tries. Last error: {last_err}") from last_err


# ---------------------------------------------------------------------------
# Embedding Specific Stats
# ---------------------------------------------------------------------------

def build_token_label_codes(concept_name: str, dataset, tokenizer) -> Dict[int, int]:
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    origin = {}
    for sent, lbl in dataset:
        if not sent: continue
        for tid in tokenizer(sent, add_special_tokens=False)["input_ids"]:
            tid = int(tid)
            if tid in special: continue
            origin.setdefault(tid, set()).add(
                "neutral" if lbl == "Neutral" else "concept" if lbl == concept_name else str(lbl))

    token_to_code = {}
    for tid, s in origin.items():
        if "concept" in s and "neutral" in s:
            token_to_code[tid] = 2
        elif "concept" in s:
            token_to_code[tid] = 1
        else:
            token_to_code[tid] = 0
    return token_to_code


def compute_embedding_stats(G_tok: torch.Tensor, vprime_token_ids: List[int], token_to_code: Dict[int, int],
                            concept_name: str, nonzero_eps: float = 1e-12, ratio_eps: float = 1e-8) -> pd.DataFrame:
    """Per-feature concept-vs-neutral stats for the embedding track.

    G_tok : (|V'|, K) - signed per-token activation scores from SparseMatrixFactorization.
    Abs is applied before computing means so concept/neutral separation is measured
    by activation magnitude rather than sign. 'Both' tokens are counted but excluded
    from ratio/mean/sum computations. Returns both mean-based stats (primary, comparable
    to MLP ratio_abs) and sum-based stats (secondary).
    """
    Vp, K = G_tok.shape
    G_abs = G_tok.abs().numpy()  # (Vp, K)

    codes = np.zeros(Vp, dtype=np.int8)
    for i, tid in enumerate(vprime_token_ids):
        codes[i] = token_to_code.get(tid, 0)

    is_concept = codes == 1
    is_neutral  = codes == 0
    is_both     = codes == 2

    rows = []
    for k in range(K):
        g = G_abs[:, k]
        nz = g > nonzero_eps

        mean_con = float(g[is_concept].mean()) if is_concept.any() else 0.0
        mean_neu = float(g[is_neutral].mean()) if is_neutral.any() else 0.0

        sum_con  = float(g[is_concept].sum())
        sum_neu  = float(g[is_neutral].sum())

        num_con  = int((nz & is_concept).sum())
        num_neu  = int((nz & is_neutral).sum())
        num_both = int((nz & is_both).sum())

        rows.append({
            "concept":           concept_name,
            "feature":           k,
            "num_concept":       num_con,
            "num_neutral":       num_neu,
            "num_both":          num_both,
            "mean_abs_concept":  mean_con,
            "mean_abs_neutral":  mean_neu,
            "diff_abs":          mean_con - mean_neu,
            "ratio_abs":         mean_con / (mean_neu + ratio_eps),
            "sum_abs_concept":   sum_con,
            "sum_abs_neutral":   sum_neu,
            "sum_ratio_cn":      (sum_con + ratio_eps) / (sum_neu + ratio_eps),
        })
    return pd.DataFrame(rows)


def collect_feature_rows_for_embeddings(G_tok: torch.Tensor, vprime_token_ids: Sequence[int],
                                        token_label_map: Dict[int, str], model, model_name: str, concept_name: str,
                                        rank: int, max_activating: int = 200) -> List[Dict[str, Any]]:
    """Build feature CSV rows from G (signed per-token activation scores, |V'| × K).

    Activating tokens = nonzero rows of G[:, k], sorted by descending |activation|.
    Projection tokens are filled in by the caller after projecting F columns through W_U.
    """
    n_vocabprime, K = G_tok.shape
    token_strs = model.to_str_tokens(torch.tensor(list(vprime_token_ids), dtype=torch.long))
    flat_strs = []
    for t in token_strs:
        if isinstance(t, list) and len(t) > 0 and isinstance(t[0], list):
            flat_strs.append(str(t[0][0]))
        elif isinstance(t, list):
            flat_strs.append(str(t[0]))
        else:
            flat_strs.append(str(t))
    vprime_tok_strs = _safe_tokens(flat_strs)

    rows = []
    for k in range(min(rank, K)):
        col = G_tok[:, k]
        nz = torch.where(col.abs() > 1e-12)[0].tolist()
        nz_sorted = sorted(nz, key=lambda i: float(col.abs()[i]), reverse=True)[:max_activating] if len(nz) > 1 else nz

        act_tokens = [vprime_tok_strs[i] for i in nz_sorted]
        labels_list = [token_label_map.get(int(vprime_token_ids[i]), "Neutral") for i in nz_sorted]
        frac = sum(1 for lab in labels_list if lab in (concept_name, "both")) / max(len(labels_list), 1)

        rows.append({
            "model": model_name, "concept": concept_name, "rank": min(rank, K), "feature": k,
            "num_activating_tokens_all": len(nz), "activating_tokens": act_tokens, "labels": labels_list,
            "num_concept_related": sum(1 for lab in labels_list if lab in (concept_name, "both")),
            "is_concept_related": bool(frac >= 0.7),
            "projection_top_tokens": [], "projection_bottom_tokens": [], "projection_abs_top_tokens": []
        })
    return rows


# ---------------------------------------------------------------------------
# MLP Specific Stats
# ---------------------------------------------------------------------------

def compute_mlp_layer_stats(G: torch.Tensor, is_concept: np.ndarray, is_neutral: np.ndarray, layer: int, rank: int,
                            model_name: str, concept_name: str, eps: float = 1e-8) -> pd.DataFrame:
    G_np, G_abs = G.numpy(), G.abs().numpy()
    K = G.shape[1]

    c_abs_mean = G_abs[is_concept].mean(axis=0) if is_concept.any() else np.zeros(K)
    n_abs_mean = G_abs[is_neutral].mean(axis=0) if is_neutral.any() else np.zeros(K)
    c_sign_mean = G_np[is_concept].mean(axis=0) if is_concept.any() else np.zeros(K)
    n_sign_mean = G_np[is_neutral].mean(axis=0) if is_neutral.any() else np.zeros(K)

    return pd.DataFrame({
        "model": [model_name] * K, "concept": [concept_name] * K, "rank": [rank] * K,
        "layer": [layer] * K, "feature": list(range(K)),
        "mean_abs_concept": c_abs_mean, "mean_abs_neutral": n_abs_mean,
        "diff_abs": c_abs_mean - n_abs_mean, "ratio_abs": c_abs_mean / (n_abs_mean + eps),
        "mean_signed_concept": c_sign_mean, "mean_signed_neutral": n_sign_mean, "diff_signed": c_sign_mean - n_sign_mean
    })


__all__ = [
    "SparseMatrixFactorization",
    "update_timing", "set_seed", "resolve_device",
    "_safe_model_name", "_safe_concept", "_safe_tokens",
    "get_pipeline_path", "save_df_to_csv",
    "get_embedding_matrix", "get_special_token_ids", "vector_to_logits",
    "generate_token_contexts", "get_top_activating_indices_magnitude",
    "collect_feature_rows_for_layer", "fit_with_ridge",
    "build_token_label_codes", "compute_embedding_stats",
    "collect_feature_rows_for_embeddings", "compute_mlp_layer_stats",
]
