"""SNMF / SparseMatrixFactorization feature loading + selection (MLP and embedding space).

Three groups of helpers:

1. **MLP features** -- per (layer, feature_idx) directions in d_model space,
   built from the SNMF factorization F_ and the MLP weights.
2. **Embedding features** -- feature columns from the SparseMatrixFactorization
   of the token embedding matrix.
3. **ConceptContext** -- per-concept cache that holds the potential-features
   DataFrame and lazily-built per-range feature tensors. Reused across the
   grid / validate-top-K / final-test stages so SNMF pickles are loaded at
   most once per (layer_range, w_mode) per concept.
"""
from __future__ import annotations

import ast
import hashlib
import io
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from ember.utils import _safe_concept, _safe_model_name, get_pipeline_path
from ember.erasure import log

ROOT_DIR = Path(__file__).resolve().parents[2]
MF_OUTPUTS_ROOT = ROOT_DIR / "mf_outputs"

VALID_FEATURE_SOURCES = ("activation", "projection", "all", "mixed")
VALID_W_MODES = ("in", "out", "both")


# ========================================================================== #
# Low-level pickle/weight helpers                                             #
# ========================================================================== #

def _pickle_load(f) -> Any:
    """pickle.load, remapping CUDA tensors to CPU on machines without CUDA.

    The factorization pickles hold tensors saved on GPU; on a CUDA machine they
    load normally, otherwise torch's storage loader is briefly redirected to CPU.
    """
    if torch.cuda.is_available():
        return pickle.load(f)
    import torch.storage as _ts
    original = _ts._load_from_bytes
    _ts._load_from_bytes = lambda b: torch.load(
        io.BytesIO(b), map_location="cpu", weights_only=False)
    try:
        return pickle.load(f)
    finally:
        _ts._load_from_bytes = original


def _load_nmf_pickle(path) -> Any:
    with open(path, "rb") as f:
        return _pickle_load(f)


def _load_feature(model, nmf, layer: int, feature_idx: int,
                  out: bool = True) -> torch.Tensor:
    """Build a d_model-space direction from F_ column feature_idx.

    W_in_TL == up_proj_HF.T,  W_out_TL == down_proj_HF.T.
    Both produce a [d_model] vector via contraction with the MLP weights.
    """
    F = nmf.F_
    if not torch.is_tensor(F):
        F = torch.as_tensor(F)
    f = F[:, int(feature_idx)]  # [d_mlp]

    mlp = model.model.layers[int(layer)].mlp
    if out:
        W = mlp.down_proj.weight  # [d_model, d_mlp]
        f = f.to(device=W.device, dtype=W.dtype)
        return (W @ f).detach()
    else:
        W = mlp.up_proj.weight   # [d_mlp, d_model]
        f = f.to(device=W.device, dtype=W.dtype)
        return (f @ W).detach()


# ========================================================================== #
# MLP path helpers                                                            #
# ========================================================================== #

def _mlp_pickle_path(model_name: str, concept_name: str, layer: int,
                     rank: int, seed: int) -> Path:
    return Path(get_pipeline_path(
        str(MF_OUTPUTS_ROOT), _safe_model_name(model_name), "pickles",
        rank, _safe_concept(concept_name), "mlp", f"layer{layer}.pkl", seed=seed,
    ))


def _mass_norm_cache_dir(model_name: str, concept_name: str,
                         rank: int, seed: int) -> Path:
    """Cache directory for per-neuron mass_norm/degree arrays."""
    return (MF_OUTPUTS_ROOT / _safe_model_name(model_name) / "cache"
            / f"rank{rank}" / f"seed{seed}"
            / _safe_concept(concept_name) / "mlp" / "mass_norm")


# ========================================================================== #
# MLP feature loaders                                                         #
# ========================================================================== #

def load_mlp_potential_df(
        model_name: str,
        concept_name: str,
        rank: int,
        seed: int,
        ratio_thresh: Optional[float] = None,
) -> pd.DataFrame:
    """Read the MLP potential-features CSV produced by interpret_features.py.

    Raises FileNotFoundError if the CSV is missing.
    Filters by ratio_abs >= ratio_thresh when provided.
    """
    csv_path = Path(get_pipeline_path(
        str(MF_OUTPUTS_ROOT), _safe_model_name(model_name), "interpretations",
        rank, _safe_concept(concept_name), "mlp", "potential_features.csv", seed=seed,
    ))
    if not csv_path.exists():
        raise FileNotFoundError(f"MLP potential_features.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "ratio_abs" not in df.columns and "metric_score" in df.columns:
        df = df.rename(columns={"metric_score": "ratio_abs"})
    if ratio_thresh is not None:
        if "ratio_abs" not in df.columns:
            raise ValueError(f"{csv_path} missing 'ratio_abs'/'metric_score' (needed for ratio_thresh)")
        df = df[pd.to_numeric(df["ratio_abs"], errors="coerce") >= float(ratio_thresh)].copy()
    return df


def select_mlp_specs(df: pd.DataFrame, feature_source: str) -> List[Tuple[int, int]]:
    """Return sorted (layer, feature_idx) pairs surviving the source filter.

    feature_source:
        activation -- rows where source == "activating_tokens"
        projection -- rows where source == "projection_top_tokens"
        all        -- everything
        mixed      -- activations for layers <=17, projections for >=18
    """
    if feature_source not in VALID_FEATURE_SOURCES:
        raise ValueError(f"Unknown feature_source {feature_source!r}; expected one of {VALID_FEATURE_SOURCES}")
    for col in ("layer", "feature", "source"):
        if col not in df.columns:
            raise ValueError(f"potential features CSV missing column {col!r}")

    work = df.copy()
    work["layer"] = pd.to_numeric(work["layer"], errors="coerce")
    work["feature"] = pd.to_numeric(work["feature"], errors="coerce")
    work = work.dropna(subset=["layer", "feature"])
    work["layer"] = work["layer"].astype(int)
    work["feature"] = work["feature"].astype(int)

    if feature_source == "activation":
        work = work[work["source"] == "activating_tokens"]
    elif feature_source == "projection":
        work = work[work["source"] == "projection_top_tokens"]
    elif feature_source == "mixed":
        lower = (work["layer"] <= 17) & (work["source"] == "activating_tokens")
        upper = (work["layer"] >= 18) & (work["source"] == "projection_top_tokens")
        work = work[lower | upper]

    work = work.drop_duplicates(subset=["layer", "feature"]).sort_values(["layer", "feature"])
    return [(int(r.layer), int(r.feature)) for r in work.itertuples(index=False)]


def build_layer_features(
        model: Any,
        model_name: str,
        concept_name: str,
        specs: Sequence[Tuple[int, int]],
        rank: int,
        seed: int,
        w_mode: str,
) -> List[Tuple]:
    """Materialize (layer, f_out[, f_in], mask) tuples from SNMF specs.

    Return shape per item:
        w_mode="in"   -> (layer, f_in,  mask)
        w_mode="out"  -> (layer, f_out, mask)
        w_mode="both" -> (layer, f_out, f_in, mask)
    """
    if w_mode not in VALID_W_MODES:
        raise ValueError(f"Unknown w_mode {w_mode!r}; expected one of {VALID_W_MODES}")

    out: List[Tuple] = []
    for (layer, feat_idx) in specs:
        nmf = _load_nmf_pickle(_mlp_pickle_path(model_name, concept_name, layer, rank, seed))
        f_out = f_in = None
        if w_mode != "out":
            f_in = _load_feature(model, nmf, layer, feat_idx, out=False)
        if w_mode != "in":
            f_out = _load_feature(model, nmf, layer, feat_idx, out=True)

        F = nmf.F_
        if F is None:
            raise ValueError(f"SNMF F_ is None for layer={layer} concept={concept_name!r}")
        mask_np = (F[:, int(feat_idx)] != 0)
        ref = f_out if f_out is not None else f_in
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=ref.device)

        if w_mode == "both":
            out.append((layer, f_out, f_in, mask))
        elif w_mode == "out":
            out.append((layer, f_out, mask))
        else:
            out.append((layer, f_in, mask))
    return out


# ========================================================================== #
# Selection filters: neurons / coverage                                       #
# ========================================================================== #

def apply_neurons_filter(layers_fs: List[Tuple], w_mode: str,
                         neurons_thresh: Optional[int]) -> List[Tuple]:
    """Keep only d_mlp neurons covered by >= neurons_thresh features per layer."""
    if neurons_thresh is None:
        return layers_fs

    thr = int(neurons_thresh)
    by_layer: Dict[int, List[Tuple]] = {}
    for item in layers_fs:
        by_layer.setdefault(int(item[0]), []).append(item)

    out: List[Tuple] = []
    for L, items in by_layer.items():
        mask_idx = 3 if w_mode == "both" else 2
        masks = [it[mask_idx].to(dtype=torch.int32) for it in items]
        if not masks:
            continue
        count = torch.stack(masks, dim=0).sum(dim=0)
        keep = count >= thr
        for it in items:
            mask = it[mask_idx]
            out.append(it[:mask_idx] + (mask & keep.to(mask.device),))
    return out


def _thresh_tag(t: Optional[float]) -> str:
    if t is None:
        return "ratioNone"
    s = f"{t:.2f}".rstrip("0").rstrip(".")
    return "ratio" + s.replace(".", "p")


def _compute_mass_norm(
        concept_name: str,
        model_name: str,
        rank: int,
        seed: int,
        layer: int,
        feature_source: str,
        ratio_thresh: Optional[float],
        feat_ids_sorted: List[int],
        device: torch.device,
        eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-neuron (mass_norm, degree) over the requested feature ids. Cached on disk."""
    cache_dir = _mass_norm_cache_dir(model_name, concept_name, rank, seed)
    cache_dir.mkdir(parents=True, exist_ok=True)

    feats_hash = hashlib.sha1(",".join(map(str, feat_ids_sorted)).encode()).hexdigest()[:10]
    cache_path = cache_dir / (
        f"layer{layer:02d}__fs{feature_source}__{_thresh_tag(ratio_thresh)}"
        f"__nfeats{len(feat_ids_sorted)}__h{feats_hash}.npz"
    )

    if cache_path.exists():
        arr = np.load(cache_path)
        return (torch.from_numpy(arr["mass_norm"]).to(device),
                torch.from_numpy(arr["degree"]).to(device))

    nmf = _load_nmf_pickle(_mlp_pickle_path(model_name, concept_name, layer, rank, seed))
    F = nmf.F_
    if F is None:
        raise ValueError(f"SNMF F_ is None for layer={layer} concept={concept_name!r}")
    F_np = F.detach().cpu().numpy() if hasattr(F, "detach") else np.asarray(F)

    d_mlp = int(F_np.shape[0])
    mass_norm = np.zeros(d_mlp, dtype=np.float64)
    degree = np.zeros(d_mlp, dtype=np.int32)

    for feat in feat_ids_sorted:
        if feat < 0 or feat >= F_np.shape[1]:
            continue
        col = F_np[:, int(feat)]
        active_mask = col != 0
        if not np.any(active_mask):
            continue
        w_abs = np.abs(col[active_mask]).astype(np.float64)
        denom = float(np.sqrt(np.sum(w_abs * w_abs)))
        if denom <= eps:
            continue
        mass_norm[active_mask] += w_abs / (denom + eps)
        degree[active_mask] += 1

    np.savez_compressed(cache_path, mass_norm=mass_norm, degree=degree)
    return (torch.from_numpy(mass_norm).to(device),
            torch.from_numpy(degree).to(device))


def _coverage_to_k(mass_norm: torch.Tensor, degree: torch.Tensor,
                   coverage: float, eps: float = 1e-12) -> int:
    """Smallest k such that the top-k entries cover coverage fraction of total mass."""
    active = degree > 0
    vals = mass_norm[active]
    total = float(vals.sum().item())
    if total <= eps or vals.numel() == 0:
        return 0
    vals_sorted, _ = torch.sort(vals, descending=True)
    cumsum = torch.cumsum(vals_sorted, dim=0)
    target = torch.tensor(coverage * total, device=cumsum.device)
    k = int(torch.searchsorted(cumsum, target, right=False).item()) + 1
    return max(0, min(k, int(vals_sorted.numel())))


def apply_coverage_filter(
        layers_fs: List[Tuple],
        *,
        concept_name: str,
        model_name: str,
        rank: int,
        seed: int,
        feature_source: str,
        ratio_thresh: Optional[float],
        coverage_thresh: Optional[float],
        feat_ids_by_layer: Dict[int, List[int]],
        w_mode: str,
        device: torch.device,
) -> List[Tuple]:
    """Keep top-k neurons per layer that cover coverage_thresh of normalized mass."""
    if coverage_thresh is None:
        return layers_fs

    c = float(coverage_thresh)
    by_layer: Dict[int, List[Tuple]] = {}
    for item in layers_fs:
        by_layer.setdefault(int(item[0]), []).append(item)

    mask_idx = 3 if w_mode == "both" else 2
    out: List[Tuple] = []
    for L, items in by_layer.items():
        feat_ids = sorted(set(int(x) for x in feat_ids_by_layer.get(int(L), [])))
        if not feat_ids:
            out.extend(items)
            continue

        mass_norm, degree = _compute_mass_norm(
            concept_name=concept_name, model_name=model_name,
            rank=rank, seed=seed, layer=int(L),
            feature_source=feature_source, ratio_thresh=ratio_thresh,
            feat_ids_sorted=feat_ids, device=device,
        )
        k = _coverage_to_k(mass_norm=mass_norm, degree=degree, coverage=c)

        active = torch.nonzero(degree > 0, as_tuple=False).squeeze(-1)
        if active.numel() == 0 or k <= 0:
            keep = torch.zeros_like(degree, dtype=torch.bool)
        else:
            order = torch.argsort(mass_norm[active], descending=True)
            top_ids = active[order[:k]]
            keep = torch.zeros_like(degree, dtype=torch.bool)
            keep[top_ids] = True

        for it in items:
            mask = it[mask_idx]
            out.append(it[:mask_idx] + (mask & keep.to(mask.device),))
    return out


# ========================================================================== #
# Embedding feature loaders                                                   #
# ========================================================================== #

def _embedding_paths(model_name: str, concept_name: str,
                     rank: int, seed: int) -> Tuple[Path, Path]:
    """Return (embeddings.pkl, potential_features.csv) paths for the embedding track."""
    pkl = Path(get_pipeline_path(
        str(MF_OUTPUTS_ROOT), _safe_model_name(model_name), "pickles",
        rank, _safe_concept(concept_name), "embedding", "embedding.pkl", seed=seed,
    ))
    if not pkl.exists():
        raise FileNotFoundError(f"embedding SparseMatrixFactorization pickle not found: {pkl}")
    pot = Path(get_pipeline_path(
        str(MF_OUTPUTS_ROOT), _safe_model_name(model_name), "interpretations",
        rank, _safe_concept(concept_name), "embedding", "potential_features.csv", seed=seed,
    ))
    if not pot.exists():
        raise FileNotFoundError(f"embedding potential_features.csv not found: {pot}")
    return pkl, pot


def load_embedding_payload(pkl_path: Path) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Load the embedding SparseMatrixFactorization pickle.

    Returns (F_dir, G_tok, vprime_token_ids):
        F_dir : [d_model, K]  -- dense erasure directions in embedding space
        G_tok : [|V'|, K]    -- WTA-sparse signed per-token activation scores
    """
    with open(pkl_path, "rb") as f:
        payload = _pickle_load(f)
    if not (isinstance(payload, dict) and "nmf" in payload and "vprime_token_ids" in payload):
        raise ValueError(f"Unexpected embedding.pkl format at {pkl_path}")

    F_dir = payload["nmf"].F_
    G_tok = payload["nmf"].G_
    if not torch.is_tensor(F_dir):
        F_dir = torch.as_tensor(F_dir)
    if not torch.is_tensor(G_tok):
        G_tok = torch.as_tensor(G_tok)
    F_dir = F_dir.detach().cpu()
    G_tok = G_tok.detach().cpu()
    if F_dir.ndim != 2 or G_tok.ndim != 2:
        raise ValueError(f"F_/G_ expected 2D, got F={tuple(F_dir.shape)}, G={tuple(G_tok.shape)}")
    if F_dir.shape[1] != G_tok.shape[1]:
        raise ValueError(f"F/G rank mismatch: F={tuple(F_dir.shape)} G={tuple(G_tok.shape)}")
    vprime_ids = list(payload["vprime_token_ids"])
    if G_tok.shape[0] != len(vprime_ids):
        raise ValueError(f"G rows ({G_tok.shape[0]}) != len(vprime_token_ids) ({len(vprime_ids)})")
    return F_dir, G_tok, vprime_ids


def select_embed_feature_ids(df: pd.DataFrame,
                             ratio_thresh: Optional[float] = None) -> List[int]:
    """Return concept-relevant feature column indices from the embedding potential CSV.

    Falls back to all features if ratio_thresh would leave zero.
    """
    if "feature" not in df.columns:
        raise ValueError("embedding potential features CSV missing 'feature'")

    all_ids = sorted(set(pd.to_numeric(df["feature"], errors="coerce").dropna().astype(int).tolist()))

    if ratio_thresh is None:
        return all_ids
    if "metric_score" not in df.columns:
        raise ValueError("embedding potential features CSV missing 'metric_score'")

    mask = pd.to_numeric(df["metric_score"], errors="coerce") >= float(ratio_thresh)
    filtered_ids = sorted(set(
        pd.to_numeric(df[mask]["feature"], errors="coerce").dropna().astype(int).tolist()
    ))
    if filtered_ids:
        return filtered_ids
    log.warning("embed ratio_thresh=%.2f left 0 features; falling back to all %d",
                ratio_thresh, len(all_ids))
    return all_ids


def load_token_label_map(model_name: str, concept_name: str,
                         rank: int, seed: int) -> Optional[Dict[str, str]]:
    """Map {safe_token_str: concept_label} from the embedding token CSV.

    Returns None if the CSV is missing; callers should fall back to editing
    all v'-tokens.
    """
    csv_path = Path(get_pipeline_path(
        str(MF_OUTPUTS_ROOT), _safe_model_name(model_name), "csvs",
        rank, _safe_concept(concept_name), "embedding", "token_features.csv", seed=seed,
    ))
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        tokens = ast.literal_eval(row["activating_tokens"])
        labels = ast.literal_eval(row["labels"])
        for t, lbl in zip(tokens, labels):
            out.setdefault(t, lbl)
    return out or None


# ========================================================================== #
# ConceptContext: per-concept cached SNMF data                                #
# ========================================================================== #

@dataclass
class ConceptContext:
    """Per-concept cached data shared across grid / validate / test.

    Usage::

        ctx = ConceptContext(model_name="...", concept_name="Harry Potter",
                             rank=100, seed=42, ratio_thresh=2.0,
                             coverage_thresh=0.95, feature_source="all")
        ctx.load()
        layers_fs = ctx.get_layer_features(model, layer_lo=0, layer_hi=8, w_mode="out")

    The cache in _layer_cache is keyed by (layer_lo, layer_hi, w_mode). First
    call per key loads SNMF pickles and applies filters; subsequent calls return
    the cached list directly.
    """
    model_name: str
    concept_name: str
    rank: int
    seed: int
    feature_source: str = "all"
    ratio_thresh: Optional[float] = None
    coverage_thresh: Optional[float] = None
    neurons_thresh: Optional[int] = None

    df_pot: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    specs_all: List[Tuple[int, int]] = field(init=False, default_factory=list)
    _layer_cache: Dict[Tuple[int, int, str], List[Tuple]] = field(
        init=False, default_factory=dict)

    def load(self) -> None:
        """Load the potential-features CSV and select layer/feature specs. Idempotent."""
        if self.specs_all:
            return
        log.info("loading MLP features for concept=%r rank=%d", self.concept_name, self.rank)
        self.df_pot = load_mlp_potential_df(
            self.model_name, self.concept_name, self.rank, self.seed,
            ratio_thresh=self.ratio_thresh,
        )
        self.specs_all = select_mlp_specs(self.df_pot, self.feature_source)
        log.info("found %d (layer, feature) specs (feature_source=%s, ratio_thresh=%s)",
                 len(self.specs_all), self.feature_source, self.ratio_thresh)

    def get_layer_features(self, model: Any, layer_lo: int, layer_hi: int,
                           w_mode: str) -> List[Tuple]:
        """Build (or recall) layer-feature tensors for the requested range + w_mode."""
        if w_mode not in VALID_W_MODES:
            raise ValueError(f"Unknown w_mode {w_mode!r}")
        if not self.specs_all:
            self.load()

        key = (int(layer_lo), int(layer_hi), w_mode)
        if key in self._layer_cache:
            return self._layer_cache[key]

        specs = [t for t in self.specs_all if layer_lo <= t[0] <= layer_hi]
        if not specs:
            self._layer_cache[key] = []
            return []

        device = next(model.parameters()).device
        layers_fs = build_layer_features(
            model, self.model_name, self.concept_name,
            specs, self.rank, self.seed, w_mode,
        )
        layers_fs = apply_neurons_filter(layers_fs, w_mode, self.neurons_thresh)
        if self.coverage_thresh is not None:
            feat_by_layer: Dict[int, List[int]] = {}
            for (L, feat) in specs:
                feat_by_layer.setdefault(int(L), []).append(int(feat))
            layers_fs = apply_coverage_filter(
                layers_fs,
                concept_name=self.concept_name,
                model_name=self.model_name,
                rank=self.rank, seed=self.seed,
                feature_source=self.feature_source,
                ratio_thresh=self.ratio_thresh,
                coverage_thresh=self.coverage_thresh,
                feat_ids_by_layer=feat_by_layer,
                w_mode=w_mode, device=device,
            )

        self._layer_cache[key] = layers_fs
        return layers_fs

    def clear_cache(self) -> None:
        """Drop cached layer-feature tensors (useful when switching device)."""
        self._layer_cache.clear()


def has_nonempty_mask(layers_fs: Sequence[Tuple]) -> bool:
    """True iff any item in layers_fs has a non-empty support mask."""
    if not layers_fs:
        return False
    for it in layers_fs:
        mask = it[3] if len(it) == 4 else it[2]
        if bool(mask.any().item()):
            return True
    return False


__all__ = [
    "MF_OUTPUTS_ROOT",
    "VALID_FEATURE_SOURCES", "VALID_W_MODES",
    "load_mlp_potential_df", "select_mlp_specs", "build_layer_features",
    "apply_neurons_filter", "apply_coverage_filter",
    "load_embedding_payload", "select_embed_feature_ids", "load_token_label_map",
    "_embedding_paths",
    "ConceptContext", "has_nonempty_mask",
]
