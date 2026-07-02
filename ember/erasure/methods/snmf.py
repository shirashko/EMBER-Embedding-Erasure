"""SNMF erasure method: MLP-weight projection-out.

Each HP cell sweeps:
    delta_in × (layer_range_in)  ×  delta_out × (layer_range_out)

with the per-concept best embed delta (from step 1) spliced in. The feature
directions and support masks for each ``(layer_range, w_mode)`` are built
lazily and cached on a :class:`features.ConceptContext` instance, so the
SNMF pickles are loaded at most once per (range, w_mode) per concept.

``w_mode`` semantics:
    - ``"in"``:   only apply up_proj-side projections; ``out_deltas`` forced
                  to ``[0.0]``.
    - ``"out"``:  only apply down_proj-side projections; ``in_deltas`` forced
                  to ``[0.0]``.
    - ``"both"``: full cartesian product; in-side and out-side are applied
                  sequentially within one HP cell.

Layer-range constants are the publication defaults (per model). They live
here -- not in :mod:`features` -- because they encode a method-specific
prior about *where* concepts tend to be linearly represented.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from ember.erasure import embed_edit, features, io, log, mlp_edit
from ember.erasure.config import RunConfig
from ember.erasure.methods.base import Method, register

# Per-model layer ranges (lo, hi) used by the SNMF grid.
GEMMA_LAYER_RANGES_IN: List[Tuple[int, int]]  = [(0, 25), (0, 8), (0, 12)]
GEMMA_LAYER_RANGES_OUT: List[Tuple[int, int]] = [(0, 8), (9, 17), (13, 25)]
LLAMA_LAYER_RANGES_IN: List[Tuple[int, int]]  = [(0, 31), (0, 10), (0, 16)]
LLAMA_LAYER_RANGES_OUT: List[Tuple[int, int]] = [(0, 10), (11, 21), (16, 31)]


def _layer_ranges(model_name: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    if "llama" in model_name.lower():
        return LLAMA_LAYER_RANGES_IN, LLAMA_LAYER_RANGES_OUT
    return GEMMA_LAYER_RANGES_IN, GEMMA_LAYER_RANGES_OUT


def validate_layer_ranges_side(
    ranges: List[Tuple[int, int]] | List[List[int]],
    model_name: str,
    *,
    side: str,
) -> List[Tuple[int, int]]:
    """Ensure each ``(layer_lo, layer_hi)`` is a valid SNMF grid range."""
    if side not in ("in", "out"):
        raise ValueError(f"side must be 'in' or 'out', got {side!r}")
    default_in, default_out = _layer_ranges(model_name)
    allowed = set(default_in if side == "in" else default_out)
    normalized = [tuple(int(x) for x in r) for r in ranges]
    invalid = [r for r in normalized if r not in allowed]
    if invalid:
        raise ValueError(
            f"Invalid SNMF layer_ranges_{side} {invalid} for {model_name!r}; "
            f"allowed: {sorted(allowed)}"
        )
    return normalized


def validate_snmf_layer_ranges(
    ranges_in: Optional[List[List[int]]],
    ranges_out: Optional[List[List[int]]],
    model_name: str,
) -> None:
    """Validate optional SNMF in/out layer-range overrides."""
    if ranges_in is not None:
        validate_layer_ranges_side(ranges_in, model_name, side="in")
    if ranges_out is not None:
        validate_layer_ranges_side(ranges_out, model_name, side="out")


def resolve_snmf_layer_ranges(
    cfg: Any, model_name: str,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return configured SNMF layer ranges, or model defaults when unset."""
    default_in, default_out = _layer_ranges(model_name)
    ranges_in = (
        validate_layer_ranges_side(cfg.layer_ranges_in, model_name, side="in")
        if cfg.layer_ranges_in is not None else default_in
    )
    ranges_out = (
        validate_layer_ranges_side(cfg.layer_ranges_out, model_name, side="out")
        if cfg.layer_ranges_out is not None else default_out
    )
    return ranges_in, ranges_out


def _enumerate_side(deltas: List[float], ranges: List[Tuple[int, int]]) -> List[Tuple[float, int, int]]:
    """Build ``(delta, lo, hi)`` cells for one side of the grid.

    ``delta == 0.0`` collapses to a single ``(0.0, -1, -1)`` "no-op" cell
    regardless of the layer ranges; otherwise each non-zero delta crosses
    every range.
    """
    out: List[Tuple[float, int, int]] = []
    seen_zero = False
    for d in deltas:
        d = float(d)
        if d == 0.0:
            if not seen_zero:
                out.append((0.0, -1, -1))
                seen_zero = True
        else:
            for lo, hi in ranges:
                out.append((d, int(lo), int(hi)))
    return out


class SNMFMethod(Method):
    """Project SNMF feature directions out of MLP up/down projections."""

    name = "snmf"
    requires_full_reload = False

    def __init__(self) -> None:
        self._ctx: Optional[features.ConceptContext] = None
        self._ctx_concept: Optional[str] = None

    # ------------------------------------------------------------------ #
    # HP enumeration                                                     #
    # ------------------------------------------------------------------ #

    def enumerate_hps(self, common: RunConfig) -> Iterable[Dict[str, Any]]:
        cfg = common.snmf
        ranges_in, ranges_out = resolve_snmf_layer_ranges(cfg, common.model_name)

        in_deltas = cfg.in_deltas if cfg.w_mode in ("in", "both") else [0.0]
        out_deltas = cfg.out_deltas if cfg.w_mode in ("out", "both") else [0.0]

        in_cells = _enumerate_side(in_deltas, ranges_in)
        out_cells = _enumerate_side(out_deltas, ranges_out)

        for d_in, in_lo, in_hi in in_cells:
            for d_out, out_lo, out_hi in out_cells:
                yield {
                    "w_mode": cfg.w_mode,
                    "feature_source": cfg.feature_source,
                    "ratio_thresh": common.selection.ratio_thresh,
                    "coverage_thresh": common.selection.coverage_thresh,
                    "neurons_thresh": common.selection.neurons_thresh,
                    "delta_in":  d_in,  "layer_lo_in":  in_lo,  "layer_hi_in":  in_hi,
                    "delta_out": d_out, "layer_lo_out": out_lo, "layer_hi_out": out_hi,
                }

    def hp_key_columns(self) -> List[str]:
        return [
            "delta_embed",
            "delta_in", "layer_lo_in", "layer_hi_in",
            "delta_out", "layer_lo_out", "layer_hi_out",
        ]

    def hp_columns(self) -> List[str]:
        return io.EMBED_COLUMNS + io.SNMF_HP_COLUMNS

    # ------------------------------------------------------------------ #
    # Per-concept setup / teardown                                       #
    # ------------------------------------------------------------------ #

    def on_concept_start(self, hf_model: Any, concept: str,
                         common: RunConfig) -> None:
        if self._ctx_concept != concept or self._ctx is None:
            self._ctx = features.ConceptContext(
                model_name=common.model_name,
                concept_name=concept,
                rank=common.rank,
                seed=common.seed,
                feature_source=common.snmf.feature_source,
                ratio_thresh=common.selection.ratio_thresh,
                coverage_thresh=common.selection.coverage_thresh,
                neurons_thresh=common.selection.neurons_thresh,
            )
            self._ctx_concept = concept
            self._ctx.load()

    def on_concept_end(self, hf_model: Any, concept: str,
                       common: RunConfig) -> None:
        if self._ctx is not None:
            self._ctx.clear_cache()
        self._ctx = None
        self._ctx_concept = None

    # ------------------------------------------------------------------ #
    # Snapshot / restore                                                 #
    # ------------------------------------------------------------------ #

    def snapshot(self, hf_model: Any) -> Any:
        return (embed_edit.snapshot(hf_model), mlp_edit.snapshot(hf_model))

    def restore(self, hf_model: Any, snap: Any) -> None:
        if snap is None:
            return
        embed_snap, mlp_snap = snap
        embed_edit.restore(hf_model, embed_snap)
        mlp_edit.restore(hf_model, mlp_snap)

    # ------------------------------------------------------------------ #
    # Apply                                                              #
    # ------------------------------------------------------------------ #

    def apply(
            self,
            hf_model: Any,
            tokenizer: Any,
            hp: Dict[str, Any],
            concept: str,
            common: RunConfig,
    ) -> Dict[str, Any]:
        if self._ctx is None or self._ctx_concept != concept:
            self.on_concept_start(hf_model, concept, common)

        info: Dict[str, Any] = {}

        delta_embed = float(hp.get("delta_embed", 0.0))
        if delta_embed != 0.0:
            info.update(embed_edit.apply_concept_embed_edit_factored(
                hf_model=hf_model,
                model_name=common.model_name,
                concept_name=concept,
                delta_embed=delta_embed,
                rank=common.rank,
                seed=common.seed,
                ratio_thresh=common.selection.ratio_thresh,
            ))
        else:
            info.update({"delta_embed": 0.0, "k_features_embed": 0,
                         "n_tokens_edited": 0})

        d_in = float(hp.get("delta_in", 0.0))
        in_lo = int(hp.get("layer_lo_in", -1))
        in_hi = int(hp.get("layer_hi_in", -1))
        if d_in != 0.0 and in_lo >= 0:
            lfs_in = self._ctx.get_layer_features(hf_model, in_lo, in_hi, "in")
            if features.has_nonempty_mask(lfs_in):
                mlp_edit.apply_interventions(hf_model, lfs_in,
                                             delta=d_in, w_mode="in")
            info["k_features_mlp_in"] = len(lfs_in)
        else:
            info["k_features_mlp_in"] = 0

        d_out = float(hp.get("delta_out", 0.0))
        out_lo = int(hp.get("layer_lo_out", -1))
        out_hi = int(hp.get("layer_hi_out", -1))
        if d_out != 0.0 and out_lo >= 0:
            lfs_out = self._ctx.get_layer_features(hf_model, out_lo, out_hi, "out")
            if features.has_nonempty_mask(lfs_out):
                mlp_edit.apply_interventions(hf_model, lfs_out,
                                             delta=d_out, w_mode="out")
            info["k_features_mlp_out"] = len(lfs_out)
        else:
            info["k_features_mlp_out"] = 0

        return info


register(SNMFMethod())
