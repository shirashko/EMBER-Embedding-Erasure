"""HuggingFace MLP-weight erasure primitives.

Projects concept feature directions out of MLP up/down projection matrices.
Erasure math:
    out: down_proj.weight[:, idx] -= delta * f ⊗ (f @ W_cols)
    in:  up_proj.weight[idx, :]   -= delta * (W_rows @ f) ⊗ f
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union

import torch

_F_EPS = 1e-8


def _mlp_layer(hf_model: Any, layer_idx: int) -> torch.nn.Module:
    try:
        return hf_model.model.layers[layer_idx].mlp
    except (AttributeError, IndexError) as e:
        raise ValueError(
            f"HF model {type(hf_model).__name__} has no MLP at layer {layer_idx}: {e}"
        )


def _n_layers(hf_model: Any) -> int:
    return len(hf_model.model.layers)


def _normalize(f: torch.Tensor) -> torch.Tensor:
    return f / (f.norm() + _F_EPS)


# ========================================================================== #
# Snapshot / restore                                                          #
# ========================================================================== #

def snapshot(hf_model: Any) -> List[Dict[str, Any]]:
    """Return a CPU snapshot of every layer's up_proj and down_proj weight."""
    snaps: List[Dict[str, Any]] = []
    with torch.no_grad():
        for layer_idx in range(_n_layers(hf_model)):
            mlp = _mlp_layer(hf_model, layer_idx)
            snaps.append({
                "layer": layer_idx,
                "up_proj":   mlp.up_proj.weight.data.detach().cpu().clone(),
                "down_proj": mlp.down_proj.weight.data.detach().cpu().clone(),
            })
    return snaps


def restore(hf_model: Any, snaps: Sequence[Dict[str, Any]]) -> None:
    """Restore the up/down weights from a snapshot."""
    with torch.no_grad():
        for entry in snaps:
            mlp = _mlp_layer(hf_model, entry["layer"])
            mlp.up_proj.weight.data.copy_(
                entry["up_proj"].to(mlp.up_proj.weight.device,
                                    dtype=mlp.up_proj.weight.dtype)
            )
            mlp.down_proj.weight.data.copy_(
                entry["down_proj"].to(mlp.down_proj.weight.device,
                                      dtype=mlp.down_proj.weight.dtype)
            )


def snapshot_down_only(hf_model: Any) -> List[Dict[str, Any]]:
    """Snapshot only down_proj weights (faster variant for nested in/out grid)."""
    snaps: List[Dict[str, Any]] = []
    with torch.no_grad():
        for layer_idx in range(_n_layers(hf_model)):
            mlp = _mlp_layer(hf_model, layer_idx)
            snaps.append({
                "layer": layer_idx,
                "down_proj": mlp.down_proj.weight.data.detach().cpu().clone(),
            })
    return snaps


def restore_down_only(hf_model: Any, snaps: Sequence[Dict[str, Any]]) -> None:
    """Restore only down_proj weights from snapshot_down_only."""
    with torch.no_grad():
        for entry in snaps:
            mlp = _mlp_layer(hf_model, entry["layer"])
            mlp.down_proj.weight.data.copy_(
                entry["down_proj"].to(mlp.down_proj.weight.device,
                                      dtype=mlp.down_proj.weight.dtype)
            )


# ========================================================================== #
# Single-layer intervention                                                   #
# ========================================================================== #

def intervene(hf_model: Any, layer: int, f: torch.Tensor,
              mask_dmlp: torch.Tensor, kind: str, delta: float = 1.0) -> None:
    """Project direction f out of the active neurons of one MLP layer.

    kind="out" updates down_proj columns; kind="in" updates up_proj rows.
    mask_dmlp is a boolean mask over d_mlp; only masked neurons are touched.
    """
    if kind not in ("in", "out"):
        raise ValueError(f"kind must be 'in' or 'out', got {kind!r}")
    mlp = _mlp_layer(hf_model, layer)
    weight = (mlp.down_proj if kind == "out" else mlp.up_proj).weight

    with torch.no_grad():
        f = _normalize(f.to(device=weight.device, dtype=weight.dtype))
        idx = mask_dmlp.nonzero(as_tuple=False).squeeze(-1).to(weight.device)
        if idx.numel() == 0:
            return

        if kind == "out":
            W_cols = weight.data.index_select(1, idx)   # [d_model, n_active]
            inner = f @ W_cols                           # [n_active]
            W_cols = W_cols - delta * f.unsqueeze(1) * inner.unsqueeze(0)
            weight.data.index_copy_(1, idx, W_cols)
        else:
            W_rows = weight.data.index_select(0, idx)   # [n_active, d_model]
            inner = W_rows @ f                           # [n_active]
            W_rows = W_rows - delta * inner.unsqueeze(1) * f.unsqueeze(0)
            weight.data.index_copy_(0, idx, W_rows)


# ========================================================================== #
# Multi-layer driver                                                          #
# ========================================================================== #

LayerFeatureItem = Union[
    Tuple[int, torch.Tensor, torch.Tensor],
    Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor],
]


def apply_interventions(
        hf_model: Any,
        layer_features: Sequence[LayerFeatureItem],
        delta: float,
        w_mode: str,
) -> None:
    """Apply MLP-weight projection-out across many layers.

    layer_features: output of ConceptContext.get_layer_features:
        w_mode in {"in","out"}: (layer, f, mask) tuples
        w_mode == "both":       (layer, f_out, f_in, mask) tuples
    """
    if w_mode not in ("in", "out", "both"):
        raise ValueError(f"w_mode must be 'in', 'out', or 'both'; got {w_mode!r}")

    items = list(layer_features) if not isinstance(layer_features, tuple) else [layer_features]

    for item in items:
        if w_mode == "both":
            if len(item) != 4:
                raise ValueError(
                    "layer_features must be (layer, f_out, f_in, mask) when w_mode='both'"
                )
            L, f_out, f_in, mask = item
            L = int(L)
            intervene(hf_model, layer=L, f=f_out, mask_dmlp=mask, kind="out", delta=delta)
            intervene(hf_model, layer=L, f=f_in,  mask_dmlp=mask, kind="in",  delta=delta)
        else:
            if len(item) != 3:
                raise ValueError(
                    f"layer_features must be (layer, f, mask) when w_mode={w_mode!r}"
                )
            L, f, mask = item
            intervene(hf_model, layer=int(L), f=f, mask_dmlp=mask, kind=w_mode, delta=delta)


__all__ = [
    "snapshot", "restore",
    "snapshot_down_only", "restore_down_only",
    "intervene", "apply_interventions",
]
