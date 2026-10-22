"""Locate and load a frozen SNMF dictionary (``layer_*/snmf_factors.pt``)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch


def parse_layer_spec(spec: Optional[str]) -> Optional[List[int]]:
    """Parse ``0-5,10`` into a sorted list of layer indices. ``None`` means every layer."""
    if not spec:
        return None
    layers = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(chunk))
    return sorted(set(layers))


def discover_layers(
    factors_dir: Path,
    filename: str,
    requested: Optional[Sequence[int]],
) -> List[Tuple[int, Path]]:
    """Return ``(layer, checkpoint path)`` pairs that exist under ``factors_dir``."""
    found: Dict[int, Path] = {}
    for path in factors_dir.glob("layer_*/" + filename):
        suffix = path.parent.name.split("_", 1)[-1]
        if suffix.isdigit():
            found[int(suffix)] = path
    if not found:
        raise SystemExit(f"No layer_*/{filename} checkpoints under {factors_dir}")
    layers = sorted(found)
    if requested is not None:
        missing = [layer for layer in requested if layer not in found]
        if missing:
            raise SystemExit(f"Requested layers not in {factors_dir}: {missing}")
        layers = [layer for layer in layers if layer in set(requested)]
    return [(layer, found[layer]) for layer in layers]


def _site_from_blob(blob: dict, path: Path) -> str:
    site = blob.get("mode")
    if not isinstance(site, str) or not site.strip():
        nested = blob.get("config")
        if isinstance(nested, dict):
            site = (nested.get("snmf") or {}).get("mode")
    if not isinstance(site, str) or not site.strip():
        raise SystemExit(f"{path} does not record an activation site (mode)")
    return site.strip()


def dictionary_site(factors_dir: Path) -> str:
    """Activation site recorded in ``config.json`` (for example ``mlp_intermediate``)."""
    config_path = factors_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"No config.json under {factors_dir}; cannot tell which activation site was factorized")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    site = (payload.get("snmf") or {}).get("mode") if isinstance(payload, dict) else None
    if not isinstance(site, str) or not site.strip():
        raise SystemExit(f"{config_path} has no snmf.mode")
    return site.strip()


def load_dictionary(path: Path) -> Tuple[torch.Tensor, str]:
    """Return ``F`` with shape ``(d_mlp, K)`` and the checkpoint's activation site."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "F" not in blob:
        raise SystemExit(f"{path} has no 'F' tensor")
    factors = torch.as_tensor(blob["F"]).detach().float().cpu()
    if factors.ndim != 2:
        raise SystemExit(f"{path} F has shape {tuple(factors.shape)}; expected a matrix")
    # Training saves F as (d_mlp, K). A swapped matrix would have the rank on axis 0.
    if factors.shape[0] < factors.shape[1]:
        factors = factors.T.contiguous()
    return factors, _site_from_blob(blob, path)


def infer_model_name(factors_dir: Path, explicit: Optional[str]) -> str:
    """Use the CLI model, else ``model_path`` in the factors ``config.json``."""
    if explicit:
        return explicit
    config_path = factors_dir / "config.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        model_path = payload.get("model_path") or payload.get("model_name")
        if isinstance(model_path, str) and model_path.strip():
            return model_path.strip()
    return "meta-llama/Llama-3.1-8B-Instruct"
