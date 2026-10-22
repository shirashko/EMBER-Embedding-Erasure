"""Load a complete frozen-SNMF scoring config.

Every key below must be present in the YAML. Missing keys are an error, so a
job does not silently pick up argparse defaults.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]

REQUIRED_KEYS = (
    "factors_dir",
    "factors_filename",
    "model_name",
    "concepts",
    "concept_path",
    "neutral_path",
    "outdir",
    "layers",
    "layer_chunk",
    "batch_size",
    "eps",
    "g_iters",
    "reg",
    "top_k",
    "k_proj",
    "judge_top_tokens",
    "confidence_thresh",
    "gemini_model",
    "max_workers",
    "skip_interpret",
    "device",
    "overwrite",
)


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _as_layers(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    raise SystemExit(f"layers must be a string such as '0-31' or null, got {value!r}")


def _as_concepts(value: Any) -> List[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise SystemExit("concepts must be a non-empty list of concept names")
    return [item.strip() for item in value]


def load_score_config(path: Path, root: Path = ROOT_DIR) -> Namespace:
    """Return the namespace ``score_dictionary`` expects. Relative paths use ``root``."""
    config_path = path.expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    payload: Dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{config_path} must be a YAML mapping")

    missing = [key for key in REQUIRED_KEYS if key not in payload]
    unknown = [key for key in payload if key not in REQUIRED_KEYS]
    if missing or unknown:
        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if unknown:
            parts.append("unknown " + ", ".join(unknown))
        raise SystemExit(f"{config_path} " + "; ".join(parts))

    return Namespace(
        factors_dir=_resolve_path(str(payload["factors_dir"]), root),
        factors_filename=str(payload["factors_filename"]),
        model_name=str(payload["model_name"]),
        concepts=_as_concepts(payload["concepts"]),
        concept_path=_resolve_path(str(payload["concept_path"]), root),
        neutral_path=str(_resolve_path(str(payload["neutral_path"]), root)),
        outdir=_resolve_path(str(payload["outdir"]), root),
        layers=_as_layers(payload["layers"]),
        layer_chunk=int(payload["layer_chunk"]),
        batch_size=int(payload["batch_size"]),
        eps=float(payload["eps"]),
        g_iters=int(payload["g_iters"]),
        reg=float(payload["reg"]),
        top_k=int(payload["top_k"]),
        k_proj=int(payload["k_proj"]),
        judge_top_tokens=int(payload["judge_top_tokens"]),
        confidence_thresh=float(payload["confidence_thresh"]),
        gemini_model=str(payload["gemini_model"]),
        max_workers=int(payload["max_workers"]),
        skip_interpret=bool(payload["skip_interpret"]),
        device=str(payload["device"]),
        overwrite=bool(payload["overwrite"]),
    )
