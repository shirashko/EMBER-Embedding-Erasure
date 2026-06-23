"""Save unlearned model checkpoints after the final-test apply step."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ember.erasure import io, log
from ember.erasure.config import RunConfig
from ember.utils import _safe_concept, _safe_model_name


def checkpoint_dir(cfg: RunConfig, concept: str) -> Path:
    """``<root>/<method>/<model>/<concept>/`` under the repo unless ``root`` is absolute."""
    use_embed_suffix = (cfg.method != "ember") and bool(cfg.ember_step.enabled)
    method = io.method_dir_name(cfg.method, use_embed_suffix)
    root = Path(cfg.checkpoint.root)
    if not root.is_absolute():
        root = io.ROOT_DIR / root
    return root / method / _safe_model_name(cfg.model_name) / _safe_concept(concept)


def _checkpoint_populated(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.iterdir())


def save_unlearned_checkpoint(
        model: Any,
        tokenizer: Any,
        cfg: RunConfig,
        concept: str,
        *,
        hyperparameters: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Persist the post-unlearning model (and tokenizer) for ``concept``."""
    if not cfg.checkpoint.enabled:
        return None

    out_dir = checkpoint_dir(cfg, concept)
    if _checkpoint_populated(out_dir) and not cfg.overwrite:
        log.info("checkpoint already exists at %s; skip save", out_dir)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("saving unlearned checkpoint -> %s", out_dir)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "method": cfg.method,
        "model_name": cfg.model_name,
        "concept": concept,
        "rank": cfg.rank,
        "seed": cfg.seed,
        "train_eval": cfg.train_eval,
        "hyperparameters": hyperparameters or {},
    }
    (out_dir / "unlearned_checkpoints.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_dir


__all__ = ["checkpoint_dir", "save_unlearned_checkpoint"]
