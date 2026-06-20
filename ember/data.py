"""Fetch precomputed concept features from the Hugging Face dataset.

The erasure pipeline reads factorizations and interpretations from ``mf_outputs/``
(see :data:`ember.erasure.features.MF_OUTPUTS_ROOT`). This module populates that
directory on demand from the ``ClSu/ember-features`` dataset, so a run does not
require training the features locally.

Downloads are scoped to the run's model, rank, seed and concepts, and
``snapshot_download`` skips files already present, so the call is cheap when the
features are already on disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ember.utils import _safe_model_name, _safe_concept
from ember.erasure import log

DATASET_REPO = "ClSu/ember-features"
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "mf_outputs"


def ensure_features(model_name: str, concepts: Iterable[str], rank: int, seed: int,
                    root: Optional[Path] = None,
                    repo_id: str = DATASET_REPO) -> None:
    """Download the features for ``concepts`` into ``root`` from the HF dataset.

    Scoped to ``model_name`` / ``rank`` / ``seed`` and the given concepts (pickles,
    interpretations and csvs). Idempotent: files already present are skipped.
    """
    root = Path(root) if root is not None else DEFAULT_ROOT
    model_safe = _safe_model_name(model_name)
    patterns = [
        f"{model_safe}/*/rank{rank}/seed{seed}/{_safe_concept(c)}/**"
        for c in concepts
    ]

    from huggingface_hub import snapshot_download
    log.info("ensuring features for %d concept(s) of %s (rank%d, seed%d) from %s",
             len(patterns), model_name, rank, seed, repo_id)
    snapshot_download(repo_id=repo_id, repo_type="dataset",
                      local_dir=str(root), allow_patterns=patterns)


__all__ = ["ensure_features", "DATASET_REPO", "DEFAULT_ROOT"]
