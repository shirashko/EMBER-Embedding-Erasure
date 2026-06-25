"""Unified forget/retain and coherency-prompt loading for unlearning methods.

WMDP training corpora must be prepared first::

    python scripts/prepare_wmdp_corpora.py --domain all
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from ember.erasure.io import ROOT_DIR
from ember.local_datasets import ConceptDataset
from ember.wmdp.coherency import load_wmdp_coherency_prompts
from ember.wmdp.corpora import WMDP_DOMAINS, load_wmdp_forget_retain

if TYPE_CHECKING:
    from ember.erasure.config import RunConfig

ForgetRetainDict = Dict[str, List[str]]

_EMBER_COHERENCY_PATH = ROOT_DIR / "data" / "coherency_prompts.json"
_MIN_COHERENCY_PROMPTS = 20


def _filter_max_len(texts: List[str], max_len: Optional[int]) -> List[str]:
    if not max_len or max_len <= 0:
        return texts
    return [t for t in texts if len(t) <= max_len]


def _load_ember_forget_retain(
        concept: str,
        *,
        seed: int,
        max_len: Optional[int],
) -> ForgetRetainDict:
    data = ConceptDataset(concept).as_forget_retain(seed=seed)
    forget = _filter_max_len(data["forget"], max_len)
    retain = _filter_max_len(data["retain"], max_len)
    return {"forget": forget, "retain": retain}


def _load_ember_coherency_prompts(concept: str) -> List[str]:
    if not _EMBER_COHERENCY_PATH.exists():
        raise FileNotFoundError(
            f"Missing EMBER coherency prompts: {_EMBER_COHERENCY_PATH}"
        )
    mapping = json.loads(_EMBER_COHERENCY_PATH.read_text(encoding="utf-8"))
    if concept not in mapping:
        raise KeyError(
            f"Concept {concept!r} not in {_EMBER_COHERENCY_PATH}; "
            f"available: {sorted(mapping)}"
        )
    prompts = mapping[concept]
    if not isinstance(prompts, list) or len(prompts) < _MIN_COHERENCY_PROMPTS:
        raise ValueError(
            f"Need at least {_MIN_COHERENCY_PROMPTS} coherency prompts for "
            f"{concept!r}, got {len(prompts) if isinstance(prompts, list) else 'N/A'}"
        )
    return list(prompts[:_MIN_COHERENCY_PROMPTS])


def load_forget_retain(
        concept: str,
        cfg: RunConfig,
        *,
        max_len: Optional[int] = None,
) -> ForgetRetainDict:
    """Load forget/retain corpora for one concept or WMDP domain."""
    source = cfg.data.source.lower()
    if source == "ember":
        return _load_ember_forget_retain(concept, seed=cfg.seed, max_len=max_len)
    if source == "wmdp":
        wmdp = cfg.data.wmdp
        return load_wmdp_forget_retain(
            domain=concept.lower(),
            retain_type=wmdp.resolved_retain_type(concept),
            data_root=wmdp.data_root,
            seed=cfg.seed,
            n_examples=wmdp.n_examples,
            wiki_retain_max_len=wmdp.wiki_retain_max_len,
        )
    raise ValueError(
        f"data.source must be 'ember' or 'wmdp', got {cfg.data.source!r}"
    )


def load_coherency_prompts(concept: str, cfg: RunConfig) -> List[str]:
    """Load coherency prompts for CRISP training."""
    source = cfg.data.source.lower()
    if source == "ember":
        return _load_ember_coherency_prompts(concept)
    if source == "wmdp":
        return load_wmdp_coherency_prompts(concept)
    raise ValueError(
        f"data.source must be 'ember' or 'wmdp', got {cfg.data.source!r}"
    )


__all__ = [
    "ForgetRetainDict",
    "WMDP_DOMAINS",
    "load_coherency_prompts",
    "load_forget_retain",
]
