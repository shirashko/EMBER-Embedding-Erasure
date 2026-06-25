"""WMDP coherency prompts used by CRISP's coherence loss."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from ember.erasure.io import ROOT_DIR

_COHERENCY_PATH = ROOT_DIR / "data" / "wmdp" / "coherency_prompts.json"
_MIN_PROMPTS = 20


def load_wmdp_coherency_prompts(domain: str) -> List[str]:
    """Return the fixed CRISP coherency prompt list for a WMDP domain.

    Args:
        domain: ``"bio"`` or ``"cyber"``.

    Raises:
        FileNotFoundError: if ``data/wmdp/coherency_prompts.json`` is missing.
        KeyError: if ``domain`` is not present in the JSON file.
        ValueError: if fewer than 20 prompts are defined for the domain.
    """
    domain = domain.lower()
    if not _COHERENCY_PATH.exists():
        raise FileNotFoundError(
            f"Missing WMDP coherency prompts file: {_COHERENCY_PATH}"
        )
    mapping = json.loads(_COHERENCY_PATH.read_text(encoding="utf-8"))
    if domain not in mapping:
        raise KeyError(
            f"Domain {domain!r} not in {_COHERENCY_PATH}; "
            f"available: {sorted(mapping)}"
        )
    prompts = mapping[domain]
    if not isinstance(prompts, list) or len(prompts) < _MIN_PROMPTS:
        raise ValueError(
            f"Need at least {_MIN_PROMPTS} coherency prompts for {domain!r}, "
            f"got {len(prompts) if isinstance(prompts, list) else 'N/A'}"
        )
    return list(prompts[:_MIN_PROMPTS])
