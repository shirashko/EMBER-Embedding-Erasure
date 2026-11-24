"""Concept and neutral sentences used to score a frozen SNMF dictionary."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

from ember.utils import _safe_concept


def available_concepts(concept_path: Path) -> List[str]:
    """Concept names stored in ``concept_sentences.json``, in file order."""
    data = json.loads(concept_path.read_text(encoding="utf-8"))
    names = []
    for entry in data:
        name = entry.get("concept") or entry.get("Concept")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def resolve_concept_name(raw: str, names: Sequence[str]) -> str:
    """Return the stored concept name, or exit if it is not supported."""
    if raw in names:
        return raw
    key = _safe_concept(raw).lower()
    hits = [name for name in names if _safe_concept(name).lower() == key]
    if len(hits) == 1:
        return hits[0]
    supported = "\n".join(f"  - {name}" for name in names)
    raise SystemExit(
        f"Concept {raw!r} is not supported. Pass one of the concepts in the concept sentences:\n{supported}"
    )


def select_concepts(requested: Sequence[str], concept_path: Path) -> List[str]:
    """Validate ``requested`` against the repo concepts and drop duplicates."""
    names = available_concepts(concept_path)
    if not names:
        raise SystemExit(f"No concepts found in {concept_path}")
    chosen: List[str] = []
    for raw in requested:
        name = resolve_concept_name(raw, names)
        if name not in chosen:
            chosen.append(name)
    return chosen


def concept_sentences(concept_path: Path, concept_name: str) -> List[str]:
    data = json.loads(concept_path.read_text(encoding="utf-8"))
    for entry in data:
        name = entry.get("concept") or entry.get("Concept")
        if isinstance(name, str) and name.strip() == concept_name:
            sentences = [s for s in entry.get("sentences", []) if isinstance(s, str) and s.strip()]
            if not sentences:
                break
            return sentences
    raise SystemExit(f"No sentences stored for {concept_name!r} in {concept_path}")


def neutral_sentences(neutral_path: Path) -> List[str]:
    data = json.loads(neutral_path.read_text(encoding="utf-8"))
    sentences = [obj["sentence"] for obj in data if isinstance(obj, dict) and obj.get("sentence")]
    if not sentences:
        raise SystemExit(f"No neutral sentences in {neutral_path}")
    return sentences
