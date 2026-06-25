"""CRISP/WMDP document preprocessing (paper §4.1).

Applied once by :mod:`scripts.prepare_wmdp_corpora` when building cleaned JSONL
artifacts. Training runs load the prepared files without re-processing.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# CRISP paper defaults
DEFAULT_MAX_LEN = 1000
DEFAULT_BIO_N_EXAMPLES = 5000

# Markdown / link patterns (best-effort; official CAIS corpora may already be clean).
_MD_HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_INLINE_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")
_CITATION_BRACKET_RE = re.compile(r"\[[0-9,\s\-]+\]")
_CITATION_PAREN_RE = re.compile(r"\([0-9]{4}[a-z]?\)")


def clean_document(text: str) -> str:
    """Remove common formatting artifacts and normalize whitespace."""
    text = str(text)
    text = _IMAGE_LINK_RE.sub("", text)
    text = _INLINE_LINK_RE.sub("", text)
    text = _MD_HEADER_RE.sub("", text)
    text = _CITATION_BRACKET_RE.sub("", text)
    text = _CITATION_PAREN_RE.sub("", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def right_truncate(text: str, max_len: int) -> str:
    """Right-truncate to ``max_len`` characters (CRISP paper)."""
    if max_len <= 0:
        return text
    return text[:max_len]


def preprocess_document(text: str, *, max_len: int = DEFAULT_MAX_LEN) -> str:
    """Full per-document pipeline: clean then right-truncate."""
    cleaned = clean_document(text)
    if not cleaned:
        return ""
    return right_truncate(cleaned, max_len)


def preprocess_corpus(
        texts: Sequence[str],
        *,
        max_len: int = DEFAULT_MAX_LEN,
) -> List[str]:
    """Preprocess many documents, dropping empties after cleaning."""
    out: List[str] = []
    for text in texts:
        doc = preprocess_document(text, max_len=max_len)
        if doc:
            out.append(doc)
    return out


def sample_corpus(
        texts: Sequence[str],
        *,
        seed: int,
        n_examples: Optional[int],
) -> List[str]:
    """Shuffle and optionally cap to ``n_examples`` (fixed for reproducibility)."""
    out = list(texts)
    rnd = random.Random(seed)
    rnd.shuffle(out)
    if n_examples is not None and n_examples > 0:
        out = out[:n_examples]
    return out


def write_jsonl(path: Path, texts: Iterable[str]) -> int:
    """Write ``{"text": ...}`` rows; return row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for text in texts:
            fh.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")
            count += 1
    return count


def read_jsonl_texts(path: Path) -> List[str]:
    texts: List[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return texts


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
