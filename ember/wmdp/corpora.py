"""Load prepared WMDP forget/retain JSONL corpora for training.

Cleaned files are produced offline by::

    python scripts/prepare_wmdp_corpora.py --domain all

See ``data/wmdp/README.md``.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

from ember.wmdp.paths import cleaned_jsonl_path, resolve_data_root
from ember.wmdp.preprocess import read_jsonl_texts

WMDP_DOMAINS = frozenset({"bio", "cyber"})
WMDP_RETAIN_TYPES = frozenset({"wiki", "wiki-bio", "bio", "cyber"})

_DEFAULT_DATA_ROOT = resolve_data_root("data/wmdp")
_WIKI_RETAIN_MAX_LEN = 1000


def _shuffle(texts: List[str], *, seed: int) -> List[str]:
    out = list(texts)
    random.Random(seed).shuffle(out)
    return out


def _maybe_limit(texts: List[str], n_examples: Optional[int]) -> List[str]:
    if n_examples is None or n_examples <= 0:
        return texts
    return texts[:n_examples]


def _load_prepared_split(data_root: Path, domain: str, split: str) -> List[str]:
    path = cleaned_jsonl_path(data_root, domain, split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prepared WMDP corpus: {path}\n"
            "Run: python scripts/prepare_wmdp_corpora.py "
            f"--domain {domain}"
        )
    return read_jsonl_texts(path)


def _prepare_wiki_paragraphs(sentences: List[str], max_len: int) -> List[str]:
    """Chunk wiki retain text (only when retain_type is wiki-based)."""
    import re

    paragraphs: List[str] = []
    current = ""
    min_len = int(max_len * 0.8)

    for sentence in sentences:
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if not sentence:
            continue
        while len(sentence) > max_len:
            split = sentence[:max_len].rstrip().rfind(" ")
            if split == -1:
                split = max_len - 1
            if current and len(current) >= min_len:
                paragraphs.append(current)
            paragraphs.append(sentence[:split].strip())
            sentence = sentence[split:].strip()
            current = ""

        new_len = len(current) + (1 if current else 0) + len(sentence)
        if new_len <= max_len:
            current = f"{current} {sentence}".strip() if current else sentence
        elif len(current) >= min_len:
            paragraphs.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence

    if current and len(current) >= min_len:
        paragraphs.append(current)
    return paragraphs


def _load_retain_wiki(*, max_len: int) -> List[str]:
    from datasets import load_dataset

    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return _prepare_wiki_paragraphs(list(wiki["text"]), max_len=max_len)


def _load_retain_wiki_bio(*, max_len: int) -> List[str]:
    from datasets import load_dataset

    wiki_bio = load_dataset("burgerbee/biology_wiki", split="train")
    return _prepare_wiki_paragraphs(list(wiki_bio["text"]), max_len=max_len)


def _load_retain_corpus(
        retain_type: str,
        data_root: Path,
        *,
        wiki_max_len: int,
) -> List[str]:
    retain_type = retain_type.lower()
    if retain_type == "wiki":
        return _load_retain_wiki(max_len=wiki_max_len)
    if retain_type == "wiki-bio":
        return _load_retain_wiki_bio(max_len=wiki_max_len)
    if retain_type in ("bio", "cyber"):
        return _load_prepared_split(data_root, retain_type, "retain")
    raise ValueError(
        f"retain_type must be one of {sorted(WMDP_RETAIN_TYPES)}, got {retain_type!r}"
    )


def load_wmdp_forget_retain(
        domain: str,
        retain_type: str,
        *,
        data_root: str | Path = _DEFAULT_DATA_ROOT,
        seed: int = 42,
        n_examples: Optional[int] = None,
        wiki_retain_max_len: int = _WIKI_RETAIN_MAX_LEN,
) -> Dict[str, List[str]]:
    """Load prepared forget/retain lists for a training run.

    Expects cleaned JSONL artifacts under ``data_root`` (see
    :mod:`ember.wmdp.prepare_corpora`). Only shuffles for run-to-run order;
    preprocessing and sampling are fixed at prepare time.

    Args:
        domain: ``"bio"`` or ``"cyber"`` (forget set).
        retain_type: ``wiki``, ``wiki-bio``, ``bio``, or ``cyber``.
        data_root: Directory containing prepared JSONL files.
        seed: Shuffle seed (order only; corpus content is fixed).
        n_examples: Optional cap per list (debugging; deviates from paper if set).
        wiki_retain_max_len: Used only for wiki-based retain corpora.
    """
    domain = domain.lower()
    if domain not in WMDP_DOMAINS:
        raise ValueError(f"domain must be one of {sorted(WMDP_DOMAINS)}, got {domain!r}")

    root = resolve_data_root(data_root)
    forget = _load_prepared_split(root, domain, "forget")
    retain = _load_retain_corpus(retain_type, root, wiki_max_len=wiki_retain_max_len)

    forget = _maybe_limit(_shuffle(forget, seed=seed), n_examples)
    retain = _maybe_limit(_shuffle(retain, seed=seed + 1), n_examples)

    if not forget:
        raise ValueError(f"WMDP forget corpus for {domain!r} is empty")
    if not retain:
        raise ValueError(
            f"WMDP retain corpus for retain_type={retain_type!r} is empty"
        )
    return {"forget": forget, "retain": retain}
