"""Build cleaned WMDP JSONL corpora for reuse across training runs."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from ember.wmdp.corpora import WMDP_DOMAINS
from ember.wmdp.paths import (
    cleaned_jsonl_path,
    manifest_path,
    raw_jsonl_path,
    resolve_data_root,
)
from ember.wmdp.preprocess import (
    DEFAULT_BIO_N_EXAMPLES,
    DEFAULT_MAX_LEN,
    preprocess_corpus,
    read_jsonl_texts,
    sample_corpus,
    write_jsonl,
    write_manifest,
)

_WMDP_CORPORA_HF = "cais/wmdp-corpora"
_CORPORA_SUBSETS: Dict[Tuple[str, str], str] = {
    ("bio", "retain"): "bio-retain-corpus",
    ("bio", "forget"): "bio-forget-corpus",
    ("cyber", "retain"): "cyber-retain-corpus",
    ("cyber", "forget"): "cyber-forget-corpus",
}
_GATED_FORGET_HF = {
    "bio": "cais/wmdp-bio-forget-corpus",
    "cyber": "cais/wmdp-cyber-forget-corpus",
}

# Paper §4.1: 5000 bio samples; all cyber (~986 in paper, corpus may differ).
_DEFAULT_N_EXAMPLES: Dict[Tuple[str, str], Optional[int]] = {
    ("bio", "forget"): DEFAULT_BIO_N_EXAMPLES,
    ("bio", "retain"): DEFAULT_BIO_N_EXAMPLES,
    ("cyber", "forget"): None,
    ("cyber", "retain"): None,
}


def default_n_examples(domain: str, split: str) -> Optional[int]:
    return _DEFAULT_N_EXAMPLES.get((domain.lower(), split.lower()))


def _load_hf_rows(dataset_name: str, subset: Optional[str] = None, *, token: Optional[str] = None) -> List[str]:
    from datasets import load_dataset

    if subset:
        dataset = load_dataset(dataset_name, subset, split="train", token=token)
    else:
        dataset = load_dataset(dataset_name, split="train", token=token)
    texts: List[str] = []
    for row in dataset:
        text = row.get("text") if isinstance(row, dict) else None
        if text is None and hasattr(row, "get"):
            text = row.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def _load_raw_split(domain: str, split: str, data_root) -> Optional[List[str]]:
    path = raw_jsonl_path(data_root, domain, split)
    if not path.exists():
        return None
    return read_jsonl_texts(path)


def fetch_raw_corpus(
        domain: str,
        split: str,
        data_root,
        *,
        use_huggingface: bool = True,
) -> Tuple[List[str], str]:
    """Download or read raw texts before preprocessing. Returns (texts, source_label)."""
    domain = domain.lower()
    split = split.lower()
    token = os.getenv("HF_TOKEN")

    local = _load_raw_split(domain, split, data_root)
    if local is not None:
        return local, f"local:{raw_jsonl_path(data_root, domain, split)}"

    if not use_huggingface:
        raise FileNotFoundError(
            f"No raw corpus at {raw_jsonl_path(data_root, domain, split)} "
            f"and use_huggingface=False."
        )

    subset = _CORPORA_SUBSETS.get((domain, split))
    if subset:
        try:
            texts = _load_hf_rows(_WMDP_CORPORA_HF, subset, token=token)
            if texts:
                return texts, f"hf:{_WMDP_CORPORA_HF}/{subset}"
        except Exception:
            pass

    if split == "forget" and domain in _GATED_FORGET_HF:
        ds = _GATED_FORGET_HF[domain]
        if not token:
            raise EnvironmentError(
                f"HF_TOKEN is required to download gated forget corpus {ds}"
            )
        texts = _load_hf_rows(ds, token=token)
        if texts:
            return texts, f"hf:{ds}"

    raise FileNotFoundError(
        f"Could not load raw WMDP {domain}/{split} corpus. Place JSONL at "
        f"{raw_jsonl_path(data_root, domain, split)} or set HF_TOKEN and "
        f"retry with Hugging Face."
    )


def prepare_split(
        domain: str,
        split: str,
        data_root,
        *,
        seed: int = 42,
        max_len: int = DEFAULT_MAX_LEN,
        n_examples: Optional[int] = None,
        use_huggingface: bool = True,
        force: bool = False,
) -> Dict[str, object]:
    """Preprocess one split and write ``{domain}_{split}_dataset_cleaned.jsonl``."""
    domain = domain.lower()
    split = split.lower()
    root = resolve_data_root(data_root)
    out_path = cleaned_jsonl_path(root, domain, split)

    if out_path.exists() and not force:
        existing = read_jsonl_texts(out_path)
        return {
            "domain": domain,
            "split": split,
            "path": str(out_path),
            "n_rows": len(existing),
            "skipped": True,
        }

    if n_examples is None:
        n_examples = default_n_examples(domain, split)

    raw_texts, source = fetch_raw_corpus(
        domain, split, root, use_huggingface=use_huggingface,
    )
    processed = preprocess_corpus(raw_texts, max_len=max_len)
    if not processed:
        raise ValueError(f"No documents left after preprocessing {domain}/{split}")

    # Use distinct seeds per split so forget/retain shuffles are independent.
    split_seed = seed if split == "forget" else seed + 1
    final = sample_corpus(processed, seed=split_seed, n_examples=n_examples)
    n_written = write_jsonl(out_path, final)

    return {
        "domain": domain,
        "split": split,
        "path": str(out_path),
        "source": source,
        "n_raw": len(raw_texts),
        "n_after_preprocess": len(processed),
        "n_rows": n_written,
        "max_len": max_len,
        "n_examples": n_examples,
        "seed": split_seed,
        "skipped": False,
    }


def prepare_domain(
        domain: str,
        data_root,
        *,
        seed: int = 42,
        max_len: int = DEFAULT_MAX_LEN,
        forget_n_examples: Optional[int] = None,
        retain_n_examples: Optional[int] = None,
        use_huggingface: bool = True,
        force: bool = False,
) -> Dict[str, object]:
    """Prepare forget + retain cleaned JSONL for one domain."""
    domain = domain.lower()
    if domain not in WMDP_DOMAINS:
        raise ValueError(f"domain must be one of {sorted(WMDP_DOMAINS)}, got {domain!r}")

    # sample 5000 for bio only; cyber uses the full corpus.
    if domain != "bio":
        forget_n_examples = None
        retain_n_examples = None

    root = resolve_data_root(data_root)
    results = {
        "forget": prepare_split(
            domain, "forget", root,
            seed=seed, max_len=max_len,
            n_examples=forget_n_examples,
            use_huggingface=use_huggingface,
            force=force,
        ),
        "retain": prepare_split(
            domain, "retain", root,
            seed=seed, max_len=max_len,
            n_examples=retain_n_examples,
            use_huggingface=use_huggingface,
            force=force,
        ),
    }

    manifest = {
        "domain": domain,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "max_len": max_len,
        "seed": seed,
        "splits": results,
    }
    write_manifest(manifest_path(root, domain), manifest)
    return manifest


def prepare_all(
        domains: Sequence[str],
        data_root,
        **kwargs,
) -> Dict[str, object]:
    return {d: prepare_domain(d, data_root, **kwargs) for d in domains}
