#!/usr/bin/env python3
"""Prepare cleaned WMDP JSONL corpora for CRISP / EMBER training runs.

Run once before unlearning experiments. Output files are reused across runs::

    data/wmdp/bio/bio_forget_dataset_cleaned.jsonl
    data/wmdp/bio/bio_retain_dataset_cleaned.jsonl
    data/wmdp/cyber/cyber_forget_dataset_cleaned.jsonl
    data/wmdp/cyber/cyber_retain_dataset_cleaned.jsonl

Example::

    export HF_TOKEN=...   # required for gated bio forget corpus
    python scripts/prepare_wmdp_corpora.py --domain all --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ember.wmdp.corpora import WMDP_DOMAINS  # noqa: E402
from ember.wmdp.prepare_corpora import prepare_domain  # noqa: E402
from ember.wmdp.preprocess import DEFAULT_BIO_N_EXAMPLES, DEFAULT_MAX_LEN  # noqa: E402


def _parse_domains(value: str) -> list[str]:
    if value.lower() == "all":
        return sorted(WMDP_DOMAINS)
    domains = [d.strip().lower() for d in value.split(",") if d.strip()]
    bad = [d for d in domains if d not in WMDP_DOMAINS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown domain(s) {bad}; expected bio, cyber, or all"
        )
    return domains


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cleaned WMDP JSONL corpora (CRISP paper preprocessing).",
    )
    parser.add_argument(
        "--domain", type=_parse_domains, default="all",
        help="bio, cyber, or all (default: all)",
    )
    parser.add_argument(
        "--data-root", default="data/wmdp",
        help="Output root for prepared JSONL (default: data/wmdp)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-len", type=int, default=DEFAULT_MAX_LEN,
        help="Right-truncate length in characters (default: 1000)",
    )
    parser.add_argument(
        "--bio-n-examples", type=int, default=DEFAULT_BIO_N_EXAMPLES,
        help="Sample size for bio forget/retain only (default: 5000; cyber uses all rows)",
    )
    parser.add_argument(
        "--no-hf", action="store_true",
        help="Only read local raw JSONL under data/wmdp/raw/ (no Hugging Face)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing cleaned JSONL files",
    )
    args = parser.parse_args()

    kwargs = dict(
        seed=args.seed,
        max_len=args.max_len,
        use_huggingface=not args.no_hf,
        force=args.force,
    )

    if len(args.domain) == 1:
        domain = args.domain[0]
        if domain == "bio":
            kwargs["forget_n_examples"] = args.bio_n_examples
            kwargs["retain_n_examples"] = args.bio_n_examples
        report = prepare_domain(domain, args.data_root, **kwargs)
    else:
        report = {}
        for domain in args.domain:
            domain_kwargs = dict(kwargs)
            if domain == "bio":
                domain_kwargs["forget_n_examples"] = args.bio_n_examples
                domain_kwargs["retain_n_examples"] = args.bio_n_examples
            report[domain] = prepare_domain(domain, args.data_root, **domain_kwargs)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
