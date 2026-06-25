#!/usr/bin/env python3
"""Download WMDP + MMLU MCQ files for CRISP-style evaluation.

Writes CRISP ``{questions, answers, choices}`` JSON under ``data/wmdp/``::

    bio/bio_mcq.json
    bio/high_school_bio_mcq.json
    bio/college_bio_mcq.json
    cyber/cyber_mcq.json
    cyber/high_school_computer_science_mcq.json
    cyber/college_computer_science_mcq.json

Example::

    python scripts/prepare_wmdp_mcq.py --domain all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ember.wmdp.corpora import WMDP_DOMAINS  # noqa: E402
from ember.wmdp.mcq import rows_to_crisp, write_crisp_mcq_json  # noqa: E402
from ember.wmdp.paths import (  # noqa: E402
    WMDP_AUXILIARY_MCQ,
    WMDP_PRIMARY_MCQ,
    mcq_json_path,
    resolve_data_root,
)

_WMDP_HF_CONFIG = {
    "bio": "wmdp-bio",
    "cyber": "wmdp-cyber",
}
_MMLU_HF_SUBJECT = {
    "high_school_bio_mcq.json": "high_school_biology",
    "college_bio_mcq.json": "college_biology",
    "high_school_computer_science_mcq.json": "high_school_computer_science",
    "college_computer_science_mcq.json": "college_computer_science",
}


def _hf_rows_wmdp(domain: str) -> List[Dict[str, object]]:
    from datasets import load_dataset

    config = _WMDP_HF_CONFIG[domain]
    ds = load_dataset("cais/wmdp", config, split="test")
    rows: List[Dict[str, object]] = []
    for row in ds:
        rows.append(dict(row))
    return rows


def _hf_rows_mmlu(subject: str) -> List[Dict[str, object]]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", subject, split="test")
    rows: List[Dict[str, object]] = []
    for row in ds:
        choices = row.get("choices")
        answer = row.get("answer")
        rows.append({
            "question": row.get("question", ""),
            "choices": choices,
            "answer": answer,
        })
    return rows


def _export_set(
        data_root: Path,
        domain: str,
        filename: str,
        rows: Sequence[Dict[str, object]],
        *,
        force: bool,
) -> Dict[str, object]:
    out = mcq_json_path(data_root, domain, filename)
    if out.exists() and not force:
        return {"path": str(out), "n_rows": len(json.loads(out.read_text())["questions"]), "skipped": True}

    questions, answers, choices = rows_to_crisp(rows)
    if not questions:
        raise ValueError(f"No MCQ rows exported for {domain}/{filename}")
    n = write_crisp_mcq_json(out, questions, answers, choices)
    return {"path": str(out), "n_rows": n, "skipped": False}


def prepare_domain(domain: str, data_root: Path, *, force: bool = False) -> Dict[str, object]:
    domain = domain.lower()
    if domain not in WMDP_DOMAINS:
        raise ValueError(f"domain must be one of {sorted(WMDP_DOMAINS)}, got {domain!r}")

    report: Dict[str, object] = {"domain": domain, "files": {}}
    files: Dict[str, object] = {}

    wmdp_rows = _hf_rows_wmdp(domain)
    primary_name = WMDP_PRIMARY_MCQ[domain]
    files[primary_name] = _export_set(
        data_root, domain, primary_name, wmdp_rows, force=force,
    )

    for aux_name in WMDP_AUXILIARY_MCQ[domain]:
        subject = _MMLU_HF_SUBJECT[aux_name]
        mmlu_rows = _hf_rows_mmlu(subject)
        files[aux_name] = _export_set(
            data_root, domain, aux_name, mmlu_rows, force=force,
        )

    report["files"] = files
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WMDP MCQ JSON for evaluation.")
    parser.add_argument(
        "--domain", default="all",
        help="bio, cyber, or all (default: all)",
    )
    parser.add_argument("--data-root", default="data/wmdp")
    parser.add_argument("--force", action="store_true", help="Overwrite existing JSON")
    args = parser.parse_args()

    if args.domain.lower() == "all":
        domains = sorted(WMDP_DOMAINS)
    else:
        domains = [d.strip().lower() for d in args.domain.split(",") if d.strip()]

    root = resolve_data_root(args.data_root)
    if len(domains) == 1:
        report = prepare_domain(domains[0], root, force=args.force)
    else:
        report = {d: prepare_domain(d, root, force=args.force) for d in domains}

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
