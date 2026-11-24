#!/usr/bin/env python3
"""Write test harmonic, specificity, and efficacy for saved unlearning checkpoints.

Reads ``unlearned_checkpoints/**/evaluation/evaluation_summary.json``. CRISP,
SNMF, and RMU scores come from the multiple-choice test (``final_test_mc``).
PISCES scores come from the open-question test (``final_test_open``).

Example::

    python scripts/build_unlearning_test_table.py
    python scripts/build_unlearning_test_table.py --out results/unlearning_test_table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]

# CRISP, SNMF, and RMU are selected on multiple choice. PISCES is selected on
# open questions, so its reported test scores use that split.
TEST_SPLIT_BY_METHOD = {
    "crisp": "mc",
    "snmf": "mc",
    "rmu": "mc",
    "pisces": "open",
}

CONCEPT_ORDER = (
    "Ancient Rome",
    "Golf",
    "Uranium",
    "Culture of Greece",
    "Cannabis",
    "Baseball",
    "Gambling",
    "Republic of Ireland",
    "wmdp-bio",
    "wmdp-cyber",
)

COLUMNS = (
    "Base Model",
    "Target Concept",
    "Unlearning Method",
    "Harmonic Score (on test)",
    "Specificity (on test)",
    "Efficacy (on test)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a test-score table from unlearned checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("unlearned_checkpoints"),
        help="Root directory containing saved checkpoints.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/unlearning_test_table.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _base_method(method: str) -> str:
    name = str(method).lower()
    if name.endswith("_ef"):
        return name[:-3]
    return name


def _norm_concept(name: str) -> str:
    return " ".join(str(name).replace("_", " ").split()).casefold()


def _concept_label(name: str) -> str:
    folded = _norm_concept(name)
    for concept in CONCEPT_ORDER:
        if _norm_concept(concept) == folded:
            return concept
    return " ".join(str(name).replace("_", " ").split())


def _test_row(summary: Dict[str, Any], method: str) -> Dict[str, Any]:
    split = TEST_SPLIT_BY_METHOD[method]
    test = summary.get("test") or {}
    rows = test.get(f"final_test_{split}") or []
    return rows[0] if rows else {}


def _load_rows(checkpoints_root: Path) -> List[Dict[str, str]]:
    root = checkpoints_root if checkpoints_root.is_absolute() else ROOT_DIR / checkpoints_root
    rows: List[Dict[str, str]] = []
    for path in sorted(root.glob("**/evaluation/evaluation_summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] failed to read {path}: {exc}")
            continue
        if not isinstance(summary, dict):
            continue
        method = _base_method(str(summary.get("method", "")))
        if method not in TEST_SPLIT_BY_METHOD:
            continue
        test_row = _test_row(summary, method)
        if not test_row:
            print(f"[warn] no final_test_{TEST_SPLIT_BY_METHOD[method]} row in {path}")
            continue
        rows.append({
            "Base Model": str(summary.get("model_name", "")).strip(),
            "Target Concept": _concept_label(str(summary.get("concept", "")).strip()),
            "Unlearning Method": method,
            "Harmonic Score (on test)": _fmt(_to_float(test_row.get("harmonic"))),
            "Specificity (on test)": _fmt(_to_float(test_row.get("specificity"))),
            "Efficacy (on test)": _fmt(_to_float(test_row.get("efficacy"))),
        })
    return rows


def _sort_key(row: Dict[str, str]) -> tuple:
    harmonic = row["Harmonic Score (on test)"]
    # Highest harmonic first within each model. Blank scores sort last.
    harmonic_key = -float(harmonic) if harmonic else float("inf")
    return (
        row["Base Model"].casefold(),
        harmonic_key,
        row["Target Concept"].casefold(),
        row["Unlearning Method"],
    )


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.checkpoints_root)
    rows.sort(key=_sort_key)
    out_path = args.out if args.out.is_absolute() else ROOT_DIR / args.out
    _write_csv(out_path, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
