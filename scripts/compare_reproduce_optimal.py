#!/usr/bin/env python3
"""Compare reproduced checkpoint scores against optimal reference YAML.

Reads:
  - configs/reproduce_optimal_configs/optimal_unlearning_hyperparams.yaml
  - unlearned_checkpoints/**/evaluation/evaluation_summary.json

Writes:
  - comparison_summary.csv      (one row per expected checkpoint)
  - comparison_detailed.csv     (one row per compared metric)

Use this to quickly see what matched and what did not.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


MODEL_FAMILY_TO_NAME = {
    "Gemma": "google/gemma-2-2b-it",
    "Llama": "meta-llama/Llama-3.1-8B-Instruct",
}

TRAIN_EVAL_BY_METHOD = {
    "pisces": "open",
    "rmu": "mc",
    "crisp": "mc",
    "snmf": "mc",
}

ALPACA_METRICS = {
    "alpaca_instr",
    "alp_instr_frac",
    "alpaca_flu",
    "alp_flu_frac",
    "coherence",
    "harmonic_alpaca",
}

OPEN_JUDGE_DEPENDENT_METRICS = {
    "qa_acc",
    "qa_frac",
    "simdom_acc",
    "simdom_frac",
    "efficacy",
    "specificity",
    "harmonic",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare reproduced checkpoint metrics to optimal reference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--optimal-yaml",
        type=Path,
        default=Path("configs/reproduce_optimal_configs/optimal_unlearning_hyperparams.yaml"),
        help="Reference YAML with expected hyperparameters/evaluation.",
    )
    p.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("unlearned_checkpoints"),
        help="Root containing checkpoint evaluation_summary.json files.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/reproduce_optimal_compare"),
        help="Output directory for comparison CSV files.",
    )
    p.add_argument(
        "--abs-tol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for numeric comparisons.",
    )
    p.add_argument(
        "--rel-tol",
        type=float,
        default=1e-3,
        help="Relative tolerance for numeric comparisons.",
    )
    p.add_argument(
        "--ignore-diff-below",
        type=float,
        default=0.05,
        help="Treat numeric mismatches below this absolute diff as ignored.",
    )
    return p.parse_args()


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    s = str(v).strip()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    return str(v).strip() == ""


def _approx_equal(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b), 1.0))


def _stringify(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _is_nullish(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and not math.isfinite(v):
        return True
    s = str(v).strip().lower()
    return s in {"", "none", "null", "nan", "na", "n/a"}


def _build_expected_entries(optimal_yaml: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for family_name, family_data in optimal_yaml.items():
        if family_name not in MODEL_FAMILY_TO_NAME:
            continue
        model_name = MODEL_FAMILY_TO_NAME[family_name]
        if not isinstance(family_data, dict):
            continue
        for method_name, method_data in family_data.items():
            if not isinstance(method_data, dict):
                continue
            method = method_name.lower()
            for concept_name, concept_data in method_data.items():
                if not isinstance(concept_data, dict):
                    continue
                out.append(
                    {
                        "family": family_name,
                        "model_name": model_name,
                        "method": method,
                        "concept": concept_name,
                        "expected_hyperparameters": concept_data.get("hyperparameters", {}) or {},
                        "expected_evaluation": concept_data.get("evaluation", {}) or {},
                    }
                )
    return out


def _load_actual_summaries(checkpoints_root: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    idx: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for p in checkpoints_root.glob("**/evaluation/evaluation_summary.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = (
            str(data.get("method", "")).lower(),
            str(data.get("model_name", "")),
            str(data.get("concept", "")),
        )
        idx[key] = data
    return idx


def _pick_actual_test_row(summary: Dict[str, Any], method: str) -> Dict[str, Any]:
    test = summary.get("test", {}) or {}
    preferred = f"final_test_{TRAIN_EVAL_BY_METHOD.get(method, 'mc')}"
    for key in (preferred, "final_test_mc", "final_test_open"):
        rows = test.get(key, []) or []
        if rows:
            return rows[0]
    return {}


def _compare_hyperparameters(
    expected_hp: Dict[str, Any],
    actual_summary: Dict[str, Any],
    *,
    abs_tol: float,
    rel_tol: float,
) -> Tuple[int, int, List[str]]:
    actual_selected = actual_summary.get("selected_hyperparameters", {}) or {}
    mismatches: List[str] = []
    compared = 0
    matched = 0

    # These are stored at top-level in summary (not in selected_hyperparameters).
    root_keys = {"rank", "seed"}

    for k, expected in expected_hp.items():
        if k in {"n_sentences", "embed_step_enabled"}:
            continue
        if k in root_keys:
            actual = actual_summary.get(k)
        elif k == "embed_step_enabled":
            actual = actual_summary.get("embed_step_enabled")
            if actual is None:
                # Summary may not store this key; fallback to selected hp / source row.
                actual = actual_selected.get(k)
        else:
            actual = actual_selected.get(k)

        if actual is None:
            mismatches.append(f"{k}: missing (expected={expected})")
            compared += 1
            continue

        compared += 1
        if _is_nullish(expected) and _is_nullish(actual):
            matched += 1
            continue
        ef = _to_float(expected)
        af = _to_float(actual)
        if ef is not None and af is not None:
            if _approx_equal(ef, af, abs_tol, rel_tol):
                matched += 1
            else:
                mismatches.append(f"{k}: expected={expected} actual={actual}")
        else:
            if str(expected).strip() == str(actual).strip():
                matched += 1
            else:
                mismatches.append(f"{k}: expected={expected} actual={actual}")
    return matched, compared, mismatches


def _compare_metrics(
    expected_eval: Dict[str, Any],
    actual_test_row: Dict[str, Any],
    *,
    abs_tol: float,
    rel_tol: float,
    ignore_diff_below: float,
) -> Tuple[List[Dict[str, Any]], int, int]:
    detail_rows: List[Dict[str, Any]] = []
    matched = 0
    compared = 0

    # If open-judge metrics are blank in the chosen test row, treat those
    # missing values as optional (e.g., skip_llm_judge/open run).
    open_judge_missing = (
        bool(actual_test_row)
        and _is_blank(actual_test_row.get("qa_acc"))
        and _is_blank(actual_test_row.get("simdom_acc"))
    )

    for metric, expected in expected_eval.items():
        actual = actual_test_row.get(metric)
        expected_f = _to_float(expected)
        actual_f = _to_float(actual)

        if actual is None or (isinstance(actual, str) and actual.strip() == ""):
            if metric in ALPACA_METRICS:
                status = "ignored_missing_alpaca"
            elif metric.startswith("relearning_qa_"):
                status = "ignored_missing_relearning"
            elif metric in OPEN_JUDGE_DEPENDENT_METRICS and open_judge_missing:
                status = "ignored_missing_open_judge"
            else:
                status = "missing_actual"
            diff = ""
        elif expected_f is not None and actual_f is not None:
            ok = _approx_equal(expected_f, actual_f, abs_tol, rel_tol)
            raw_diff = actual_f - expected_f
            diff = raw_diff
            if ok:
                status = "match"
            elif abs(raw_diff) < ignore_diff_below:
                status = "ignored_small_diff"
            else:
                status = "mismatch"
            if status == "match":
                matched += 1
            if status not in {"ignored_small_diff"}:
                compared += 1
        else:
            ok = str(expected).strip() == str(actual).strip()
            status = "match" if ok else "mismatch"
            diff = ""
            if ok:
                matched += 1
            compared += 1

        detail_rows.append(
            {
                "metric": metric,
                "expected": _stringify(expected),
                "actual": _stringify(actual),
                "diff": diff,
                "status": status,
            }
        )
    return detail_rows, matched, compared


def _write_csv(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _to_abs_float(v: Any) -> Optional[float]:
    f = _to_float(v)
    if f is None:
        return None
    return abs(f)


def _build_triage_rows(detail_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    triage: List[Dict[str, Any]] = []
    for r in detail_rows:
        status = str(r.get("status", ""))
        if status in {
            "match",
            "ignored_small_diff",
            "ignored_missing_alpaca",
            "ignored_missing_relearning",
            "ignored_missing_open_judge",
        }:
            continue
        abs_diff = _to_abs_float(r.get("diff"))
        triage.append(
            {
                "family": r.get("family", ""),
                "method": r.get("method", ""),
                "model_name": r.get("model_name", ""),
                "concept": r.get("concept", ""),
                "metric": r.get("metric", ""),
                "status": status,
                "expected": r.get("expected", ""),
                "actual": r.get("actual", ""),
                "diff": r.get("diff", ""),
                "abs_diff": ("" if abs_diff is None else abs_diff),
            }
        )

    def _sort_key(row: Dict[str, Any]) -> Tuple[int, float, str, str, str]:
        # Prioritize numeric mismatches, then missing values/checkpoints.
        status = str(row.get("status", ""))
        priority = 0 if status == "mismatch" else 1
        abs_diff = _to_float(row.get("abs_diff"))
        if abs_diff is None:
            abs_diff = -1.0
        return (
            priority,
            -abs_diff,
            str(row.get("method", "")),
            str(row.get("model_name", "")),
            str(row.get("concept", "")),
        )

    triage.sort(key=_sort_key)
    return triage


def main() -> None:
    args = parse_args()

    optimal_data = yaml.safe_load(args.optimal_yaml.read_text(encoding="utf-8"))
    if not isinstance(optimal_data, dict):
        raise ValueError(f"Unexpected YAML shape in {args.optimal_yaml}")

    expected_entries = _build_expected_entries(optimal_data)
    actual_idx = _load_actual_summaries(args.checkpoints_root)

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    missing_checkpoints = 0
    total_metric_compared = 0
    total_metric_matched = 0

    for ent in expected_entries:
        key = (ent["method"], ent["model_name"], ent["concept"])
        actual = actual_idx.get(key)

        base_info = {
            "family": ent["family"],
            "method": ent["method"],
            "model_name": ent["model_name"],
            "concept": ent["concept"],
        }

        if actual is None:
            missing_checkpoints += 1
            summary_rows.append(
                {
                    **base_info,
                    "checkpoint_found": "no",
                    "hp_matched": 0,
                    "hp_compared": len(ent["expected_hyperparameters"]),
                    "metric_matched": 0,
                    "metric_compared": len(ent["expected_evaluation"]),
                    "metric_match_rate": 0.0,
                    "hp_mismatches": "missing checkpoint",
                }
            )
            for metric, expected in ent["expected_evaluation"].items():
                detail_rows.append(
                    {
                        **base_info,
                        "metric": metric,
                        "expected": _stringify(expected),
                        "actual": "",
                        "diff": "",
                        "status": "missing_checkpoint",
                    }
                )
            continue

        hp_matched, hp_compared, hp_mismatches = _compare_hyperparameters(
            ent["expected_hyperparameters"], actual, abs_tol=args.abs_tol, rel_tol=args.rel_tol
        )
        actual_test_row = _pick_actual_test_row(actual, ent["method"])
        metric_details, metric_matched, metric_compared = _compare_metrics(
            ent["expected_evaluation"],
            actual_test_row,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
            ignore_diff_below=args.ignore_diff_below,
        )

        total_metric_compared += metric_compared
        total_metric_matched += metric_matched

        summary_rows.append(
            {
                **base_info,
                "checkpoint_found": "yes",
                "hp_matched": hp_matched,
                "hp_compared": hp_compared,
                "metric_matched": metric_matched,
                "metric_compared": metric_compared,
                "metric_match_rate": (
                    round(metric_matched / metric_compared, 4) if metric_compared else 0.0
                ),
                "hp_mismatches": "; ".join(hp_mismatches),
            }
        )

        for row in metric_details:
            detail_rows.append({**base_info, **row})

    summary_csv = args.out_dir / "comparison_summary.csv"
    detailed_csv = args.out_dir / "comparison_detailed.csv"
    triage_csv = args.out_dir / "comparison_triage.csv"
    _write_csv(
        summary_csv,
        [
            "family",
            "method",
            "model_name",
            "concept",
            "checkpoint_found",
            "hp_matched",
            "hp_compared",
            "metric_matched",
            "metric_compared",
            "metric_match_rate",
            "hp_mismatches",
        ],
        summary_rows,
    )
    _write_csv(
        detailed_csv,
        [
            "family",
            "method",
            "model_name",
            "concept",
            "metric",
            "expected",
            "actual",
            "diff",
            "status",
        ],
        detail_rows,
    )
    triage_rows = _build_triage_rows(detail_rows)
    _write_csv(
        triage_csv,
        [
            "family",
            "method",
            "model_name",
            "concept",
            "metric",
            "status",
            "expected",
            "actual",
            "diff",
            "abs_diff",
        ],
        triage_rows,
    )

    total_expected = len(expected_entries)
    found = total_expected - missing_checkpoints
    match_rate = (total_metric_matched / total_metric_compared) if total_metric_compared else 0.0

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {detailed_csv}")
    print(f"Wrote: {triage_csv}")
    print(
        "Overview: "
        f"expected={total_expected}, found={found}, missing={missing_checkpoints}, "
        f"metric_match={total_metric_matched}/{total_metric_compared} "
        f"({match_rate:.2%})"
    )


if __name__ == "__main__":
    main()
