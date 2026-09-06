#!/usr/bin/env python3
"""Print test specificity / efficacy / harmonic tables for seed-42 checkpoints.

Reads ``unlearned_checkpoints/**/evaluation/evaluation_summary.json`` and shows
one table per (model, method) for the requested concepts, plus mean and std
across concepts in that group.

Example::

    python scripts/summarize_seed42_test_scores.py
    python scripts/summarize_seed42_test_scores.py --out results/seed42_test_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_CONCEPTS = [
    "Ancient Rome",
    "Golf",
    "Uranium",
    "Culture of Greece",
    "Cannabis",
    "Baseball",
]

DEFAULT_MODEL_SUBSTRINGS = ("google", "meta", "Qwen3.5-2B")

TRAIN_EVAL_BY_METHOD = {
    "pisces": "open",
    "crisp": "mc",
    "rmu": "mc",
    "snmf": "mc",
    "ember": "mc",
}

METRICS = ("specificity", "efficacy", "harmonic")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Group seed-42 unlearned-checkpoint test scores by model and method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("unlearned_checkpoints"),
        help="Root directory containing saved checkpoints.",
    )
    p.add_argument("--seed", type=int, default=42, help="Keep only this seed.")
    p.add_argument(
        "--concepts",
        nargs="+",
        default=list(DEFAULT_CONCEPTS),
        help="Concepts to include (order is used in tables).",
    )
    p.add_argument(
        "--model-substrings",
        nargs="+",
        default=list(DEFAULT_MODEL_SUBSTRINGS),
        help="Keep models whose name contains any of these (case-insensitive).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV path for per-concept rows plus mean/std rows.",
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


def _norm_concept(name: str) -> str:
    return " ".join(str(name).replace("_", " ").split()).casefold()


def _base_method(method: str) -> str:
    name = str(method).lower()
    if name.endswith("_ef"):
        return name[:-3]
    return name


def _resolve_train_eval(summary: Dict[str, Any]) -> str:
    train_eval = str(summary.get("train_eval", "")).strip().lower()
    if train_eval in {"mc", "open"}:
        return train_eval
    method = _base_method(str(summary.get("method", "")))
    return TRAIN_EVAL_BY_METHOD.get(method, "mc")


def _pick_test_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    train_eval = _resolve_train_eval(summary)
    test = summary.get("test", {}) or {}
    rows = test.get(f"final_test_{train_eval}", []) or []
    return rows[0] if rows else {}


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:.4f}"


def _mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None, None
    mean = statistics.fmean(nums)
    if len(nums) == 1:
        return mean, 0.0
    return mean, statistics.stdev(nums)


def _model_family(model_name: str) -> str:
    name = model_name.casefold()
    if "google" in name:
        return "google"
    if "meta" in name:
        return "meta"
    if "qwen" in name:
        return "qwen"
    return "other"


def _matches_model(model_name: str, substrings: Sequence[str]) -> bool:
    name = model_name.casefold()
    return any(s.casefold() in name for s in substrings)


def _load_summaries(checkpoints_root: Path) -> List[Dict[str, Any]]:
    root = checkpoints_root if checkpoints_root.is_absolute() else ROOT_DIR / checkpoints_root
    summaries: List[Dict[str, Any]] = []
    for path in sorted(root.glob("**/evaluation/evaluation_summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] failed to read {path}: {exc}")
            continue
        if isinstance(data, dict):
            summaries.append(data)
    return summaries


def _extract_rows(
    summaries: Iterable[Dict[str, Any]],
    *,
    seed: int,
    concepts: Sequence[str],
    model_substrings: Sequence[str],
) -> List[Dict[str, Any]]:
    wanted = {_norm_concept(c): c for c in concepts}
    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        if _to_float(summary.get("seed")) != float(seed):
            continue
        model = str(summary.get("model_name", "")).strip()
        if not _matches_model(model, model_substrings):
            continue
        concept_raw = str(summary.get("concept", "")).strip()
        concept_key = _norm_concept(concept_raw)
        if concept_key not in wanted:
            continue
        test_row = _pick_test_row(summary)
        rows.append(
            {
                "model": model,
                "family": _model_family(model),
                "method": _base_method(str(summary.get("method", ""))),
                "concept": wanted[concept_key],
                "train_eval": _resolve_train_eval(summary),
                "specificity": _to_float(test_row.get("specificity")),
                "efficacy": _to_float(test_row.get("efficacy")),
                "harmonic": _to_float(test_row.get("harmonic")),
            }
        )
    return rows


def _group_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (row["family"], row["model"], row["method"])


def _markdown_table(headers: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    def fmt_row(cells: Sequence[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    lines = [
        fmt_row(headers),
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    lines.extend(fmt_row(row) for row in body)
    return "\n".join(lines)


def _print_group_tables(rows: List[Dict[str, Any]], concepts: Sequence[str]) -> List[Dict[str, Any]]:
    export: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)

    for family, model, method in sorted(groups):
        by_concept = {r["concept"]: r for r in groups[(family, model, method)]}
        body: List[List[str]] = []
        metric_lists = {m: [] for m in METRICS}
        for concept in concepts:
            rec = by_concept.get(concept)
            if rec is None:
                body.append([concept, "", "", ""])
                for m in METRICS:
                    metric_lists[m].append(None)
                export.append(
                    {
                        "model": model,
                        "method": method,
                        "concept": concept,
                        "specificity": "",
                        "efficacy": "",
                        "harmonic": "",
                    }
                )
                continue
            body.append([concept] + [_fmt(rec[m]) for m in METRICS])
            for m in METRICS:
                metric_lists[m].append(rec[m])
            export.append(
                {
                    "model": model,
                    "method": method,
                    "concept": concept,
                    "specificity": _fmt(rec["specificity"]),
                    "efficacy": _fmt(rec["efficacy"]),
                    "harmonic": _fmt(rec["harmonic"]),
                }
            )

        mean_cells = ["mean"]
        std_cells = ["std"]
        mean_export: Dict[str, str] = {
            "model": model, "method": method, "concept": "mean",
        }
        std_export: Dict[str, str] = {
            "model": model, "method": method, "concept": "std",
        }
        for m in METRICS:
            mean, std = _mean_std(metric_lists[m])
            mean_cells.append(_fmt(mean))
            std_cells.append(_fmt(std))
            mean_export[m] = _fmt(mean)
            std_export[m] = _fmt(std)
        body.extend([mean_cells, std_cells])
        export.extend([mean_export, std_export])

        n_present = sum(1 for c in concepts if c in by_concept)
        print(f"## {model}  /  {method}  (n={n_present}/{len(concepts)})")
        print()
        print(_markdown_table(["concept", *METRICS], body))
        print()

    return export


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "method", "concept", *METRICS]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summaries = _load_summaries(args.checkpoints_root)
    rows = _extract_rows(
        summaries,
        seed=args.seed,
        concepts=args.concepts,
        model_substrings=args.model_substrings,
    )
    if not rows:
        print("No matching checkpoint evaluations found.")
        return

    print(
        f"Seed {args.seed} test scores from {args.checkpoints_root} "
        f"({len(rows)} concept rows, grouped by model and method)\n"
    )
    export = _print_group_tables(rows, args.concepts)
    if args.out is not None:
        out_path = args.out if args.out.is_absolute() else ROOT_DIR / args.out
        _write_csv(out_path, export)
        print(f"Wrote {len(export)} rows to {out_path}")


if __name__ == "__main__":
    main()
