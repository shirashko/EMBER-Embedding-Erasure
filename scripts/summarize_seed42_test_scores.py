#!/usr/bin/env python3
"""Print test specificity / efficacy / harmonic tables for seed-42 checkpoints.

Reads ``unlearned_checkpoints/**/evaluation/evaluation_summary.json`` and shows
one per-config table (model / method / concept) and a separate mean/std table
across concepts in each (model, method) group.

Gambling is appended for Gemma/Llama on pisces, crisp, rmu, and snmf, and for
Qwen on snmf, rmu, and crisp. Republic of Ireland, wmdp-bio, and wmdp-cyber
are appended for matching families/methods when a seed-42 checkpoint exists.

Example::

    python scripts/summarize_seed42_test_scores.py
    python scripts/summarize_seed42_test_scores.py --out results/seed42_test_scores.csv
    # also writes results/seed42_test_scores_summary.csv
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

# Extra concepts appended after DEFAULT_CONCEPTS for matching (family, method).
# ``only_if_present`` skips the row when that checkpoint is missing.
EXTRA_CONCEPTS: List[Dict[str, Any]] = [
    {
        "name": "Gambling",
        "methods_by_family": {
            "google": frozenset({"pisces", "crisp", "rmu", "snmf"}),
            "meta": frozenset({"pisces", "crisp", "rmu", "snmf"}),
            "qwen": frozenset({"snmf", "rmu", "crisp"}),
        },
        "only_if_present": False,
    },
    {
        "name": "Republic of Ireland",
        "methods_by_family": {
            "google": frozenset({"pisces", "crisp", "rmu", "snmf"}),
            "meta": frozenset({"pisces", "crisp", "rmu", "snmf"}),
            "qwen": frozenset({"snmf", "rmu", "crisp"}),
        },
        "only_if_present": True,
    },
    {
        "name": "wmdp-bio",
        "methods_by_family": {
            "google": frozenset({"pisces", "crisp", "rmu", "snmf"}),
            "meta": frozenset({"pisces", "crisp", "rmu", "snmf"}),
            "qwen": frozenset({"pisces", "crisp", "rmu", "snmf"}),
        },
        "only_if_present": True,
    },
    {
        "name": "wmdp-cyber",
        "methods_by_family": {
            "google": frozenset({"pisces", "crisp", "rmu", "snmf"}),
            "meta": frozenset({"pisces", "crisp", "rmu", "snmf"}),
            "qwen": frozenset({"pisces", "crisp", "rmu", "snmf"}),
        },
        "only_if_present": True,
    },
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
        help="Concepts to include (order is used in tables). Extra concepts such as Gambling, Republic of Ireland, wmdp-bio, and wmdp-cyber are appended for allowed model/method pairs.",
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
        help="Optional CSV path for per-config rows. Mean/std are written to "
             "<stem>_summary.csv next to this file.",
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


def _extra_by_key() -> Dict[str, Dict[str, Any]]:
    return {_norm_concept(str(spec["name"])): spec for spec in EXTRA_CONCEPTS}


def _allows_extra(spec: Dict[str, Any], family: str, method: str) -> bool:
    methods_by_family = spec.get("methods_by_family") or {}
    return method in methods_by_family.get(family, ())


def _concepts_for_group(
    base_concepts: Sequence[str],
    family: str,
    method: str,
    present: Optional[Iterable[str]] = None,
) -> List[str]:
    extra_keys = set(_extra_by_key())
    present_keys = {_norm_concept(c) for c in present} if present is not None else None
    concepts = [c for c in base_concepts if _norm_concept(c) not in extra_keys]
    for spec in EXTRA_CONCEPTS:
        name = str(spec["name"])
        if not _allows_extra(spec, family, method):
            continue
        if spec.get("only_if_present") and present_keys is not None:
            if _norm_concept(name) not in present_keys:
                continue
        concepts.append(name)
    return concepts


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
    extras = _extra_by_key()
    wanted = {_norm_concept(c): c for c in concepts}
    for spec in EXTRA_CONCEPTS:
        wanted.setdefault(_norm_concept(str(spec["name"])), str(spec["name"]))
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
        family = _model_family(model)
        method = _base_method(str(summary.get("method", "")))
        extra_spec = extras.get(concept_key)
        if extra_spec is not None and not _allows_extra(extra_spec, family, method):
            continue
        test_row = _pick_test_row(summary)
        rows.append(
            {
                "model": model,
                "family": family,
                "method": method,
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


SUMMARY_FIELDS = [
    "model",
    "method",
    "n",
    "specificity_mean",
    "specificity_std",
    "efficacy_mean",
    "efficacy_std",
    "harmonic_mean",
    "harmonic_std",
]


def _print_score_tables(
    rows: List[Dict[str, Any]], concepts: Sequence[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_config: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)

    per_body: List[List[str]] = []
    summary_body: List[List[str]] = []
    for family, model, method in sorted(groups):
        by_concept = {r["concept"]: r for r in groups[(family, model, method)]}
        group_concepts = _concepts_for_group(
            concepts, family, method, present=by_concept
        )
        metric_lists = {m: [] for m in METRICS}
        for concept in group_concepts:
            rec = by_concept.get(concept)
            if rec is None:
                per_body.append([model, method, concept, "", "", ""])
                for m in METRICS:
                    metric_lists[m].append(None)
                per_config.append(
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
            per_body.append([model, method, concept] + [_fmt(rec[m]) for m in METRICS])
            for m in METRICS:
                metric_lists[m].append(rec[m])
            per_config.append(
                {
                    "model": model,
                    "method": method,
                    "concept": concept,
                    "specificity": _fmt(rec["specificity"]),
                    "efficacy": _fmt(rec["efficacy"]),
                    "harmonic": _fmt(rec["harmonic"]),
                }
            )

        n_present = sum(1 for c in group_concepts if c in by_concept)
        summary: Dict[str, str] = {
            "model": model,
            "method": method,
            "n": str(n_present),
        }
        summary_cells = [model, method, str(n_present)]
        for m in METRICS:
            mean, std = _mean_std(metric_lists[m])
            summary[f"{m}_mean"] = _fmt(mean)
            summary[f"{m}_std"] = _fmt(std)
            summary_cells.extend([_fmt(mean), _fmt(std)])
        summaries.append(summary)
        summary_body.append(summary_cells)

    print("## Per-config test scores")
    print()
    print(_markdown_table(["model", "method", "concept", *METRICS], per_body))
    print()
    print("## Mean / std by model and method")
    print()
    print(
        _markdown_table(
            [
                "model",
                "method",
                "n",
                "specificity_mean",
                "specificity_std",
                "efficacy_mean",
                "efficacy_std",
                "harmonic_mean",
                "harmonic_std",
            ],
            summary_body,
        )
    )
    print()
    return per_config, summaries


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _summary_csv_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_summary{out_path.suffix}")


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
    per_config, summary_rows = _print_score_tables(rows, args.concepts)
    if args.out is not None:
        out_path = args.out if args.out.is_absolute() else ROOT_DIR / args.out
        summary_path = _summary_csv_path(out_path)
        _write_csv(out_path, per_config, ["model", "method", "concept", *METRICS])
        _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
        print(f"Wrote {len(per_config)} per-config rows to {out_path}")
        print(f"Wrote {len(summary_rows)} mean/std rows to {summary_path}")


if __name__ == "__main__":
    main()
