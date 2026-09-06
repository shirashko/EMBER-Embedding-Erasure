#!/usr/bin/env python3
"""Build a compact summary CSV from saved unlearning checkpoints.

Reads ``unlearned_checkpoints/**/evaluation/evaluation_summary.json`` (chosen
hyperparameter configurations) and writes one row per checkpoint with train/test
metrics and a method-specific config tuple.

Example::

    python scripts/build_results_summary_table.py \\
        --checkpoints-root unlearned_checkpoints \\
        --out results/summary_table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]

SNMF_HP_COLUMNS = [
    "w_mode", "feature_source",
    "ratio_thresh", "coverage_thresh", "neurons_thresh",
    "delta_in", "layer_lo_in", "layer_hi_in", "k_features_mlp_in",
    "delta_out", "layer_lo_out", "layer_hi_out", "k_features_mlp_out",
]

RMU_HP_COLUMNS = [
    "lr", "alpha", "steering",
    "setting_name", "layer_id", "layer_ids", "param_ids",
]

CRISP_HP_COLUMNS = [
    "k_features", "alpha", "lr",
    "layer_lo", "layer_hi", "layer_step",
    "num_epochs", "lora_rank",
]

PISCES_HP_COLUMNS = [
    "k_pisces", "value_pisces", "ratio_thresh",
]

HP_COLUMNS_BY_METHOD = {
    "snmf": SNMF_HP_COLUMNS,
    "rmu": RMU_HP_COLUMNS,
    "crisp": CRISP_HP_COLUMNS,
    "pisces": PISCES_HP_COLUMNS,
    "ember": [],
}

TRAIN_EVAL_BY_METHOD = {
    "pisces": "open",
    "crisp": "mc",
    "rmu": "mc",
    "snmf": "mc",
    "ember": "mc",
}


EXCLUDED_CONCEPTS = {"bio"}

CONFIG_EXCLUDE = {
    "embed_step_enabled",
    "delta_embed",
    "k_features_embed",
    "n_tokens_edited",
    "wall_time_s",
}

OUTPUT_COLUMNS = [
    "model",
    "concept",
    "method",
    "optimized_by",
    "train_efficacy",
    "train_specificity",
    "train_harmonic",
    "test_relearning_qa",
    "train_mmlu (acc, frac, invalid)",
    "train_qa (acc, frac, invalid)",
    "train_simdom (acc, frac, invalid)",
    "test_efficacy",
    "test_specificity",
    "test_harmonic",
    "test_mmlu (acc, frac, invalid)",
    "test_qa (acc, frac, invalid)",
    "test_simdom (acc, frac, invalid)",
    "config",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build summary CSV from unlearned checkpoint evaluations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("unlearned_checkpoints"),
        help="Root directory containing saved checkpoints.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/summary_table.csv"),
        help="Output CSV path.",
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
    if isinstance(v, float) and not math.isfinite(v):
        return True
    return str(v).strip() == ""


def _round_num(v: Any) -> str:
    f = _to_float(v)
    if f is None:
        return ""
    return f"{round(f, 2):.2f}"


def _format_config_value(v: Any) -> str:
    if _is_blank(v):
        return ""
    f = _to_float(v)
    if f is not None:
        if f.is_integer():
            return str(int(f))
        text = f"{f:.12g}"
        if "e" in text or "E" in text:
            return text
        if "." in text:
            return text.rstrip("0").rstrip(".")
        return text
    return str(v).strip()


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


def _pick_train_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    train = summary.get("train", {}) or {}
    for key in ("top_hps_valid", "top_hps", "hps"):
        rows = train.get(key, []) or []
        if rows:
            return rows[0]
    return {}


def _value_matches(row_val: Any, expected_val: Any) -> bool:
    if _is_blank(row_val) and _is_blank(expected_val):
        return True
    if _is_blank(row_val) or _is_blank(expected_val):
        return False
    try:
        return abs(float(row_val) - float(expected_val)) <= 1e-12
    except (TypeError, ValueError):
        return str(row_val).strip() == str(expected_val).strip()


def _read_matching_csv_rows(
        csv_path: Path,
        concept: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []

    hp = hyperparameters or {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows: List[Dict[str, Any]] = []
        for row in csv.DictReader(f):
            if str(row.get("concept", "")) != str(concept):
                continue
            ok = True
            for key, expected in hp.items():
                if key in row and not _value_matches(row.get(key), expected):
                    ok = False
                    break
            if ok:
                rows.append(row)
    return rows


def _selected_hyperparameters(summary: Dict[str, Any]) -> Dict[str, Any]:
    selected = dict(summary.get("selected_hyperparameters", {}) or {})
    selected.setdefault("rank", summary.get("rank"))
    selected.setdefault("seed", summary.get("seed"))
    return selected


def _pick_test_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    train_eval = _resolve_train_eval(summary)
    test = summary.get("test", {}) or {}
    rows = test.get(f"final_test_{train_eval}", []) or []
    if rows:
        return rows[0]

    sources = summary.get("sources", {}) or {}
    csv_key = f"final_test_{train_eval}_csv"
    csv_path = Path(str(sources.get(csv_key, "")))
    if not csv_path.is_absolute():
        csv_path = ROOT_DIR / csv_path
    fallback_rows = _read_matching_csv_rows(
        csv_path,
        str(summary.get("concept", "")),
        hyperparameters=_selected_hyperparameters(summary),
    )
    if fallback_rows:
        return fallback_rows[0]

    # Last resort: any row for this concept in the results CSV.
    fallback_rows = _read_matching_csv_rows(
        csv_path,
        str(summary.get("concept", "")),
    )
    return fallback_rows[0] if fallback_rows else {}


def _relearning_metric_key(train_eval: str) -> str:
    return f"relearning_qa_{train_eval}"


def _format_triple(row: Dict[str, Any], prefix: str, *, train_eval: str) -> str:
    acc = _round_num(row.get(f"{prefix}_acc"))
    frac = _round_num(row.get(f"{prefix}_frac"))
    invalid_key = f"{prefix}_invalid"
    use_invalid = prefix == "mmlu" or train_eval == "mc"
    invalid = _round_num(row.get(invalid_key)) if use_invalid else ""
    if not acc and not frac and not invalid:
        return ""
    if use_invalid:
        return f"({acc}, {frac}, {invalid})"
    return f"({acc}, {frac})"


def _config_keys(method: str) -> List[str]:
    hp_cols = HP_COLUMNS_BY_METHOD.get(_base_method(method), [])
    return ["rank", "seed"] + [c for c in hp_cols if c not in CONFIG_EXCLUDE]


def _format_config(summary: Dict[str, Any]) -> str:
    method = str(summary.get("method", ""))
    selected = _selected_hyperparameters(summary)

    parts: List[str] = []
    for key in _config_keys(method):
        value = _format_config_value(selected.get(key))
        if value:
            parts.append(f"{key}={value}")
    return f"({', '.join(parts)})"


def _row_from_summary(summary: Dict[str, Any]) -> Dict[str, str]:
    train_eval = _resolve_train_eval(summary)
    train_row = _pick_train_row(summary)
    test_row = _pick_test_row(summary)
    relearn_key = _relearning_metric_key(train_eval)

    return {
        "model": str(summary.get("model_name", "")),
        "concept": str(summary.get("concept", "")),
        "method": _base_method(str(summary.get("method", ""))),
        "optimized_by": train_eval,
        "train_efficacy": _round_num(train_row.get("efficacy")),
        "train_specificity": _round_num(train_row.get("specificity")),
        "train_harmonic": _round_num(train_row.get("harmonic")),
        "test_relearning_qa": _round_num(test_row.get(relearn_key)),
        "train_mmlu (acc, frac, invalid)": _format_triple(train_row, "mmlu", train_eval=train_eval),
        "train_qa (acc, frac, invalid)": _format_triple(train_row, "qa", train_eval=train_eval),
        "train_simdom (acc, frac, invalid)": _format_triple(train_row, "simdom", train_eval=train_eval),
        "test_efficacy": _round_num(test_row.get("efficacy")),
        "test_specificity": _round_num(test_row.get("specificity")),
        "test_harmonic": _round_num(test_row.get("harmonic")),
        "test_mmlu (acc, frac, invalid)": _format_triple(test_row, "mmlu", train_eval=train_eval),
        "test_qa (acc, frac, invalid)": _format_triple(test_row, "qa", train_eval=train_eval),
        "test_simdom (acc, frac, invalid)": _format_triple(test_row, "simdom", train_eval=train_eval),
        "config": _format_config(summary),
    }


def _load_summaries(checkpoints_root: Path) -> List[Dict[str, Any]]:
    root = checkpoints_root
    if not root.is_absolute():
        root = ROOT_DIR / root

    summaries: List[Dict[str, Any]] = []
    for path in sorted(root.glob("**/evaluation/evaluation_summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] failed to read {path}: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        concept = str(data.get("concept", ""))
        if concept in EXCLUDED_CONCEPTS:
            continue
        summaries.append(data)
    return summaries


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    summaries = _load_summaries(args.checkpoints_root)
    rows = [_row_from_summary(s) for s in summaries]
    rows.sort(key=lambda r: (r["method"], r["model"], r["concept"]))

    out_path = args.out
    if not out_path.is_absolute():
        out_path = ROOT_DIR / out_path

    _write_csv(out_path, OUTPUT_COLUMNS, rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
