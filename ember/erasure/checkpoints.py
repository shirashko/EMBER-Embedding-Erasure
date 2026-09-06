"""Save unlearned model checkpoints after the final-test apply step."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from ember.erasure import io, log
from ember.erasure.config import RunConfig
from ember.utils import _safe_concept, _safe_model_name


def checkpoint_dir(cfg: RunConfig, concept: str) -> Path:
    """``<root>/<method>/<model>/<concept>/`` under the repo unless ``root`` is absolute."""
    use_embed_suffix = (cfg.method != "ember") and bool(cfg.ember_step.enabled)
    method = io.method_dir_name(cfg.method, use_embed_suffix)
    root = Path(cfg.checkpoint.root)
    if not root.is_absolute():
        root = io.ROOT_DIR / root
    return root / method / _safe_model_name(cfg.model_name) / _safe_concept(concept)


def _checkpoint_populated(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.iterdir())


def save_unlearned_checkpoint(
        model: Any,
        tokenizer: Any,
        cfg: RunConfig,
        concept: str,
        *,
        hyperparameters: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Persist the post-unlearning model (and tokenizer) for ``concept``."""
    if not cfg.checkpoint.enabled:
        return None

    out_dir = checkpoint_dir(cfg, concept)
    if _checkpoint_populated(out_dir) and not cfg.overwrite:
        log.info("checkpoint already exists at %s; skip save", out_dir)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("saving unlearned checkpoint -> %s", out_dir)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "method": cfg.method,
        "model_name": cfg.model_name,
        "concept": concept,
        "rank": cfg.rank,
        "seed": cfg.seed,
        "train_eval": cfg.train_eval,
        "hyperparameters": hyperparameters or {},
    }
    (out_dir / "unlearned_checkpoints.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_dir


def _value_matches(row_val: Any, expected_val: Any) -> bool:
    def _is_nullish(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, float) and not math.isfinite(v):
            return True
        s = str(v).strip().lower()
        return s in {"", "none", "null", "nan", "na", "n/a"}

    if _is_nullish(row_val) and _is_nullish(expected_val):
        return True
    if _is_nullish(row_val) or _is_nullish(expected_val):
        return False

    if expected_val is None:
        return True
    if row_val is None:
        return False
    try:
        return abs(float(row_val) - float(expected_val)) <= 1e-12
    except (TypeError, ValueError):
        return str(row_val).strip() == str(expected_val).strip()


def _read_matching_rows(
        src_csv: Path,
        concept: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not src_csv.exists():
        return []

    hp = hyperparameters or {}
    with src_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if str(row.get("concept", "")) != str(concept):
                continue
            ok = True
            for k, v in hp.items():
                if k in row and not _value_matches(row.get(k), v):
                    ok = False
                    break
            if ok:
                rows.append(row)
    return rows


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_concept_key(per_concept: Dict[str, Any], concept: str) -> Optional[str]:
    if concept in per_concept:
        return concept
    concept_spaced = concept.replace("_", " ")
    if concept_spaced in per_concept:
        return concept_spaced
    return None


def _collect_baseline_values(cfg: RunConfig, concept: str) -> Dict[str, Any]:
    bdir = io.baseline_dir(cfg.model_name)
    out: Dict[str, Any] = {"qa": {}, "simdom": {}, "mmlu": {}}

    for kind in ("qa", "simdom"):
        for split in ("train", "test"):
            for suffix in ("mc", "open"):
                name = f"{kind}_{split}_{suffix}"
                path = bdir / f"baseline_{name}.json"
                data = _load_json_if_exists(path)
                if not data:
                    continue
                per_concept = data.get("meta", {}).get("per_concept", {})
                if not isinstance(per_concept, dict):
                    continue
                key = _resolve_concept_key(per_concept, concept)
                if key is None:
                    continue
                entry = per_concept.get(key, {})
                if not isinstance(entry, dict):
                    continue
                out[kind][name] = {
                    "source_concept_key": key,
                    "accuracy_generation": entry.get("accuracy_generation"),
                    "accuracy": entry.get("accuracy"),
                    "invalid_rate_generation": entry.get("invalid_rate_generation"),
                    "num_correct_generation": entry.get("num_correct_generation"),
                    "total": entry.get("total"),
                }

    for split in ("train", "test"):
        name = f"mmlu_{split}"
        path = bdir / f"baseline_{name}.json"
        data = _load_json_if_exists(path)
        if not data:
            continue
        metrics = data.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        out["mmlu"][name] = {
            "accuracy_generation": metrics.get("accuracy_generation"),
            "invalid_rate_generation": metrics.get("invalid_rate_generation"),
        }

    return out


def _clear_legacy_eval_outputs(eval_dir: Path) -> None:
    legacy_files = [
        "train_hps_selected.csv",
        "train_top_hps_selected.csv",
        "train_top_hps_valid_selected.csv",
        "final_test_mc.csv",
        "final_test_open.csv",
        "relearning.csv",
        "baseline_values.json",
        "evaluation_manifest.json",
    ]
    for name in legacy_files:
        p = eval_dir / name
        if p.exists():
            p.unlink()
    baselines_dir = eval_dir / "baselines"
    if baselines_dir.exists() and baselines_dir.is_dir():
        for child in baselines_dir.iterdir():
            if child.is_file():
                child.unlink()
        try:
            baselines_dir.rmdir()
        except OSError:
            pass


def _first_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return rows[0] if rows else {}


def _pick_train_row(train_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    for key in ("top_hps_valid", "top_hps", "hps"):
        row = _first_row(train_rows.get(key, []))
        if row:
            return row
    return {}


def _pick_test_row(test_rows: Dict[str, List[Dict[str, Any]]], train_eval: str) -> Dict[str, Any]:
    """Return the test row for the configured train_eval mode only (no MC/open fallback)."""
    return _first_row(test_rows.get(f"final_test_{train_eval}", []))


def _baseline_accuracy(entry: Dict[str, Any]) -> Optional[Any]:
    """Return generation or open baseline accuracy, preferring a non-null value."""
    acc = entry.get("accuracy_generation")
    if acc is None:
        acc = entry.get("accuracy")
    return acc


def _baseline_for_metric(
        baselines: Dict[str, Any],
        metric: str,
        *,
        split: str,
        eval_mode: str,
) -> Optional[Any]:
    if metric == "mmlu_acc":
        return (((baselines.get("mmlu", {}) or {}).get(f"mmlu_{split}", {}) or {})
                .get("accuracy_generation"))
    if metric == "mmlu_invalid":
        return (((baselines.get("mmlu", {}) or {}).get(f"mmlu_{split}", {}) or {})
                .get("invalid_rate_generation"))
    if metric == "mmlu_frac":
        return 1.0

    if metric.startswith("qa_"):
        entry = (((baselines.get("qa", {}) or {}).get(f"qa_{split}_{eval_mode}", {}) or {}))
        if metric == "qa_acc":
            return _baseline_accuracy(entry)
        if metric == "qa_invalid":
            return entry.get("invalid_rate_generation")
        if metric == "qa_frac":
            return 1.0

    if metric.startswith("simdom_"):
        entry = (((baselines.get("simdom", {}) or {})
                  .get(f"simdom_{split}_{eval_mode}", {}) or {}))
        if metric == "simdom_acc":
            return _baseline_accuracy(entry)
        if metric == "simdom_invalid":
            return entry.get("invalid_rate_generation")
        if metric == "simdom_frac":
            return 1.0

    return None


def _write_score_comparison_csv(
        eval_dir: Path,
        *,
        train_eval: str,
        baselines: Dict[str, Any],
        train_rows: Dict[str, List[Dict[str, Any]]],
        test_rows: Dict[str, List[Dict[str, Any]]],
) -> None:
    train_row = _pick_train_row(train_rows)
    test_row = _pick_test_row(test_rows, train_eval)
    train_mode = train_eval if train_eval in ("mc", "open") else "mc"

    metrics: List[str] = []
    for mode in (f"train_{train_mode}", f"test_{train_mode}"):
        for m in io.metric_keys_for_mode(mode):
            if m not in metrics:
                metrics.append(m)

    out_path = eval_dir / "score_comparison.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "baseline_train",
                "train_after_unlearning",
                "baseline_test",
                "test_after_unlearning",
            ],
        )
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "metric": metric,
                    "baseline_train": _baseline_for_metric(
                        baselines, metric, split="train", eval_mode=train_mode
                    ),
                    "train_after_unlearning": train_row.get(metric, ""),
                    "baseline_test": _baseline_for_metric(
                        baselines, metric, split="test", eval_mode=train_mode
                    ),
                    "test_after_unlearning": test_row.get(metric, ""),
                }
            )


def save_checkpoint_evaluation_data(
        cfg: RunConfig,
        concept: str,
        *,
        train_concept_dir: Path,
        final_mc_csv: Path,
        final_open_csv: Path,
        selected_hyperparameters: Optional[Dict[str, Any]] = None,
        relearning_csv: Optional[Path] = None,
) -> Optional[Path]:
    """Save compact train/test/baseline summary under the checkpoint directory."""
    if not cfg.checkpoint.enabled:
        return None

    out_dir = checkpoint_dir(cfg, concept)
    if not out_dir.exists():
        return None

    eval_dir = out_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _clear_legacy_eval_outputs(eval_dir)

    train_rows = {
        "hps": _read_matching_rows(
            train_concept_dir / "hps.csv",
            concept,
            hyperparameters=selected_hyperparameters,
        ),
        "top_hps": _read_matching_rows(
            train_concept_dir / "top_hps.csv",
            concept,
            hyperparameters=selected_hyperparameters,
        ),
        "top_hps_valid": _read_matching_rows(
            train_concept_dir / "top_hps_valid.csv",
            concept,
            hyperparameters=selected_hyperparameters,
        ),
    }

    mc_rows = _read_matching_rows(
        final_mc_csv,
        concept,
        hyperparameters=selected_hyperparameters,
    )

    open_rows = _read_matching_rows(
        final_open_csv,
        concept,
        hyperparameters=selected_hyperparameters,
    )

    relearning_rows: List[Dict[str, Any]] = []
    if relearning_csv and relearning_csv.exists():
        with relearning_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            relearning_rows = list(reader)

    baseline_values = _collect_baseline_values(cfg, concept)
    summary = {
        "model_name": cfg.model_name,
        "method": cfg.method,
        "concept": concept,
        "rank": cfg.rank,
        "seed": cfg.seed,
        "train_eval": cfg.train_eval,
        "selected_hyperparameters": selected_hyperparameters or {},
        "train": train_rows,
        "test": {
            "final_test_mc": mc_rows,
            "final_test_open": open_rows,
        },
        "relearning": relearning_rows,
        "baselines": baseline_values,
        "sources": {
            "train_hps_csv": str(train_concept_dir / "hps.csv"),
            "train_top_hps_csv": str(train_concept_dir / "top_hps.csv"),
            "train_top_hps_valid_csv": str(train_concept_dir / "top_hps_valid.csv"),
            "final_test_mc_csv": str(final_mc_csv),
            "final_test_open_csv": str(final_open_csv),
            "relearning_csv": str(relearning_csv) if relearning_csv else None,
            "baseline_dir": str(io.baseline_dir(cfg.model_name)),
        },
    }
    (eval_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_score_comparison_csv(
        eval_dir,
        train_eval=cfg.train_eval,
        baselines=baseline_values,
        train_rows=train_rows,
        test_rows={"final_test_mc": mc_rows, "final_test_open": open_rows},
    )
    log.info("saved compact checkpoint evaluation summary -> %s", eval_dir / "evaluation_summary.json")
    return eval_dir


__all__ = [
    "checkpoint_dir",
    "save_unlearned_checkpoint",
    "save_checkpoint_evaluation_data",
]
