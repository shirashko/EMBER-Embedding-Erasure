"""Baseline computation + on-disk cache.

The baselines are the pristine model's eval scores on each set. Every
erasure-method grid cell compares its post-erasure metrics to the baseline.
Computing baselines is expensive (~10 minutes per set on Gemma), so we cache
JSON files keyed by:

    data/baselines/<safe_model_name>/baseline_<set_name>.json

Set names: ``qa_train_mc``, ``qa_train_open``, ``qa_test_mc``,
``qa_test_open``, ``simdom_*`` (same axes), ``mmlu_train``,
``mmlu_test``, ``alpaca_train``, ``alpaca_test``.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ember.local_datasets import (
    load_mc_qa_items,
    load_mmlu_indices,
    load_open_qa_examples,
)
from ember.evals.alpaca import evaluate_alpaca
from ember.evals.gemini import GeminiEvaluator
from ember.evals.mc import evaluate_mc_generation, prepare_mc_items, stable_item_id
from ember.evals.mmlu import evaluate_mmlu_generation, load_mmlu_items
from ember.evals.model_wrap import ensure_wrapped_model
from ember.evals.open_qa import evaluate_open_qa
from ember.evals.schema import BaselineResult

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DEFAULT_BASELINE_DIR = DATA_DIR / "baselines"
SAMPLES_DIR = DATA_DIR / "baseline_samples"
DEFAULT_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

VALID_MODES = ("train_open", "train_mc", "test_open", "test_mc")


def _model_safe_name(name: str) -> str:
    return (name.replace("/", "_").replace(":", "_")
                .replace("@", "_").replace(" ", "_"))


def _save_sample(set_name: str, model_safe: str,
                 records: List[Dict[str, Any]], fraction: float = 1.0) -> None:
    if not records:
        return
    n = len(records)
    k = max(1, int(round(fraction * n)))
    rng = random.Random(12345)
    sampled = [records[i] for i in rng.sample(range(n), k)]
    out_path = SAMPLES_DIR / f"samples_{set_name}_{model_safe}.json"
    out_path.write_text(json.dumps(sampled, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[baseline] saved {k} sampled examples to {out_path}")


# ========================================================================== #
# compute_or_read_baseline                                                    #
# ========================================================================== #

def _baseline_needs_llm_judge(set_name: str) -> bool:
    return set_name.startswith("alpaca_") or set_name.endswith("_open")


def compute_or_read_baseline(
        model: Any,
        set_name: str,
        *,
        tokenizer: Any = None,
        out_dir: Optional[str] = None,
        sample_fraction: float = 1.0,
        alpaca_batch_size: int = 32,
        required_concepts: Optional[List[str]] = None,
        skip_llm_judge: bool = False,
) -> Optional[BaselineResult]:
    """Return the baseline result for ``set_name``, reading or computing as needed.

    When ``skip_llm_judge`` is set, LLM-judge sets (Alpaca, open QA) are read
    from cache only; returns ``None`` if no cache exists.
    """
    tm = ensure_wrapped_model(model, tokenizer)
    model_name = tm.tokenizer_name()
    model_safe = _model_safe_name(model_name)

    baseline_root = Path(out_dir) if out_dir is not None else DEFAULT_BASELINE_DIR
    model_dir = baseline_root / model_safe
    model_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = model_dir / f"baseline_{set_name}.json"

    if skip_llm_judge and _baseline_needs_llm_judge(set_name):
        if baseline_path.exists():
            data = json.loads(baseline_path.read_text(encoding="utf-8"))
            missing: set = set()
            if required_concepts and "per_concept" in data.get("meta", {}):
                existing = set(data["meta"]["per_concept"].keys())
                missing = set(required_concepts) - existing
            if not missing:
                return BaselineResult(
                    set_name=data["set_name"],
                    n_questions=data["n_questions"],
                    metrics=data["metrics"],
                    meta=data.get("meta", {}),
                )
        print(f"[baseline] skip_llm_judge: skipping {set_name} (no cache)")
        return None

    if baseline_path.exists():
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
        missing: set = set()
        if required_concepts and "per_concept" in data.get("meta", {}):
            existing = set(data["meta"]["per_concept"].keys())
            missing = set(required_concepts) - existing
        if not missing:
            return BaselineResult(
                set_name=data["set_name"],
                n_questions=data["n_questions"],
                metrics=data["metrics"],
                meta=data.get("meta", {}),
            )
        print(f"[baseline] missing concepts {sorted(missing)} in {set_name}; "
              f"recomputing baseline...")

    print(f"[baseline] computing baseline for set={set_name}, model={model_name}")

    if set_name.startswith("mmlu_"):
        metrics, meta, records_for_sample, correct_ids = _compute_mmlu_baseline(tm, set_name)
    elif set_name.startswith("alpaca_"):
        metrics, meta, records_for_sample, correct_ids = _compute_alpaca_baseline(
            tm, set_name, alpaca_batch_size,
        )
    else:
        metrics, meta, records_for_sample, correct_ids = _compute_qa_baseline(tm, set_name)

    meta.setdefault("model_name", model_name)
    _save_sample(set_name, model_safe, records_for_sample, fraction=sample_fraction)

    data = {
        "set_name": set_name,
        "n_questions": meta.get("total", meta.get("num_examples", 0)),
        "metrics": metrics,
        "meta": meta,
    }
    baseline_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"[baseline] saved baseline to {baseline_path}")

    if correct_ids is not None and not set_name.startswith("alpaca_"):
        correct_path = model_dir / baseline_path.name.replace("baseline_", "correct_ids_")
        correct_path.write_text(json.dumps(correct_ids, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print(f"[baseline] saved {len(correct_ids)} correct IDs to {correct_path}")

    return BaselineResult(
        set_name=set_name,
        n_questions=data["n_questions"],
        metrics=metrics,
        meta=meta,
    )


# ========================================================================== #
# Per-set computation helpers                                                 #
# ========================================================================== #

def _compute_mmlu_baseline(tm, set_name):
    split = set_name.split("_", 1)[1]
    indices = load_mmlu_indices(split)
    items = load_mmlu_items(indices)

    correct_gen, invalid_gen, rec_gen = evaluate_mmlu_generation(tm, items)

    total = len(items)
    metrics = {
        "accuracy_generation": correct_gen / total if total else 0.0,
        "invalid_rate_generation": invalid_gen / total if total else 0.0,
    }
    meta = {
        "num_correct_generation": correct_gen,
        "num_invalid_generation": invalid_gen,
        "total": total,
        "split": split,
    }
    correct_ids = [int(idx) for idx, rec in zip(indices, rec_gen)
                   if rec.get("is_correct_generation", False)]
    return metrics, meta, rec_gen, correct_ids


def _compute_alpaca_baseline(tm, set_name, alpaca_batch_size):
    split = set_name.split("_", 1)[1]
    evaluator = GeminiEvaluator()
    instruct, fluency, records = evaluate_alpaca(
        tm, evaluator, split, batch_size=alpaca_batch_size,
    )
    total = len(instruct)
    metrics = {
        "mean_instruct_score": float(np.mean(instruct)) if total else 0.0,
        "mean_fluency_score":  float(np.mean(fluency))  if total else 0.0,
    }
    meta = {"split": split, "num_examples": total, "total": total}
    return metrics, meta, records, None


def _compute_qa_baseline(tm, set_name):
    parts = set_name.split("_")
    if len(parts) != 3:
        raise ValueError(f"Bad QA/SimDom set_name: {set_name!r}")
    kind, split, mode = parts
    if kind not in {"qa", "simdom"}:
        raise ValueError(f"Unknown QA/SimDom kind in {set_name!r}")
    if split not in {"train", "test"}:
        raise ValueError(f"split must be train/test in {set_name!r}")
    if mode not in {"open", "mc"}:
        raise ValueError(f"mode must be open/mc in {set_name!r}")
    set_core = f"{kind}_{split}"

    if mode == "open":
        return _compute_qa_open(tm, set_name, kind, split, set_core)
    return _compute_qa_mc(tm, set_name, kind, split, set_core)


def _compute_qa_open(tm, set_name, kind, split, set_core):
    items = load_open_qa_examples(set_core)
    total = len(items)
    evaluator = GeminiEvaluator()
    correct, records = evaluate_open_qa(tm, evaluator, items)

    correct_ids: Dict[str, List[str]] = {}
    per_concept: Dict[str, Dict[str, Any]] = {}
    for it, rec in zip(items, records):
        c = it.concept
        if rec.get("is_correct", False):
            correct_ids.setdefault(c, []).append(stable_item_id(it))
        pc = per_concept.setdefault(c, {"num_correct": 0, "total": 0})
        pc["total"] += 1
        if rec["is_correct"]:
            pc["num_correct"] += 1
    for c, pc in per_concept.items():
        pc["accuracy"] = pc["num_correct"] / pc["total"] if pc["total"] else 0.0

    metrics = {"overall_accuracy": correct / total if total else 0.0}
    meta = {
        "kind": kind, "split": split,
        "num_correct": correct, "total": total,
        "per_concept": per_concept,
    }
    return metrics, meta, records, correct_ids


def _compute_qa_mc(tm, set_name, kind, split, set_core):
    raw_items = load_mc_qa_items(set_core)
    prepared = prepare_mc_items(raw_items)
    total = len(prepared)

    correct_gen, invalid_gen, rec_gen = evaluate_mc_generation(tm, prepared)

    correct_ids: Dict[str, List[str]] = {}
    per_concept: Dict[str, Dict[str, Any]] = {}
    for it, rec_g in zip(raw_items, rec_gen):
        c = it.concept
        if rec_g.get("is_correct", False):
            correct_ids.setdefault(c, []).append(stable_item_id(it))
        pc = per_concept.setdefault(c, {
            "num_correct_generation": 0,
            "num_invalid_generation": 0,
            "total": 0,
        })
        pc["total"] += 1
        if rec_g["is_correct"]:
            pc["num_correct_generation"] += 1
        if not rec_g["is_valid"]:
            pc["num_invalid_generation"] += 1
    for c, pc in per_concept.items():
        t = pc["total"]
        pc["accuracy_generation"] = pc["num_correct_generation"] / t if t else 0.0
        pc["invalid_rate_generation"] = pc["num_invalid_generation"] / t if t else 0.0

    metrics = {
        "accuracy_generation":     correct_gen / total if total else 0.0,
        "invalid_rate_generation": invalid_gen / total if total else 0.0,
    }
    meta = {
        "kind": kind, "split": split,
        "num_correct_generation": correct_gen,
        "num_invalid_generation": invalid_gen,
        "total": total,
        "per_concept": per_concept,
    }
    return metrics, meta, rec_gen, correct_ids


# ========================================================================== #
# Mode-level aggregator                                                       #
# ========================================================================== #

def get_baselines_for_mode(
        model: Any,
        mode: str,
        *,
        baseline_out_dir: Optional[Path] = None,
        tokenizer: Any = None,
        alpaca_batch_size: int = 32,
        required_concepts: Optional[List[str]] = None,
        skip_llm_judge: bool = False,
) -> Dict[str, Any]:
    """Compute (or read) baselines for ``mode``.

    Returns ``mmlu_acc`` always. Other keys are present only when loaded or
    computed: ``qa_per_concept``, ``simdom_per_concept``, ``alpaca_instr``,
    ``alpaca_flu``. With ``skip_llm_judge``, LLM-judge sets are omitted unless
    already cached on disk.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode {mode!r}")

    split = "train" if mode.startswith("train") else "test"
    is_open = mode.endswith("open")
    mode_suffix = "open" if is_open else "mc"

    out_dir_str = str(baseline_out_dir) if baseline_out_dir is not None \
        else str(DEFAULT_BASELINE_DIR)

    tm = ensure_wrapped_model(model, tokenizer)

    qa_base: Optional[BaselineResult] = None
    sim_base: Optional[BaselineResult] = None
    if not (skip_llm_judge and is_open):
        qa_base = compute_or_read_baseline(
            tm, f"qa_{split}_{mode_suffix}",
            out_dir=out_dir_str, required_concepts=required_concepts,
            skip_llm_judge=skip_llm_judge,
        )
        sim_base = compute_or_read_baseline(
            tm, f"simdom_{split}_{mode_suffix}",
            out_dir=out_dir_str, required_concepts=required_concepts,
            skip_llm_judge=skip_llm_judge,
        )

    mmlu_base = compute_or_read_baseline(
        tm, f"mmlu_{split}", out_dir=out_dir_str,
    )
    alpaca_base = compute_or_read_baseline(
        tm, f"alpaca_{split}", out_dir=out_dir_str,
        alpaca_batch_size=alpaca_batch_size,
        skip_llm_judge=skip_llm_judge,
    )

    if mmlu_base is None:
        raise RuntimeError(f"MMLU baseline missing for {mode!r}")

    out: Dict[str, Any] = {
        "mmlu_acc": float(mmlu_base.metrics.get("accuracy_generation", 0.0)),
    }
    if qa_base is not None:
        out["qa_per_concept"] = qa_base.meta.get("per_concept", {})
    if sim_base is not None:
        out["simdom_per_concept"] = sim_base.meta.get("per_concept", {})
    if alpaca_base is not None:
        out["alpaca_instr"] = float(
            alpaca_base.metrics.get("mean_instruct_score", 0.0))
        out["alpaca_flu"] = float(
            alpaca_base.metrics.get("mean_fluency_score", 0.0))
    return out


# ========================================================================== #
# Concept-level baseline lookup                                               #
# ========================================================================== #

def get_concept_baseline(
        baselines: Dict[str, Any],
        concept_name: str,
        kind: str,
        mode: str,
) -> Optional[float]:
    """Look up the per-concept baseline accuracy for ``concept_name``."""
    per = baselines.get("qa_per_concept" if kind == "qa" else "simdom_per_concept", {})
    entry = per.get(concept_name)
    if not isinstance(entry, dict):
        return None

    if mode.endswith("open"):
        return entry.get("accuracy")
    return entry.get("accuracy_generation")


__all__ = [
    "DEFAULT_BASELINE_DIR", "SAMPLES_DIR",
    "VALID_MODES",
    "compute_or_read_baseline",
    "get_baselines_for_mode",
    "get_concept_baseline",
]
