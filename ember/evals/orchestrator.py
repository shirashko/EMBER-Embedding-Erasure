"""Top-level eval orchestrator: ``evaluate_model_for_mode``.

Runs the full eval flow for a single (model, concept, mode) combination:
    1. MMLU (generation) -- optional via ``eval_mmlu``.
    2. Early-stop if ``mmlu_frac < min_mmlu``.
    3. Concept QA (open or MC).
    4. Early-stop if ``qa_frac > max_qa_acc``.
    5. SimDom QA (open or MC).
    6. Compute ``harmonic`` from qa-drop, mmlu, simdom.
    7. Alpaca (optional via ``eval_alpaca``) + ``harmonic_alpaca``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ember.local_datasets import (
    load_mc_qa_items,
    load_mmlu_indices,
    load_open_qa_examples,
)
from ember.evals.alpaca import evaluate_alpaca
from ember.evals.baselines import get_concept_baseline
from ember.evals.gemini import GeminiEvaluator
from ember.evals.harmonic import harmonic_mean
from ember.evals.mc import (
    evaluate_mc_generation,
    prepare_mc_items,
    stable_item_id,
)
from ember.evals.mmlu import (
    evaluate_mmlu_generation,
    load_mmlu_items,
)
from ember.evals.model_wrap import ensure_wrapped_model
from ember.evals.open_qa import evaluate_open_qa
from ember.evals.schema import GeminiTokenStats

VALID_MODES = ("train_open", "train_mc", "test_open", "test_mc")


def evaluate_model_for_mode(
        model: Any,
        baselines: Dict[str, Any],
        *,
        mode: str,
        concept_name: Optional[str] = None,
        min_mmlu: Optional[float] = 0.6,
        max_qa_acc: Optional[float] = 0.9,
        min_alpaca: Optional[float] = None,
        eval_alpaca: bool = False,
        eval_mmlu: bool = True,
        eval_qa: bool = True,
        eval_simdom: bool = True,
        precomputed_metrics: Optional[Dict[str, float]] = None,
        tokenizer: Any = None,
        custom_indices_qa: Optional[List[int]] = None,
        custom_indices_mmlu: Optional[List[int]] = None,
        dynamic_baselines: Optional[Dict[str, Any]] = None,
        alpaca_batch_size: int = 32,
        skip_llm_judge: bool = False,
) -> Tuple[Dict[str, float], Dict[str, List[Dict[str, Any]]]]:
    """Full eval for one (model, concept, mode). Returns ``(metrics, records_by_set)``.

    Early-stop semantics:
        - If ``min_mmlu`` is set and ``mmlu_frac < min_mmlu``, return early.
        - If ``max_qa_acc`` is set and ``qa_frac > max_qa_acc``, return early.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected {VALID_MODES}")
    tm = ensure_wrapped_model(model, tokenizer)
    gem_stats = GeminiTokenStats()
    evaluator: Optional[GeminiEvaluator] = None

    split = "train" if mode.startswith("train") else "test"
    is_open = mode.endswith("open")

    if skip_llm_judge:
        eval_alpaca = False
        if is_open:
            eval_qa = False
            eval_simdom = False

    qa_base = (get_concept_baseline(baselines, concept_name, "qa", mode)
               if eval_qa else None)
    sim_base = (get_concept_baseline(baselines, concept_name, "simdom", mode)
                if eval_simdom else None)
    if eval_qa and qa_base is None:
        raise KeyError(
            f"No QA baseline for concept {concept_name!r} in mode {mode}.\n"
            f"baselines={baselines!r}"
        )
    if eval_simdom and sim_base is None:
        raise KeyError(
            f"No Simdom baseline for concept {concept_name!r} in mode {mode}.\n"
            f"baselines={baselines!r}"
        )
    mmlu_base = baselines["mmlu_acc"]
    alp_instr_base = baselines.get("alpaca_instr") if eval_alpaca else None
    alp_flu_base = baselines.get("alpaca_flu") if eval_alpaca else None
    if eval_alpaca and (alp_instr_base is None or alp_flu_base is None):
        raise KeyError(
            f"Alpaca baselines required for eval_alpaca in mode {mode!r} "
            f"but missing from baselines={baselines!r}"
        )

    metrics: Dict[str, float] = {}
    records_by_set: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------ #
    # MMLU                                                               #
    # ------------------------------------------------------------------ #
    mmlu_frac: float
    if eval_mmlu:
        mmlu_indices = load_mmlu_indices(split)
        if custom_indices_mmlu is not None:
            mmlu_indices = [mmlu_indices[i] for i in custom_indices_mmlu]
        mmlu_items = load_mmlu_items(mmlu_indices)

        if dynamic_baselines is not None and "mmlu" in dynamic_baselines:
            correct_list = dynamic_baselines["mmlu"]
            base_correct = sum(1 for idx in mmlu_indices if int(idx) in correct_list)
            mmlu_base = (base_correct / len(mmlu_indices)) if mmlu_indices else 0.0

        mmlu_correct, mmlu_invalid, mmlu_records = evaluate_mmlu_generation(tm, mmlu_items)
        mmlu_total = len(mmlu_items)
        mmlu_acc = mmlu_correct / mmlu_total if mmlu_total else 0.0
        records_by_set[f"mmlu_{split}"] = mmlu_records

        mmlu_frac = min(1.0, max((mmlu_acc - 0.25) / (mmlu_base - 0.25), 0.0))
        metrics["mmlu_acc"] = mmlu_acc
        metrics["mmlu_frac"] = mmlu_frac
        metrics["mmlu_invalid"] = mmlu_invalid / mmlu_total if mmlu_total else 0.0

        if min_mmlu is not None and mmlu_frac < min_mmlu:
            for k in ("qa_acc", "qa_frac", "simdom_acc", "simdom_frac",
                      "efficacy", "specificity", "harmonic"):
                metrics[k] = 0.0
            if not is_open:
                metrics["qa_invalid"] = 0.0
                metrics["simdom_invalid"] = 0.0
            return _finalize(metrics, records_by_set, gem_stats)
    else:
        pre = precomputed_metrics or {}
        for k in ("mmlu_acc", "mmlu_frac", "mmlu_invalid"):
            metrics[k] = float(pre.get(k, float("nan")))
        mmlu_frac = metrics["mmlu_frac"]

    # ------------------------------------------------------------------ #
    # QA + SimDom                                                        #
    # ------------------------------------------------------------------ #
    if not eval_qa:
        pre = precomputed_metrics or {}
        for k in ("qa_acc", "qa_frac") + (("qa_invalid",) if not is_open else ()):
            metrics[k] = float(pre.get(k, float("nan")))
        qa_acc = metrics["qa_acc"]
        qa_frac = metrics["qa_frac"]
    if not eval_simdom:
        pre = precomputed_metrics or {}
        for k in ("simdom_acc", "simdom_frac") + (("simdom_invalid",) if not is_open else ()):
            metrics[k] = float(pre.get(k, float("nan")))
        sim_acc = metrics["simdom_acc"]
        sim_frac = metrics["simdom_frac"]

    if is_open:
        if eval_qa:
            if evaluator is None:
                evaluator = GeminiEvaluator(token_stats=gem_stats)
            qa_acc, qa_frac, _ = _eval_open_set(
                tm, evaluator, f"qa_{split}", concept_name, qa_base,
                records_by_set,
            )
            metrics["qa_acc"] = qa_acc
            metrics["qa_frac"] = qa_frac
            if max_qa_acc is not None and qa_frac > max_qa_acc:
                for k in ("simdom_acc", "simdom_frac",
                          "efficacy", "specificity", "harmonic"):
                    metrics[k] = 0.0
                return _finalize(metrics, records_by_set, gem_stats)

        if eval_simdom:
            if evaluator is None:
                evaluator = GeminiEvaluator(token_stats=gem_stats)
            sim_acc, sim_frac, _ = _eval_open_set(
                tm, evaluator, f"simdom_{split}", concept_name, sim_base,
                records_by_set,
            )
            metrics["simdom_acc"] = sim_acc
            metrics["simdom_frac"] = sim_frac
    else:
        if eval_qa:
            qa_acc, qa_frac, qa_invalid_rate = _eval_mc_set(
                tm, f"qa_{split}", concept_name, qa_base,
                records_by_set,
                custom_indices=custom_indices_qa,
                dynamic_correct_ids=(dynamic_baselines or {}).get("qa", {}).get(concept_name),
            )
            metrics["qa_acc"] = qa_acc
            metrics["qa_frac"] = qa_frac
            metrics["qa_invalid"] = qa_invalid_rate

        if eval_simdom:
            sim_acc, sim_frac, sim_invalid_rate = _eval_mc_set(
                tm, f"simdom_{split}", concept_name, sim_base,
                records_by_set,
                custom_indices=custom_indices_qa,
                dynamic_correct_ids=(dynamic_baselines or {}).get("simdom", {}).get(concept_name),
            )
            metrics["simdom_acc"] = sim_acc
            metrics["simdom_frac"] = sim_frac
            metrics["simdom_invalid"] = sim_invalid_rate

    efficacy = 1.0 - qa_frac if np.isfinite(qa_frac) else float("nan")
    specificity = harmonic_mean([mmlu_frac, sim_frac])
    metrics["efficacy"] = efficacy
    metrics["specificity"] = specificity
    metrics["harmonic"] = harmonic_mean([efficacy, specificity])

    # ------------------------------------------------------------------ #
    # Alpaca (optional)                                                  #
    # ------------------------------------------------------------------ #
    if eval_alpaca:
        if evaluator is None:
            evaluator = GeminiEvaluator(token_stats=gem_stats)
        instruct, fluency, alp_records = evaluate_alpaca(
            tm, evaluator, split, batch_size=alpaca_batch_size,
        )
        alpaca_instr = float(np.mean(instruct)) if instruct else 0.0
        alpaca_flu = float(np.mean(fluency)) if fluency else 0.0
        alp_instr_frac = (min(1.0, alpaca_instr / alp_instr_base)
                          if alp_instr_base and alp_instr_base > 0 else float("nan"))
        alp_flu_frac = (min(1.0, alpaca_flu / alp_flu_base)
                        if alp_flu_base and alp_flu_base > 0 else float("nan"))

        coherence = harmonic_mean([alp_instr_frac, alp_flu_frac])
        harmonic_alpaca = harmonic_mean([efficacy, specificity, coherence])

        if min_alpaca is not None and (alp_instr_frac < min_alpaca
                                       or alp_flu_frac < min_alpaca):
            harmonic_alpaca = 0.0

        records_by_set[f"alpaca_{split}"] = alp_records
        metrics["alpaca_instr"] = alpaca_instr
        metrics["alp_instr_frac"] = alp_instr_frac
        metrics["alpaca_flu"] = alpaca_flu
        metrics["alp_flu_frac"] = alp_flu_frac
        metrics["coherence"] = coherence
        metrics["harmonic_alpaca"] = harmonic_alpaca

    return _finalize(metrics, records_by_set, gem_stats)


# ========================================================================== #
# Helpers                                                                     #
# ========================================================================== #

def _finalize(metrics, records_by_set, gem_stats):
    metrics["gemini_prompt_tokens"] = gem_stats.prompt_tokens
    metrics["gemini_output_tokens"] = gem_stats.output_tokens
    metrics["gemini_total_tokens"] = gem_stats.total_tokens
    metrics["gemini_calls"] = gem_stats.calls
    return metrics, records_by_set


def _eval_open_set(tm, evaluator, set_core, concept_name, base,
                   records_by_set) -> Tuple[float, float, int]:
    items = load_open_qa_examples(set_core, concept=concept_name)
    correct, records = evaluate_open_qa(tm, evaluator, items)
    total = len(items)
    acc = correct / total if total else 0.0
    frac = (min(1.0, acc / base) if base and base > 0 else float("nan"))
    records_by_set[f"{set_core}_open"] = records
    return acc, frac, total


def _eval_mc_set(tm, set_core, concept_name, base,
                 records_by_set, *, custom_indices,
                 dynamic_correct_ids) -> Tuple[float, float, float]:
    raw_items = load_mc_qa_items(set_core, concept=concept_name)
    if custom_indices is not None:
        raw_items = [raw_items[i] for i in custom_indices]

    if dynamic_correct_ids is not None:
        base_correct = sum(1 for it in raw_items
                           if stable_item_id(it) in dynamic_correct_ids)
        base = base_correct / len(raw_items) if raw_items else 0.0

    prepared = prepare_mc_items(raw_items)
    total = len(prepared)

    correct, invalid, records = evaluate_mc_generation(tm, prepared)

    acc = correct / total if total else 0.0
    if (base - 0.25) == 0:
        frac = 0.0
    else:
        frac = min(1.0, max((acc - 0.25) / (base - 0.25), 0.0))

    records_by_set[f"{set_core}_mc"] = records
    return acc, frac, (invalid / total if total else 0.0)


__all__ = [
    "VALID_MODES",
    "evaluate_model_for_mode",
]
