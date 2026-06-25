"""Thin glue over ember.evals for grid / validate / test stages."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ember.evals import baselines as _baselines_mod
from ember.evals import orchestrator as _orchestrator_mod

from ember.erasure import io as ember_io

VALID_MODES = ("train_open", "train_mc", "test_open", "test_mc")


def get_baselines(
        model: Any,
        mode: str,
        *,
        model_name: str,
        tokenizer: Any = None,
        alpaca_batch_size: int = 50,
        required_concepts: Optional[List[str]] = None,
        baseline_out_dir: Optional[Path] = None,
        skip_llm_judge: bool = False,
        data_source: str = "ember",
        wmdp_data_root: str = "data/wmdp",
) -> Dict[str, Any]:
    """Compute (or read from cache) the per-mode baseline metrics."""
    _check_mode(mode)
    # Root dir; get_baselines_for_mode appends the safe model name.
    if baseline_out_dir is None:
        baseline_out_dir = ember_io.DATA_DIR / "baselines"
    return _baselines_mod.get_baselines_for_mode(
        model, mode,
        tokenizer=tokenizer,
        baseline_out_dir=baseline_out_dir,
        alpaca_batch_size=alpaca_batch_size,
        required_concepts=required_concepts,
        skip_llm_judge=skip_llm_judge,
        data_source=data_source,
        wmdp_data_root=wmdp_data_root,
    )


def evaluate_model(
        model: Any,
        *,
        baselines: Dict[str, Any],
        concept_name: str,
        mode: str,
        eval_alpaca: bool = False,
        min_mmlu: Optional[float] = None,
        max_qa_acc: Optional[float] = None,
        min_alpaca: Optional[float] = None,
        tokenizer: Any = None,
        alpaca_batch_size: int = 50,
        eval_mmlu: bool = True,
        eval_qa: bool = True,
        eval_simdom: bool = True,
        precomputed_metrics: Optional[Dict[str, float]] = None,
        skip_llm_judge: bool = False,
        data_source: str = "ember",
        wmdp_data_root: str = "data/wmdp",
) -> Tuple[Dict[str, float], Dict[str, List[Dict[str, Any]]]]:
    """Evaluate the currently mutated model and return (metrics, records)."""
    _check_mode(mode)

    raw_metrics, records = _orchestrator_mod.evaluate_model_for_mode(
        model, baselines,
        mode=mode,
        concept_name=concept_name,
        eval_alpaca=eval_alpaca,
        min_mmlu=min_mmlu,
        max_qa_acc=max_qa_acc,
        min_alpaca=min_alpaca,
        tokenizer=tokenizer,
        alpaca_batch_size=alpaca_batch_size,
        eval_mmlu=eval_mmlu,
        eval_qa=eval_qa,
        eval_simdom=eval_simdom,
        precomputed_metrics=precomputed_metrics,
        skip_llm_judge=skip_llm_judge,
        data_source=data_source,
        wmdp_data_root=wmdp_data_root,
    )

    normalized = ember_io.normalize_metrics_for_mode(mode, raw_metrics)
    return normalized, records


def _check_mode(mode: str) -> None:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {VALID_MODES}")


__all__ = [
    "VALID_MODES",
    "get_baselines", "evaluate_model",
]
