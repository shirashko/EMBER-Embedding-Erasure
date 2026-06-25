"""Baseline cache for WMDP MCQ evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ember.evals.baselines import _model_safe_name
from ember.evals.model_wrap import ensure_wrapped_model
from ember.wmdp.corpora import WMDP_DOMAINS
from ember.wmdp.mcq import load_auxiliary_mcqs, load_primary_mcq
from ember.wmdp.mcq_eval import evaluate_wmdp_domain
from ember.wmdp.paths import resolve_data_root

DEFAULT_BASELINE_DIR = Path(__file__).resolve().parents[2] / "data" / "baselines"


def _wmdp_baseline_path(model_safe: str, mode: str, baseline_root: Path) -> Path:
    return baseline_root / model_safe / f"baseline_wmdp_{mode}.json"


def _per_concept_entry(acc: float, total: int) -> Dict[str, Any]:
    return {
        "num_correct_generation": int(round(acc * total)),
        "num_invalid_generation": 0,
        "total": total,
        "accuracy_generation": acc,
        "invalid_rate_generation": 0.0,
    }


def compute_wmdp_baselines(
        model: Any,
        mode: str,
        *,
        data_root: str,
        tokenizer: Any = None,
        baseline_out_dir: Optional[Path] = None,
        required_concepts: Optional[List[str]] = None,
        batch_size: int = 8,
) -> Dict[str, Any]:
    """Compute or load cached WMDP MCQ baselines for ``mode``.

    Returns a dict compatible with :func:`ember.evals.baselines.get_baselines_for_mode`
    (``qa_per_concept``, ``simdom_per_concept``, plus ``mmlu_acc`` placeholder).
    """
    if mode not in ("train_mc", "test_mc"):
        raise ValueError(
            f"WMDP MCQ baselines only support train_mc/test_mc, got {mode!r}"
        )

    tm = ensure_wrapped_model(model, tokenizer)
    model_safe = _model_safe_name(tm.tokenizer_name())
    baseline_root = Path(baseline_out_dir) if baseline_out_dir else DEFAULT_BASELINE_DIR
    baseline_root.mkdir(parents=True, exist_ok=True)
    path = _wmdp_baseline_path(model_safe, mode, baseline_root)

    concepts = [c.lower() for c in (required_concepts or sorted(WMDP_DOMAINS))]
    bad = [c for c in concepts if c not in WMDP_DOMAINS]
    if bad:
        raise ValueError(f"WMDP baselines require domains in {sorted(WMDP_DOMAINS)}, got {bad}")

    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        existing = set(data.get("qa_per_concept", {}))
        if set(concepts) <= existing:
            return data

    root = resolve_data_root(data_root)
    qa_per: Dict[str, Dict[str, Any]] = {}
    sim_per: Dict[str, Dict[str, Any]] = {}

    for domain in concepts:
        qa_acc, sim_acc, _ = evaluate_wmdp_domain(
            tm, domain, str(root), batch_size=batch_size,
        )
        primary = load_primary_mcq(domain, root)
        aux = load_auxiliary_mcqs(domain, root)
        aux_total = sum(a.n_items for a in aux) if aux else 0
        qa_per[domain] = _per_concept_entry(qa_acc, primary.n_items)
        sim_per[domain] = _per_concept_entry(
            sim_acc, aux_total or primary.n_items,
        )

    out = {
        "qa_per_concept": qa_per,
        "simdom_per_concept": sim_per,
        "data_root": str(root),
        "mode": mode,
    }
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[baseline] saved WMDP MCQ baselines to {path}")
    return out


__all__ = ["compute_wmdp_baselines"]
