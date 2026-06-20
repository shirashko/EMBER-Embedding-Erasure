"""Alpaca-Eval scoring (Gemini as judge).

For each instruction in the configured split:
1. Get the model's completion (via ``model.generate_multiple``).
2. Ask Gemini to score 0..2 on instruction-relevance AND fluency separately.

Gemini retries on bad-finish/RuntimeError up to 3 times with progressive
truncation (full → 150 chars → 50 chars). Persistent failure scores the
item 0 for both axes.

Instructions come from the public ``tatsu-lab/alpaca_eval`` dataset fetched
via ``hf_hub_download``.
"""
from __future__ import annotations

import concurrent.futures
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

from huggingface_hub import hf_hub_download

from ember.local_datasets import load_alpaca_indices
from ember.evals.gemini import GeminiBadFinishError, GeminiEvaluator

_TRUNC_LENGTHS: List[int | None] = [None, 150, 50]


def load_alpaca_eval_local() -> List[Dict[str, Any]]:
    """Load alpaca_eval.json directly from the HF hub."""
    path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        repo_type="dataset",
        filename="alpaca_eval.json",
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_alpaca(
        model: Any,
        evaluator: GeminiEvaluator,
        split: str,
        batch_size: int = 32,
) -> Tuple[List[int], List[int], List[Dict[str, Any]]]:
    """Return ``(instruct_scores, fluency_scores, records)`` for the split.

    Scores are integers in ``{0, 1, 2}``; one per item per axis.
    """
    indices = load_alpaca_indices(split)

    items = load_alpaca_eval_local()
    instructions_full = [ex["instruction"] for ex in items if "instruction" in ex]
    instructions = [instructions_full[i] for i in indices]

    generations = model.generate_multiple(
        instructions,
        max_new_tokens=200,
        do_sample=False,
        batch_size=batch_size,
        verbose=False,
    )

    n = len(instructions)
    instruct_scores: List[int] = [0] * n
    fluency_scores: List[int] = [0] * n
    records: List[Any] = [None] * n

    fallback_count = [0]
    fallback_lock = threading.Lock()

    def score(idx: int, inst: str, completion: str
              ) -> Tuple[int, str, str, int, int]:
        last_exc: Exception | None = None
        for trunc in _TRUNC_LENGTHS:
            c = completion if trunc is None else completion[:trunc]
            try:
                s_inst = evaluator.score_alpaca_instruct(inst, c)
                s_flu = evaluator.score_alpaca_fluency(c + ".")
                return idx, inst, completion, s_inst, s_flu
            except RuntimeError as e:
                if not isinstance(e, GeminiBadFinishError) and type(e) is not RuntimeError:
                    raise
                last_exc = e
        with fallback_lock:
            fallback_count[0] += 1
        extra = (f" finish_reason={last_exc.finish_reason}"
                 if isinstance(last_exc, GeminiBadFinishError) else "")
        print(
            f"[Gemini refused/failed{extra}] alpaca item {idx} default to 0/0\n"
            f"  Task:         {inst!r}\n"
            f"  Model answer: {completion!r}\n"
            f"  Error:        {last_exc}"
        )
        return idx, inst, completion, 0, 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as pool:
        futures = [pool.submit(score, i, instructions[i], generations[i])
                   for i in range(n)]
        for fut in concurrent.futures.as_completed(futures):
            idx, inst, completion, s_inst, s_flu = fut.result()
            instruct_scores[idx] = s_inst
            fluency_scores[idx] = s_flu
            records[idx] = {
                "instruction": inst,
                "completion": completion,
                "instruct_score": s_inst,
                "fluency_score": s_flu,
            }

    if fallback_count[0] > 0:
        print(f"[Alpaca eval] {fallback_count[0]}/{n} items scored 0 "
              f"due to Gemini bad finish_reason.")
    return instruct_scores, fluency_scores, records


__all__ = ["load_alpaca_eval_local", "evaluate_alpaca"]
