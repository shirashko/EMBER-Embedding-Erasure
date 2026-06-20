"""Open-ended QA evaluation (Gemini-as-judge).

Generates a free-form answer from the model, then asks Gemini whether
the answer is correct. Gemini judging is parallelized across 30 threads.
"""
from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List, Tuple

from ember.evals.gemini import GeminiBadFinishError, GeminiEvaluator


def evaluate_open_qa(
        model: Any,
        evaluator: GeminiEvaluator,
        items: List[Any],
        batch_size: int = 64,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Open-QA eval with Gemini scoring.

    Returns ``(correct_count, per_question_records)``. Gemini refusals are
    retried up to 3x; persistent failures default the item to incorrect.
    """
    n = len(items)
    prompts = [f"Question: {ex.question}\nAnswer:" for ex in items]

    answers: List[str] = []
    for i in range(0, n, batch_size):
        batch = prompts[i:i + batch_size]
        answers.extend(model.generate_multiple(
            batch, max_new_tokens=200, do_sample=False,
            batch_size=len(batch), verbose=False,
        ))
    if len(answers) != n:
        raise RuntimeError(
            f"Open-QA generation count mismatch: {n} questions, {len(answers)} answers"
        )

    correct = 0
    records: List[Any] = [None] * n

    def grade(idx: int, ex: Any, attempted: str) -> Tuple[int, Any, str, bool]:
        last_exc: Exception | None = None
        for _ in range(3):
            try:
                is_correct = evaluator.judge_open_qa(
                    question=ex.question, answer=ex.answer, attempted=attempted,
                )
                return idx, ex, attempted, is_correct
            except RuntimeError as e:
                if not isinstance(e, GeminiBadFinishError) and type(e) is not RuntimeError:
                    raise
                last_exc = e
        extra = (f" finish_reason={last_exc.finish_reason}"
                 if isinstance(last_exc, GeminiBadFinishError) else "")
        print(
            f"[Gemini refused/failed{extra}] open-QA item {idx} default to incorrect\n"
            f"  Question:     {ex.question!r}\n"
            f"  Model answer: {attempted!r}\n"
            f"  Error:        {last_exc}"
        )
        return idx, ex, attempted, False

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as pool:
        futures = [pool.submit(grade, i, ex, answers[i])
                   for i, ex in enumerate(items)]
        for fut in concurrent.futures.as_completed(futures):
            idx, ex, attempted, is_correct = fut.result()
            if is_correct:
                correct += 1
            records[idx] = {
                "concept": ex.concept,
                "subset": ex.subset,
                "split": ex.split,
                "question": ex.question,
                "gold_answer": ex.answer,
                "model_answer": attempted,
                "is_correct": is_correct,
            }
    return correct, records


__all__ = ["evaluate_open_qa"]
