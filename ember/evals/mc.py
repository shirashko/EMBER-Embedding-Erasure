"""Multiple-choice eval: concept QA and SimDom questions.

The model is prompted for one letter (A/B/C/D) and the first letter in the
output is parsed; invalid generations are tracked separately. Option order is
shuffled deterministically per question via :func:`prepare_mc_items` so eval
is reproducible.
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Dict, List, Tuple

from ember.evals.schema import LETTER_CHOICES, PreparedMCItem


def stable_item_id(obj: Any) -> str:
    """Stable identifier for a QA item. Same item across runs -> same id."""
    qid = getattr(obj, "question_id", None)
    if qid is None and isinstance(obj, dict):
        qid = obj.get("question_id")
    if qid is not None:
        return str(qid)

    parts: List[str] = []
    for key in ("concept", "subset", "split", "question", "answer", "correct_answer"):
        v = getattr(obj, key, None)
        if v is None and isinstance(obj, dict):
            v = obj.get(key)
        if v is not None:
            parts.append(f"{key}={v}")

    opts = getattr(obj, "options", None)
    if opts is None and isinstance(obj, dict):
        opts = obj.get("options")
    if opts is not None:
        try:
            parts.append("options=" + "|".join(list(opts)))
        except Exception:
            parts.append(f"options={opts!r}")

    sig = "||".join(parts) or repr(obj)
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()


def build_mc_prompt(question: str, options_by_letter: Dict[str, str]) -> str:
    """Format an MC question + 4 options as a single prompt string."""
    lines = [
        "Choose the single best answer.",
        f"Question: {question}\n",
        "Choices:",
    ]
    for L in LETTER_CHOICES:
        lines.append(f"{L}. {options_by_letter[L]}")
    lines.append("\nAnswer only with one letter (A, B, C, or D) and no formatting.")
    return "\n".join(lines)


def parse_letter(raw: str) -> str:
    """Extract the first standalone A-D letter from a model output. ``""`` if none."""
    if not isinstance(raw, str):
        return ""
    m = re.search(r"\b([A-D])\b", raw)
    return m.group(1) if m else ""


def prepare_mc_items(raw_items: List[Any], seed: int = 42) -> List[PreparedMCItem]:
    """Shuffle each question's options deterministically and assign letters."""
    prepared: List[PreparedMCItem] = []
    for obj in raw_items:
        q = obj.question
        correct = obj.correct_answer
        options = list(obj.options)

        stable_id = stable_item_id(obj)
        stable_hash_int = int(hashlib.sha1(stable_id.encode("utf-8")).hexdigest()[:8], 16)
        per_q_seed = (seed + stable_hash_int) % (2 ** 32)
        random.Random(per_q_seed).shuffle(options)

        if correct not in options:
            raise AssertionError(
                f"Correct answer not found in options.\n"
                f"  Question: {q}\n  Answer: {correct}\n  Options: {options}"
            )

        mapping: Dict[str, str] = {}
        correct_letter: str = ""
        for L, opt in zip(LETTER_CHOICES, options[:4]):
            mapping[L] = opt
            if opt == correct:
                correct_letter = L
        if not correct_letter:
            raise AssertionError(
                f"Correct answer not among the first 4 shuffled options for: {q!r}"
            )

        prepared.append(PreparedMCItem(
            concept=obj.concept,
            subset=obj.subset,
            split=obj.split,
            question=q,
            options_by_letter=mapping,
            correct_letter=correct_letter,
            correct_text=correct,
        ))
    return prepared


# ========================================================================== #
# Evaluators                                                                  #
# ========================================================================== #

def evaluate_mc_generation(
        model: Any,
        items: List[PreparedMCItem],
        max_new_tokens: int = 4,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """One-question-at-a-time MC generation evaluation.

    Returns ``(correct, invalid, per_question_records)``.
    """
    records: List[Dict[str, Any]] = []
    correct = 0
    invalid = 0

    for item in items:
        prompt = build_mc_prompt(item.question, item.options_by_letter)
        wrapped = model.wrap_prompt(prompt)
        raw = model.generate(wrapped, max_new_tokens=max_new_tokens,
                             temperature=0.0, do_sample=False)
        letter = parse_letter(raw)
        is_valid = letter in LETTER_CHOICES
        is_correct = is_valid and (letter == item.correct_letter)

        if not is_valid:
            invalid += 1
        if is_correct:
            correct += 1

        records.append({
            "concept": item.concept,
            "subset": item.subset,
            "split": item.split,
            "question": item.question,
            "options": item.options_by_letter,
            "correct_letter": item.correct_letter,
            "correct_text": item.correct_text,
            "model_raw": raw,
            "model_letter": letter,
            "is_valid": is_valid,
            "is_correct": is_correct,
        })
    return correct, invalid, records


__all__ = [
    "stable_item_id",
    "build_mc_prompt", "parse_letter", "prepare_mc_items",
    "evaluate_mc_generation",
]
