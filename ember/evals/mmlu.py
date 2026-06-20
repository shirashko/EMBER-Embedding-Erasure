"""MMLU evaluation: load CAIS MMLU items + generation scoring.

Cached on disk at ``data/mmlu_cache.json`` so subsequent runs don't re-hit
the HuggingFace dataset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import datasets as hf_datasets  # type: ignore

from ember.local_datasets import load_mmlu_indices
from ember.evals.mc import build_mc_prompt, parse_letter
from ember.evals.schema import LETTER_CHOICES, MMLUItem

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MMLU_CACHE_PATH = DATA_DIR / "mmlu_cache.json"

_DISK_CACHE: List[Dict[str, Any]] | None = None


def _to_letter(ans: Any) -> str:
    """Normalize an MMLU answer field (int 0..3, str digit, or letter) -> letter."""
    if isinstance(ans, int):
        return LETTER_CHOICES[ans]
    if isinstance(ans, str):
        s = ans.strip().upper()
        if s in LETTER_CHOICES:
            return s
        if s.isdigit():
            idx = int(s)
            if 0 <= idx <= 3:
                return LETTER_CHOICES[idx]
            if 1 <= idx <= 4:
                return LETTER_CHOICES[idx - 1]
    raise ValueError(f"Unrecognized MMLU answer format: {ans!r}")


def load_mmlu_items(indices: List[int]) -> List[MMLUItem]:
    """Return MMLU items for ``indices``, building the cache lazily on first call."""
    global _DISK_CACHE

    if _DISK_CACHE is None:
        if MMLU_CACHE_PATH.exists():
            _DISK_CACHE = json.loads(MMLU_CACHE_PATH.read_text(encoding="utf-8"))
        else:
            train_idx = [int(i) for i in load_mmlu_indices("train")]
            test_idx = [int(i) for i in load_mmlu_indices("test")]
            needed = sorted(set(train_idx) | set(test_idx))

            ds = hf_datasets.load_dataset("cais/mmlu", "all", split="test")
            cache: List[Dict[str, Any]] = []
            for idx in needed:
                ex = ds[int(idx)]
                choices = ex["choices"]
                if not isinstance(choices, list) or len(choices) != 4:
                    continue
                cache.append({
                    "idx": int(idx),
                    "question": ex["question"],
                    "choices": choices,
                    "answer_letter": _to_letter(ex["answer"]),
                })
            MMLU_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            MMLU_CACHE_PATH.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[mmlu] cached {len(cache)} questions to {MMLU_CACHE_PATH}")
            _DISK_CACHE = cache

    by_idx = {int(item["idx"]): item for item in _DISK_CACHE}
    out: List[MMLUItem] = []
    for idx in indices:
        item = by_idx.get(int(idx))
        if item is not None:
            out.append(MMLUItem(
                question=item["question"],
                choices=item["choices"],
                answer_letter=item["answer_letter"],
            ))
    return out


def build_mmlu_prompt(question: str, choices: List[str]) -> str:
    return build_mc_prompt(question, dict(zip(LETTER_CHOICES, choices)))


# ========================================================================== #
# Evaluators                                                                  #
# ========================================================================== #

def evaluate_mmlu_generation(
        model: Any,
        items: List[MMLUItem],
        max_new_tokens: int = 6,
        batch_size: int = 32,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Batched MMLU generation eval. Returns ``(correct, invalid, records)``."""
    prompts_raw = [build_mmlu_prompt(it.question, it.choices) for it in items]
    outputs = model.generate_multiple(
        prompts_raw,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        batch_size=batch_size,
        verbose=False,
    )
    if len(outputs) != len(items):
        raise RuntimeError(
            f"MMLU generation count mismatch: {len(items)} questions, "
            f"{len(outputs)} outputs"
        )

    records: List[Dict[str, Any]] = []
    correct = 0
    invalid = 0
    for it, out in zip(items, outputs):
        pred = parse_letter(out)
        is_valid = pred in LETTER_CHOICES
        is_correct = is_valid and (pred == it.answer_letter)
        if not is_valid:
            invalid += 1
        if is_correct:
            correct += 1
        records.append({
            "question": it.question,
            "choices": dict(zip(LETTER_CHOICES, it.choices)),
            "answer_letter": it.answer_letter,
            "model_raw_generation": out,
            "model_letter_generation": pred,
            "is_valid_generation": is_valid,
            "is_correct_generation": is_correct,
        })
    return correct, invalid, records


__all__ = [
    "MMLU_CACHE_PATH",
    "load_mmlu_items", "build_mmlu_prompt",
    "evaluate_mmlu_generation",
]
