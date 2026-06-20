"""Shared dataclasses + constants used across eval modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

LETTER_CHOICES: List[str] = ["A", "B", "C", "D"]


@dataclass
class PreparedMCItem:
    """A multiple-choice item with letters assigned (after seed-shuffled options)."""
    concept: str
    subset: str
    split: str
    question: str
    options_by_letter: Dict[str, str]
    correct_letter: str
    correct_text: str


@dataclass
class MMLUItem:
    """A single MMLU question with its four choices and the gold letter."""
    question: str
    choices: List[str]
    answer_letter: str


@dataclass
class BaselineResult:
    """The on-disk baseline result for one (set, model) combination."""
    set_name: str
    n_questions: int
    metrics: Dict[str, float]
    meta: Dict[str, Any]


@dataclass
class GeminiTokenStats:
    """Running totals of Gemini API usage during one eval pass."""
    prompt_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, prompt_toks: int = 0, output_toks: int = 0,
            calls: int = 0) -> None:
        self.prompt_tokens += int(prompt_toks)
        self.output_tokens += int(output_toks)
        self.calls += int(calls)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


__all__ = [
    "LETTER_CHOICES",
    "PreparedMCItem", "MMLUItem", "BaselineResult", "GeminiTokenStats",
]
