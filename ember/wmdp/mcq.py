"""WMDP multiple-choice evaluation corpora (CRISP JSON format)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ember.wmdp.paths import (
    WMDP_AUXILIARY_MCQ,
    WMDP_PRIMARY_MCQ,
    mcq_json_path,
    resolve_data_root,
)


@dataclass(frozen=True)
class WMDPMCQSet:
    """One MCQ file: questions, answer indices, and choice lists."""

    name: str
    path: Path
    questions: Tuple[str, ...]
    answers: Tuple[int, ...]
    choices: Tuple[Tuple[str, str, str, str], ...]

    @property
    def n_items(self) -> int:
        return len(self.questions)


def load_crisp_mcq_json(path: Path) -> WMDPMCQSet:
    """Load ``{questions, answers, choices}`` JSON used by CRISP ``eval.py``."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing WMDP MCQ file: {path}\n"
            "Run: python scripts/prepare_wmdp_mcq.py --domain all"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = tuple(str(q).strip() for q in raw["questions"])
    answers = tuple(int(a) for a in raw["answers"])
    choices = tuple(tuple(str(c) for c in row) for row in raw["choices"])
    n = min(len(questions), len(answers), len(choices))
    if n == 0:
        raise ValueError(f"No MCQ items in {path}")
    return WMDPMCQSet(
        name=path.stem,
        path=path,
        questions=questions[:n],
        answers=answers[:n],
        choices=choices[:n],
    )


def _load_named(
        data_root: Path,
        domain: str,
        filename: str,
) -> WMDPMCQSet:
    return load_crisp_mcq_json(mcq_json_path(data_root, domain, filename))


def load_primary_mcq(domain: str, data_root: str | Path) -> WMDPMCQSet:
    domain = domain.lower()
    if domain not in WMDP_PRIMARY_MCQ:
        raise ValueError(f"Unknown WMDP domain {domain!r}")
    root = resolve_data_root(data_root)
    return _load_named(root, domain, WMDP_PRIMARY_MCQ[domain])


def load_auxiliary_mcqs(domain: str, data_root: str | Path) -> List[WMDPMCQSet]:
    domain = domain.lower()
    if domain not in WMDP_AUXILIARY_MCQ:
        raise ValueError(f"Unknown WMDP domain {domain!r}")
    root = resolve_data_root(data_root)
    return [
        _load_named(root, domain, name)
        for name in WMDP_AUXILIARY_MCQ[domain]
    ]


def write_crisp_mcq_json(
        path: Path,
        questions: Sequence[str],
        answers: Sequence[int],
        choices: Sequence[Sequence[str]],
) -> int:
    """Write CRISP-format MCQ JSON; return row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "questions": list(questions),
        "answers": [int(a) for a in answers],
        "choices": [list(c) for c in choices],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(payload["questions"])


def rows_to_crisp(
        rows: Sequence[Dict[str, object]],
) -> Tuple[List[str], List[int], List[List[str]]]:
    """Convert HF-style rows (question, choices, answer index) to CRISP lists."""
    questions: List[str] = []
    answers: List[int] = []
    choices: List[List[str]] = []
    for row in rows:
        q = str(row.get("question", "")).strip()
        opts = row.get("choices") or row.get("options")
        ans = row.get("answer")
        if ans is None:
            ans = row.get("answer_idx")
        if not q or not opts or ans is None:
            continue
        opt_list = [str(o).strip() for o in list(opts)[:4]]
        if len(opt_list) != 4:
            continue
        questions.append(q)
        answers.append(int(ans))
        choices.append(opt_list)
    return questions, answers, choices
