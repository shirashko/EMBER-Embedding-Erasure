import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


def _load_json(path: Path | str) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_list(path: Path | str) -> list:
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def _find_concept_entry(data: list, concept_name: str, source_path: str) -> dict:
    for entry in data:
        name = entry.get("concept") or entry.get("Concept")
        if isinstance(name, str) and name.strip() == concept_name:
            return entry
    available = [e.get("concept", e.get("Concept")) for e in data]
    raise ValueError(f"Concept '{concept_name}' not found in {source_path}. Available: {available}")


class ConceptDataset:
    """Supervised dataset for one concept: pairs of (sentence, label).

    Loads all concept sentences from ``data/concept_sentences.json`` (key: ``"sentences"``)
    and all neutral sentences from ``data/neutral_sentences.json``.
    """

    def __init__(
            self,
            concept_name: str,
            concept_path: Path | str = DATA_DIR / "concept_sentences.json",
            neutral_path: Path | str = DATA_DIR / "neutral_sentences.json",
            neutral_sample_seed: Optional[int] = None,
    ) -> None:
        self.concept_name = concept_name
        self.data: List[Tuple[str, str]] = []

        concept_data = _load_json_list(concept_path)
        concept_entry = _find_concept_entry(concept_data, concept_name, concept_path)
        for s in concept_entry.get("sentences", []):
            if s:
                self.data.append((s, concept_name))

        neutral_data = _load_json_list(neutral_path)
        neutral_pairs = [(obj["sentence"], "Neutral") for obj in neutral_data if obj.get("sentence")]
        # When a seed is given, reorder the neutral sentences with a fixed-seed
        # shuffle; otherwise keep file order.
        if neutral_sample_seed is not None:
            neutral_pairs = random.Random(neutral_sample_seed).sample(
                neutral_pairs, len(neutral_pairs))
        self.data.extend(neutral_pairs)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.data[idx]

    def get_batches(self, batch_size: int) -> List[dict]:
        batches: List[dict] = []
        for i in range(0, len(self.data), batch_size):
            batch = self.data[i: i + batch_size]
            if not batch:
                continue
            prompts, labels = zip(*batch)
            batches.append({"prompt": list(prompts), "label": list(labels)})
        return batches

    def as_forget_retain(self, seed: Optional[int] = None) -> dict:
        """Return sentences split by role: ``{"forget": [...], "retain": [...]}``.

        Forget = concept sentences, retain = neutral sentences. When ``seed`` is
        given, both lists are shuffled with a single ``random.Random(seed)``
        (forget first, then retain) so the order matches what erasure expects.
        """
        forget = [s for s, lbl in self.data if lbl == self.concept_name]
        retain = [s for s, lbl in self.data if lbl == "Neutral"]
        if seed is not None:
            rnd = random.Random(seed)
            rnd.shuffle(forget)
            rnd.shuffle(retain)
        return {"forget": forget, "retain": retain}


@dataclass
class OpenQAItem:
    concept: str
    subset: str  # "QA" or "SimdomQA"
    split: str   # "train" or "test"
    question: str
    answer: str


@dataclass
class MCQAItem:
    concept: str
    subset: str  # "QA" or "SimdomQA"
    split: str   # "train" or "test"
    question: str
    options: List[str]
    correct_answer: str


def _parse_set_name(set_name: str) -> Tuple[str, str]:
    parts = set_name.lower().split("_")
    if len(parts) != 2 or parts[1] not in {"train", "test"}:
        raise ValueError(f"Bad set_name: {set_name!r}. Expected '<qa|simdom>_<train|test>'.")
    subset_map = {"qa": "QA", "simdom": "SimdomQA"}
    if parts[0] not in subset_map:
        raise ValueError(f"Unknown QA subset kind in {set_name!r}")
    return subset_map[parts[0]], parts[1]


def load_open_qa_examples(set_name: str, concept: str | None = None) -> List[OpenQAItem]:
    """Load open-ended QA items. set_name: "qa_train", "qa_test", "simdom_train", "simdom_test"."""
    subset, split = _parse_set_name(set_name)
    json_data = _load_json(DATA_DIR / "open_questions.json")
    key = f"{subset}_{split}"
    items: List[OpenQAItem] = []
    for concept_name, concept_data in json_data.items():
        if concept is not None and concept_name != concept:
            continue
        for obj in concept_data.get(key, []):
            q, a = obj.get("q", "").strip(), obj.get("a", "").strip()
            if q and a:
                items.append(OpenQAItem(concept=concept_name, subset=subset, split=split, question=q, answer=a))
    return items


def load_mc_qa_items(set_name: str, concept: str | None = None) -> List[MCQAItem]:
    """Load multiple-choice QA items. set_name: "qa_train", "qa_test", "simdom_train", "simdom_test"."""
    subset, split = _parse_set_name(set_name)
    json_data = _load_json(DATA_DIR / "mc_questions.json")
    key = f"{subset}_{split}"
    items: List[MCQAItem] = []
    for concept_name, concept_data in json_data.items():
        if concept is not None and concept_name != concept:
            continue
        for obj in concept_data.get(key, []):
            q = obj.get("q", "").strip()
            a = obj.get("correct_answer", "").strip()
            options = obj.get("options", []) or []
            if q and a and len(options) >= 4:
                items.append(MCQAItem(concept=concept_name, subset=subset, split=split,
                                      question=q, options=list(options), correct_answer=a))
    return items


def load_mmlu_indices(split: str) -> List[int]:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split}")
    return _load_json(DATA_DIR / f"mmlu_{split}_indices.json")


def load_alpaca_indices(split: str) -> List[int]:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split}")
    return _load_json(DATA_DIR / f"alpaca_{split}_indices.json")
