#!/usr/bin/env python3
"""Prepare WMDP bio/cyber assets in EMBER's current data format.

This script updates:
  1) data/coherency_prompts.json
  2) data/concept_sentences.json
  3) data/mc_questions.json
  4) data/open_questions.json
  5) data/relearn_paragraphs.json

MC question layout matches existing EMBER concepts (50 per split):
  - QA_train / QA_test: WMDP forget-target MCQ (bio_mcq.json / cyber_mcq.json)
  - SimdomQA_train / SimdomQA_test: general academic MCQ in a related domain
    (college + high-school bio / computer-science MCQ files)

Usage:
  python3 scripts/prepare_wmdp_concepts_for_ember.py --dry-run
  python3 scripts/prepare_wmdp_concepts_for_ember.py --apply --backup
"""
from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

COHER_PATH = DATA_DIR / "coherency_prompts.json"
CONCEPT_SENT_PATH = DATA_DIR / "concept_sentences.json"
MC_QA_PATH = DATA_DIR / "mc_questions.json"
OPEN_QA_PATH = DATA_DIR / "open_questions.json"
RELEARN_PATH = DATA_DIR / "relearn_paragraphs.json"

CRISP_DATA_PATH = ROOT / "external" / "CRISP" / "crisp" / "data.py"
WMDP_DIR = DATA_DIR / "wmdp"

# Existing concepts in data/mc_questions.json use 50 questions per split.
DEFAULT_MC_SPLIT_SIZE = 50


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[backup] {path} -> {backup_path}")


def _extract_list_variable(py_file: Path, var_name: str) -> List[str]:
    """Extract a top-level Python list literal by variable name."""
    module = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, list):
                        raise TypeError(f"{var_name} must be a list, got {type(value)}")
                    return [str(x) for x in value]
    raise KeyError(f"Variable {var_name!r} not found in {py_file}")


def _load_forget_texts(jsonl_path: Path) -> List[str]:
    texts: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            txt = str(obj.get("text", "")).strip()
            if txt:
                texts.append(txt)
    return texts


def _select_subset(texts: Sequence[str], n: int, seed: int) -> List[str]:
    if n <= 0 or n >= len(texts):
        return list(texts)
    rnd = random.Random(seed)
    idx = list(range(len(texts)))
    rnd.shuffle(idx)
    keep = sorted(idx[:n])
    return [texts[i] for i in keep]


def _select_disjoint_subset(
        texts: Sequence[str],
        exclude: Sequence[str],
        n: int,
        seed: int,
) -> List[str]:
    """Sample up to ``n`` texts not present in ``exclude``."""
    if n <= 0:
        return []
    excluded = set(exclude)
    pool = [t for t in texts if t not in excluded]
    if len(pool) < n:
        raise ValueError(
            f"need {n} relearning paragraphs disjoint from concept sentences, "
            f"but only {len(pool)} remain ({len(texts)} total, "
            f"{len(excluded)} excluded)"
        )
    return _select_subset(pool, n, seed)


def _build_mc_records(wmdp_mcq: Dict[str, Any]) -> List[Dict[str, Any]]:
    questions = wmdp_mcq.get("questions", [])
    choices = wmdp_mcq.get("choices", [])
    answers = wmdp_mcq.get("answers", [])

    if not (len(questions) == len(choices) == len(answers)):
        raise ValueError(
            "WMDP MCQ arrays must have equal lengths "
            f"(got questions={len(questions)}, choices={len(choices)}, answers={len(answers)})"
        )

    out: List[Dict[str, Any]] = []
    for q, opts, ans_idx in zip(questions, choices, answers):
        if not isinstance(opts, list) or len(opts) < 2:
            continue
        if not isinstance(ans_idx, int) or ans_idx < 0 or ans_idx >= len(opts):
            continue
        qs = str(q).strip()
        options = [str(o).strip() for o in opts]
        if not qs or any(not o for o in options):
            continue
        out.append(
            {
                "q": qs,
                "correct_answer": options[ans_idx],
                "options": options,
            }
        )
    if not out:
        raise ValueError("No valid MCQ records were built from WMDP file.")
    return out


def _build_mc_records_from_files(*paths: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        records.extend(_build_mc_records(_read_json(path)))
    return records


def _sample_train_test_splits(
        records: Sequence[Dict[str, Any]],
        *,
        train_n: int,
        test_n: int,
        seed: int,
        label: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Sample disjoint train/test lists of fixed sizes."""
    need = train_n + test_n
    if len(records) < need:
        raise ValueError(
            f"{label}: need at least {need} MCQ records "
            f"(train={train_n}, test={test_n}), got {len(records)}"
        )
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train = shuffled[:train_n]
    test = shuffled[train_n:need]
    return train, test


def _build_concept_mc_payload(
        *,
        qa_paths: Sequence[Path],
        simdiv_paths: Sequence[Path],
        split_size: int,
        qa_seed: int,
        simdiv_seed: int,
        label: str,
) -> Dict[str, List[Dict[str, Any]]]:
    qa_records = _build_mc_records_from_files(*qa_paths)
    simdiv_records = _build_mc_records_from_files(*simdiv_paths)

    qa_train, qa_test = _sample_train_test_splits(
        qa_records,
        train_n=split_size,
        test_n=split_size,
        seed=qa_seed,
        label=f"{label} QA",
    )
    simdiv_train, simdiv_test = _sample_train_test_splits(
        simdiv_records,
        train_n=split_size,
        test_n=split_size,
        seed=simdiv_seed,
        label=f"{label} SimdomQA",
    )
    return {
        "QA_train": qa_train,
        "QA_test": qa_test,
        "SimdomQA_train": simdiv_train,
        "SimdomQA_test": simdiv_test,
    }


def _mc_to_open_payload(mc_payload: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, str]]]:
    """Convert MC payload to open-QA payload using ``correct_answer`` as free answer."""
    out: Dict[str, List[Dict[str, str]]] = {}
    for split_name in ("QA_train", "QA_test", "SimdomQA_train", "SimdomQA_test"):
        rows = mc_payload.get(split_name, [])
        out_rows: List[Dict[str, str]] = []
        for row in rows:
            q = str(row.get("q", "")).strip()
            a = str(row.get("correct_answer", "")).strip()
            if q and a:
                out_rows.append({"q": q, "a": a})
        out[split_name] = out_rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes to disk.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only.")
    parser.add_argument("--backup", action="store_true", help="Write .bak copies before updating.")

    parser.add_argument("--bio-concept", default="wmdp-bio")
    parser.add_argument("--cyber-concept", default="wmdp-cyber")
    parser.add_argument("--bio-sentences", type=int, default=300)
    parser.add_argument("--cyber-sentences", type=int, default=300)
    parser.add_argument("--bio-relearn-paragraphs", type=int, default=100,
                        help="Relearning paragraphs for wmdp-bio (disjoint from concept sentences).")
    parser.add_argument("--cyber-relearn-paragraphs", type=int, default=100,
                        help="Relearning paragraphs for wmdp-cyber (disjoint from concept sentences).")
    parser.add_argument("--mc-split-size", type=int, default=DEFAULT_MC_SPLIT_SIZE,
                        help="Questions per MC split (default 50, matching other concepts).")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.apply == args.dry_run:
        raise SystemExit("Choose exactly one of --apply or --dry-run.")
    if args.mc_split_size <= 0:
        raise SystemExit("--mc-split-size must be positive.")
    if args.bio_relearn_paragraphs < 0 or args.cyber_relearn_paragraphs < 0:
        raise SystemExit("--bio-relearn-paragraphs and --cyber-relearn-paragraphs must be >= 0.")

    split_size = args.mc_split_size

    # 1) Coherency prompts from external/CRISP/crisp/data.py
    bio_prompts = _extract_list_variable(CRISP_DATA_PATH, "wmdp_bio_coherency_prompts")
    cyber_prompts = _extract_list_variable(CRISP_DATA_PATH, "wmdp_cyber_coherency_prompts")
    coher = _read_json(COHER_PATH)
    coher[args.bio_concept] = bio_prompts
    coher[args.cyber_concept] = cyber_prompts

    # 2) Concept sentences from cleaned WMDP forget corpora
    bio_forget = _load_forget_texts(WMDP_DIR / "bio" / "bio_forget_dataset_cleaned.jsonl")
    cyber_forget = _load_forget_texts(WMDP_DIR / "cyber" / "cyber_forget_dataset_cleaned.jsonl")
    bio_subset = _select_subset(bio_forget, args.bio_sentences, args.seed)
    cyber_subset = _select_subset(cyber_forget, args.cyber_sentences, args.seed)

    concept_payload = _read_json(CONCEPT_SENT_PATH)
    concept_by_name: Dict[str, Dict[str, Any]] = {}
    for row in concept_payload:
        if isinstance(row, dict) and "concept" in row:
            concept_by_name[str(row["concept"])] = row

    concept_by_name[args.bio_concept] = {"concept": args.bio_concept, "sentences": bio_subset}
    concept_by_name[args.cyber_concept] = {"concept": args.cyber_concept, "sentences": cyber_subset}
    concept_out = sorted(concept_by_name.values(), key=lambda x: str(x["concept"]).lower())

    # 3) MC questions: WMDP forget MCQ for QA; academic MCQ for SimdivQA
    bio_mc = _build_concept_mc_payload(
        qa_paths=[WMDP_DIR / "bio" / "bio_mcq.json"],
        simdiv_paths=[
            WMDP_DIR / "bio" / "college_bio_mcq.json",
            WMDP_DIR / "bio" / "high_school_bio_mcq.json",
        ],
        split_size=split_size,
        qa_seed=args.seed,
        simdiv_seed=args.seed + 1,
        label=args.bio_concept,
    )
    cyber_mc = _build_concept_mc_payload(
        qa_paths=[WMDP_DIR / "cyber" / "cyber_mcq.json"],
        simdiv_paths=[
            WMDP_DIR / "cyber" / "college_computer_science_mcq.json",
            WMDP_DIR / "cyber" / "high_school_computer_science_mcq.json",
        ],
        split_size=split_size,
        qa_seed=args.seed + 2,
        simdiv_seed=args.seed + 3,
        label=args.cyber_concept,
    )

    mc_payload = _read_json(MC_QA_PATH)
    mc_payload[args.bio_concept] = bio_mc
    mc_payload[args.cyber_concept] = cyber_mc

    # 4) Open questions mirrored from MC questions (same q, a=correct_answer)
    open_payload = _read_json(OPEN_QA_PATH)
    open_payload[args.bio_concept] = _mc_to_open_payload(bio_mc)
    open_payload[args.cyber_concept] = _mc_to_open_payload(cyber_mc)

    # 5) Relearning paragraphs from forget corpora (disjoint from concept sentences)
    bio_relearn = _select_disjoint_subset(
        bio_forget,
        bio_subset,
        args.bio_relearn_paragraphs,
        args.seed + 10,
    )
    cyber_relearn = _select_disjoint_subset(
        cyber_forget,
        cyber_subset,
        args.cyber_relearn_paragraphs,
        args.seed + 11,
    )

    relearn_payload = _read_json(RELEARN_PATH)
    relearn_payload[args.bio_concept] = {"RelearnParagraphs": bio_relearn}
    relearn_payload[args.cyber_concept] = {"RelearnParagraphs": cyber_relearn}

    def _mc_summary(name: str, payload: Dict[str, List[Dict[str, Any]]]) -> str:
        return (f"{name}: QA_train={len(payload['QA_train'])}, "
                f"QA_test={len(payload['QA_test'])}, "
                f"SimdomQA_train={len(payload['SimdomQA_train'])}, "
                f"SimdomQA_test={len(payload['SimdomQA_test'])}")

    print("[summary]")
    print(f"  coherency: {args.bio_concept}={len(bio_prompts)} prompts, "
          f"{args.cyber_concept}={len(cyber_prompts)} prompts")
    print(f"  concept_sentences: {args.bio_concept}={len(bio_subset)} sentences, "
          f"{args.cyber_concept}={len(cyber_subset)} sentences")
    print(f"  mc_questions ({split_size} per split):")
    print(f"    {_mc_summary(args.bio_concept, bio_mc)}")
    print(f"    {_mc_summary(args.cyber_concept, cyber_mc)}")
    print("  open_questions (mirrored from mc_questions):")
    print(f"    {_mc_summary(args.bio_concept, open_payload[args.bio_concept])}")
    print(f"    {_mc_summary(args.cyber_concept, open_payload[args.cyber_concept])}")
    print(f"  relearn_paragraphs: {args.bio_concept}={len(bio_relearn)}, "
          f"{args.cyber_concept}={len(cyber_relearn)}")

    if args.dry_run:
        print("[dry-run] no files written.")
        return

    if args.backup:
        _backup(COHER_PATH)
        _backup(CONCEPT_SENT_PATH)
        _backup(MC_QA_PATH)
        _backup(OPEN_QA_PATH)
        _backup(RELEARN_PATH)

    _write_json(COHER_PATH, coher)
    _write_json(CONCEPT_SENT_PATH, concept_out)
    _write_json(MC_QA_PATH, mc_payload)
    _write_json(OPEN_QA_PATH, open_payload)
    _write_json(RELEARN_PATH, relearn_payload)
    print("[done] wrote updated data files.")


if __name__ == "__main__":
    main()
