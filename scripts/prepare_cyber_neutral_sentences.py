#!/usr/bin/env python3
"""Build a cyber-specific neutral/retain corpus from WMDP cyber retain JSONL.

Reads ``data/wmdp/cyber/cyber_retain_dataset_cleaned.jsonl`` and writes a JSON
list of ``{"sentence": ...}`` objects, with the same number of entries as
``data/neutral_sentences.json`` by default (300).

Usage:
  python3 scripts/prepare_cyber_neutral_sentences.py --dry-run
  python3 scripts/prepare_cyber_neutral_sentences.py --apply --backup
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DEFAULT_INPUT = DATA_DIR / "wmdp" / "cyber" / "cyber_retain_dataset_cleaned.jsonl"
DEFAULT_REFERENCE = DATA_DIR / "neutral_sentences.json"
DEFAULT_OUTPUT = DATA_DIR / "cyber_neutral_sentences.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def _backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[backup] {path} -> {backup_path}")


def _load_retain_texts(jsonl_path: Path) -> List[str]:
    texts: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            txt = str(obj.get("text", "")).strip()
            if not txt:
                raise ValueError(f"{jsonl_path}:{line_no}: missing non-empty 'text'")
            texts.append(txt)
    if not texts:
        raise ValueError(f"No retain texts found in {jsonl_path}")
    return texts


def _select_subset(texts: Sequence[str], n: int, seed: int, *, label: str = "input") -> List[str]:
    if n <= 0:
        raise ValueError("--count must be positive")
    if n > len(texts):
        raise ValueError(f"need {n} retain sentences, but {label} only has {len(texts)}")
    rnd = random.Random(seed)
    idx = list(range(len(texts)))
    rnd.shuffle(idx)
    keep = sorted(idx[:n])
    return [texts[i] for i in keep]


def _build_records(texts: Sequence[str]) -> List[Dict[str, str]]:
    return [{"sentence": sentence} for sentence in texts]


def _reference_count(reference_path: Path) -> int:
    payload = _read_json(reference_path)
    if not isinstance(payload, list):
        raise TypeError(f"{reference_path} must be a JSON list")
    return len(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write output file.")
    parser.add_argument("--dry-run", action="store_true", help="Preview only.")
    parser.add_argument("--backup", action="store_true",
                        help="Write .bak copy before overwriting output.")

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Cyber retain JSONL (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                        help="Use this file's length as default --count "
                             f"(default: {DEFAULT_REFERENCE})")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of retain sentences (default: len(reference))")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsampling (default: 42).")

    args = parser.parse_args()
    if args.apply == args.dry_run:
        raise SystemExit("Choose exactly one of --apply or --dry-run.")

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")
    if args.count is None:
        if not args.reference.is_file():
            raise SystemExit(
                f"Reference not found: {args.reference}. Pass --count explicitly."
            )
        args.count = _reference_count(args.reference)

    retain_texts = _load_retain_texts(args.input)
    subset = _select_subset(retain_texts, args.count, args.seed, label=str(args.input))
    records = _build_records(subset)

    lengths = [len(r["sentence"]) for r in records]
    print("[summary]")
    print(f"  input:      {args.input} ({len(retain_texts)} texts)")
    print(f"  reference:  {args.reference} -> count={args.count}")
    print(f"  output:     {args.output}")
    print(f"  selected:   {len(records)} sentences (seed={args.seed})")
    print(f"  length:     min={min(lengths)}, max={max(lengths)}, "
          f"avg={sum(lengths) / len(lengths):.1f}")

    if args.dry_run:
        print("[dry-run] sample record:")
        print(json.dumps(records[0], indent=2, ensure_ascii=False))
        print("[dry-run] no files written.")
        return

    if args.output.exists() and args.backup:
        _backup(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, records)
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
