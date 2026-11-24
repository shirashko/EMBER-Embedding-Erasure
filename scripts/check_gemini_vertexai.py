#!/usr/bin/env python3
"""Quick Gemini Vertex AI smoke test used by erasure evaluation.

Run this before launching long jobs:
    python scripts/check_gemini_vertexai.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ember.evals.gemini import DEFAULT_MODEL_NAME, GeminiEvaluator


def _require_api_key() -> None:
    api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GEMINI_API_TOKEN")
    )
    if not api_key:
        raise RuntimeError(
            "No API key found. Set GOOGLE_API_KEY (or GEMINI_API_KEY / GEMINI_API_TOKEN)."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Gemini model id for the smoke test.",
    )
    args = parser.parse_args()

    _require_api_key()
    evaluator = GeminiEvaluator(model_name=args.model)

    print(f"[smoke] using model={args.model!r} with vertexai=True")
    ok = evaluator.judge_open_qa(
        question="What is 2 + 2?",
        answer="4",
        attempted="4",
    )
    print(f"[smoke] open_qa_judge={ok}")
    if not ok:
        print("[smoke] unexpected judge result (expected True)", file=sys.stderr)
        return 2

    instruct_score = evaluator.score_alpaca_instruct(
        instruction="Write a short greeting.",
        completion="Hello there!",
    )
    fluency_score = evaluator.score_alpaca_fluency("Hello there!")
    print(f"[smoke] alpaca_instruct={instruct_score} alpaca_fluency={fluency_score}")
    print("[smoke] Gemini Vertex AI check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
