"""Gemini membership judge for the frozen-SNMF shortlist.

Uses the description and membership prompts from ``ember.interpret_features``
and writes the same columns as ``potential_features.csv``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

POTENTIAL_COLUMNS = [
    "model",
    "concept",
    "rank",
    "layer",
    "feature",
    "metric_score",
    "source",
    "n_tokens_used",
    "tokens_provided",
    "description",
    "is_member",
    "confidence",
]

# source column in the score table -> column that holds the tokens Gemini should see,
# and the numeric column copied into metric_score.
JUDGE_SOURCES = (
    ("top_by_concept_detection.csv", "activating_tokens", "activation_ratio", "from_activation.csv"),
    ("top_promoted.csv", "projection_top_tokens", "promotion_score", "from_projection.csv"),
)


def _members(frame: pd.DataFrame, confidence_thresh: float) -> pd.DataFrame:
    if frame.empty:
        return frame
    is_member = (frame["is_member"] == True) | (  # noqa: E712
        frame["is_member"].astype(str).str.strip().str.lower() == "true"
    )
    confidence = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0)
    kept = frame[is_member & (confidence >= confidence_thresh)].copy()
    if kept.empty:
        return kept
    return kept.drop_duplicates(subset=["layer", "feature", "source"], keep="first").reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, path: Path, columns: List[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SystemExit(
            f"{path} is missing {missing}. Re-run with --overwrite so each feature stores "
            "activating_tokens and activation_ratio."
        )


def judge_shortlist(
    concept_dir: Path,
    *,
    gemini_model: str,
    confidence_thresh: float,
    judge_top_tokens: int,
    max_workers: int,
    resume: bool,
) -> Path:
    """Judge the per-layer shortlists and write ``potential_features.csv``."""
    from dotenv import load_dotenv

    from ember.interpret_features import SimpleGeminiClient, interpret_source

    load_dotenv()
    detection_path = concept_dir / "top_by_concept_detection.csv"
    promoted_path = concept_dir / "top_promoted.csv"
    if not detection_path.is_file() or not promoted_path.is_file():
        raise SystemExit(f"Missing shortlist CSVs under {concept_dir}. Score the dictionary first.")

    client = SimpleGeminiClient(model_name=gemini_model)
    judged: List[pd.DataFrame] = []
    for table_name, token_col, score_col, out_name in JUDGE_SOURCES:
        table_path = concept_dir / table_name
        frame = pd.read_csv(table_path)
        if frame.empty:
            print(f"[skip] {table_path.name} has no rows.")
            continue
        _require_columns(frame, table_path, ["layer", "feature", "rank", "model", token_col, score_col])
        concept_name = str(frame["concept"].iloc[0])
        model_name = str(frame["model"].iloc[0])
        rank = int(frame["rank"].iloc[0])
        print(f"Judging {len(frame)} {token_col} rows for {concept_name}")
        judged.append(interpret_source(
            client,
            frame,
            concept_name,
            model_name,
            rank,
            token_col,
            concept_dir / out_name,
            score_col,
            judge_top_tokens,
            save_every=50,
            resume=resume,
            max_workers=max_workers,
        ))

    members = [_members(frame, confidence_thresh) for frame in judged if frame is not None and not frame.empty]
    potential = pd.concat(members, ignore_index=True) if members else pd.DataFrame(columns=POTENTIAL_COLUMNS)
    if not potential.empty:
        potential = potential.drop_duplicates(subset=["layer", "feature", "source"], keep="first")
        potential = potential.sort_values(["layer", "feature", "source"]).reset_index(drop=True)
        potential = potential.reindex(columns=POTENTIAL_COLUMNS)
    out_path = concept_dir / "potential_features.csv"
    potential.to_csv(out_path, index=False)
    print(f"Wrote {len(potential)} member rows to {out_path}")
    return out_path
