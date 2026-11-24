"""Per-concept score tables written by the frozen-SNMF scorer."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def layer_csv_path(concept_dir: Path, layer: int) -> Path:
    return concept_dir / "layers" / f"layer_{layer:02d}.csv"


def write_layer_scores(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _topk_per_layer(frame: pd.DataFrame, column: str, top_k: int) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby("layer", sort=True):
        parts.append(group.sort_values(column, ascending=False).head(top_k))
    if not parts:
        return frame.iloc[0:0]
    return pd.concat(parts, ignore_index=True).sort_values(
        ["layer", column], ascending=[True, False],
    ).reset_index(drop=True)


def merge_concept_tables(concept_dir: Path, top_k: int) -> None:
    """Rebuild the full score table and the two per-layer selections from layer CSVs."""
    layer_paths = sorted((concept_dir / "layers").glob("layer_*.csv"))
    if not layer_paths:
        return
    scores = pd.concat([pd.read_csv(path) for path in layer_paths], ignore_index=True)
    scores = scores.sort_values(["layer", "feature"]).reset_index(drop=True)
    scores.to_csv(concept_dir / "scores.csv", index=False)
    _topk_per_layer(scores, "s_cd", top_k).to_csv(concept_dir / "top_by_concept_detection.csv", index=False)
    _topk_per_layer(scores, "promotion_score", top_k).to_csv(concept_dir / "top_promoted.csv", index=False)
