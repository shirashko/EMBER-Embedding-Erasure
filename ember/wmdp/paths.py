"""Filesystem paths for WMDP corpora."""
from __future__ import annotations

from pathlib import Path

from ember.erasure.io import ROOT_DIR

DEFAULT_DATA_ROOT = ROOT_DIR / "data" / "wmdp"
RAW_SUBDIR = "raw"


def resolve_data_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    return root if root.is_absolute() else ROOT_DIR / root


def cleaned_jsonl_path(data_root: Path, domain: str, split: str) -> Path:
    """Path to a prepared ``{domain}_{split}_dataset_cleaned.jsonl`` file."""
    if split not in ("forget", "retain"):
        raise ValueError(f"split must be 'forget' or 'retain', got {split!r}")
    return data_root / domain / f"{domain}_{split}_dataset_cleaned.jsonl"


def manifest_path(data_root: Path, domain: str) -> Path:
    return data_root / domain / "prepared_manifest.json"


def raw_jsonl_path(data_root: Path, domain: str, split: str) -> Path:
    """Optional local raw input before preprocessing."""
    return data_root / RAW_SUBDIR / domain / f"{domain}_{split}_dataset.jsonl"


# CRISP eval.py filenames under ``{data_root}/{domain}/``
WMDP_PRIMARY_MCQ: dict[str, str] = {
    "bio": "bio_mcq.json",
    "cyber": "cyber_mcq.json",
}
WMDP_AUXILIARY_MCQ: dict[str, tuple[str, ...]] = {
    "bio": ("high_school_bio_mcq.json", "college_bio_mcq.json"),
    "cyber": (
        "high_school_computer_science_mcq.json",
        "college_computer_science_mcq.json",
    ),
}


def mcq_json_path(data_root: Path, domain: str, filename: str) -> Path:
    return data_root / domain / filename
