#!/usr/bin/env python3
"""Download MLP (SNMF) activations and features from ClSu/ember-features."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")

    from huggingface_hub import snapshot_download

    local_dir = REPO_ROOT / "mf_outputs"
    snapshot_download(
        repo_id="ClSu/ember-features",
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=["**/mlp/**"],
    )
    print(f"Downloaded MLP features to {local_dir}")


if __name__ == "__main__":
    main()
