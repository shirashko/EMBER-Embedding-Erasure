#!/usr/bin/env python3
"""Download a Hugging Face model into the shared hub cache.

Puts files in the ``models--org--name/`` layout used by
``HUGGINGFACE_HUB_CACHE`` (see ``.env``), so later ``from_pretrained``
loads from cache without hitting the network.

Does not load weights onto GPU.

Example:
    python scripts/download_hf_model.py
    python scripts/download_hf_model.py --model-id Qwen/Qwen3.5-2B
    python scripts/download_hf_model.py --verify
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                   help="Hugging Face repo id (default: %(default)s)")
    p.add_argument(
        "--hub-cache",
        default=None,
        help="Hub cache directory (models--org--name/). "
             "Defaults to HUGGINGFACE_HUB_CACHE / HF_HUB_CACHE from .env.",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="After download, load the tokenizer with local_files_only.",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    hub_cache = (
        args.hub_cache
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_HUB_CACHE")
    )
    if not hub_cache:
        raise SystemExit(
            "Hub cache is not set. Add HUGGINGFACE_HUB_CACHE to .env "
            "(see .env.example) or pass --hub-cache."
        )
    hub_path = Path(hub_cache)
    hub_path.mkdir(parents=True, exist_ok=True)
    if not os.access(hub_path, os.W_OK):
        raise SystemExit(f"Hub cache is not writable: {hub_path}")

    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_path)
    os.environ["HF_HUB_CACHE"] = str(hub_path)

    from huggingface_hub import hf_hub_download, login, snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    print(f"Downloading {args.model_id} -> {hub_path}")
    snapshot_path = snapshot_download(
        repo_id=args.model_id,
        cache_dir=str(hub_path),
        token=token,
    )
    print(f"Snapshot: {snapshot_path}")

    # Touch config so a later local_files_only load can resolve the repo.
    config_path = hf_hub_download(
        repo_id=args.model_id,
        filename="config.json",
        cache_dir=str(hub_path),
        token=token,
        local_files_only=True,
    )
    print(f"config.json: {config_path}")

    if args.verify:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            args.model_id,
            cache_dir=str(hub_path),
            local_files_only=True,
        )
        print(f"Tokenizer OK ({tok.__class__.__name__}, vocab={len(tok)})")

    print("Done. Unlearning jobs can load this model from HUGGINGFACE_HUB_CACHE.")


if __name__ == "__main__":
    main()
