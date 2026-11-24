#!/usr/bin/env python3
"""Regenerate and upload README model cards for checkpoints already on Hugging Face."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from huggingface_hub import HfApi

from readme import write_readme
from naming import repo_id_for
from upload import (
    add_repo_to_collection,
    iter_checkpoints,
    repo_has_complete_upload,
    resolve_collection_slug,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def refresh_readmes(
    root_dir: str,
    username: str,
    dry_run: bool = False,
    only_complete: bool = True,
    collection_slug: str | None = None,
) -> tuple[int, int, int]:
    api = HfApi()
    updated = 0
    skipped = 0
    failed = 0

    for checkpoint in iter_checkpoints(root_dir):
        repo_id = repo_id_for(username, checkpoint["repo_name"])
        concept_dir = checkpoint["concept_dir"]

        if only_complete and not repo_has_complete_upload(api, repo_id):
            logger.info(f"[skip] Repo missing or incomplete on Hub: {repo_id}")
            skipped += 1
            continue

        try:
            readme_path = write_readme(
                concept_dir,
                method=checkpoint["method"],
                base_model_raw=checkpoint["base_model_raw"],
                base_model_clean=checkpoint["base_model_clean"],
                concept=checkpoint["concept"],
            )
            logger.info(f"Generated README for {repo_id}")

            if dry_run:
                updated += 1
                continue

            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Update model card with unlearning evaluation metrics",
            )
            updated += 1
            logger.info(f"Uploaded README to {repo_id}")

            if collection_slug:
                add_repo_to_collection(api, repo_id, collection_slug, dry_run=False)
        except Exception as exc:
            failed += 1
            logger.error(f"Failed to refresh README for {repo_id}: {exc}")

    return updated, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Hugging Face model card READMEs.")
    parser.add_argument("--root_dir", type=str, default="unlearned_checkpoints")
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Also refresh repos that do not yet have complete model weights on the Hub",
    )
    parser.add_argument(
        "--collection-slug",
        type=str,
        default=None,
        help="HF collection slug to add each repo to after README refresh",
    )
    parser.add_argument(
        "--no-collection",
        action="store_true",
        help="Do not add repos to a Hugging Face collection",
    )
    args = parser.parse_args()

    collection_slug = resolve_collection_slug(args.collection_slug, disabled=args.no_collection)

    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN environment variable not set. Ensure you are logged in via `huggingface-cli login`.")

    updated, skipped, failed = refresh_readmes(
        args.root_dir,
        args.username,
        dry_run=args.dry_run,
        only_complete=not args.include_incomplete,
        collection_slug=collection_slug,
    )
    logger.info(f"Refresh summary: updated={updated}, skipped={skipped}, failed={failed}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
