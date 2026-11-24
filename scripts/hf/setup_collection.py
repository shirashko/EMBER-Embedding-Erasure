#!/usr/bin/env python3
"""Create an overview dataset for a Hugging Face collection and update collection metadata."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file

from upload import DEFAULT_COLLECTION_SLUG, canonical_collection_slug, resolve_collection_slug

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HF_DIR = Path(__file__).resolve().parent
DEFAULT_ASSET = HF_DIR / "assets" / "unlearned_models_collection_asset.jpeg"
DEFAULT_DATASET_REPO = "unlearned-models-overview"
IMAGE_FILENAME = "unlearned_models_collection_asset.jpeg"

OVERVIEW_README = """---
license: mit
tags:
- unlearning
- overview
---

# Unlearned Models Overview

Overview of the unlearned model checkpoints in the
[What-Did-You-Forget-Unlearned-Models](https://huggingface.co/collections/shirasko/what-did-you-forget-unlearned-models-6a78ad07bbedf556ad6555cf)
collection.

![Unlearning methods and base models]({image_filename})

## Methods

- **PISCES** — Precise In-parameter Suppression for Concept Erasure
- **RMU** — Representation Misdirection for Unlearning
- **CRISP** — Concept Removal via Interpretable Sparse Projections
- **SNMF** — Semi-Nonnegative Matrix Factorization

## Base models

- `google/gemma-2-2b-it`
- `meta-llama/Llama-3.1-8B-Instruct`

## Collection

This dataset is a visual overview card for the collection. The model checkpoints
themselves live in the linked collection as individual model repositories.
"""

DEFAULT_COLLECTION_DESCRIPTION = (
    "67 unlearned checkpoints (PISCES, RMU, CRISP, SNMF) on Gemma-2-2B-IT and "
    "Llama-3.1-8B-Instruct. See the overview dataset for details."
)


def dataset_repo_id(username: str, dataset_name: str) -> str:
    return f"{username}/{dataset_name}"


def ensure_overview_dataset(
    api: HfApi,
    username: str,
    *,
    dataset_name: str,
    asset_path: Path,
    private: bool = False,
    dry_run: bool = False,
) -> str:
    repo_id = dataset_repo_id(username, dataset_name)
    readme_text = OVERVIEW_README.format(image_filename=IMAGE_FILENAME)

    if dry_run:
        logger.info(f"[DRY RUN] Would create dataset repo: {repo_id}")
        logger.info(f"[DRY RUN] Would upload {asset_path.name} and README.md")
        return repo_id

    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    upload_file(
        path_or_fileobj=str(asset_path),
        path_in_repo=IMAGE_FILENAME,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add collection overview infographic",
    )
    upload_file(
        path_or_fileobj=readme_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add collection overview README",
    )
    logger.info(f"Published overview dataset: {repo_id}")
    return repo_id


def add_overview_to_collection(
    api: HfApi,
    collection_slug: str,
    dataset_repo_id_value: str,
    *,
    dry_run: bool = False,
) -> None:
    if dry_run:
        logger.info(
            f"[DRY RUN] Would add {dataset_repo_id_value} to {collection_slug} at position 0"
        )
        return

    collection_slug = canonical_collection_slug(api, collection_slug)

    api.add_collection_item(
        collection_slug=collection_slug,
        item_id=dataset_repo_id_value,
        item_type="dataset",
        exists_ok=True,
    )

    collection = api.get_collection(collection_slug)
    overview_item = next(
        (item for item in collection.items if item.item_id == dataset_repo_id_value),
        None,
    )
    if overview_item is None:
        raise RuntimeError(f"Overview dataset not found in collection after add: {dataset_repo_id_value}")

    if overview_item.position != 0:
        api.update_collection_item(
            collection_slug=collection_slug,
            item_object_id=overview_item.item_object_id,
            position=0,
        )
        logger.info(f"Moved overview dataset to top of collection: {dataset_repo_id_value}")
    else:
        logger.info(f"Overview dataset already at top of collection: {dataset_repo_id_value}")


def update_collection_description(
    api: HfApi,
    collection_slug: str,
    description: str,
    *,
    dry_run: bool = False,
) -> None:
    if dry_run:
        logger.info(f"[DRY RUN] Would update collection description for {collection_slug}")
        return

    collection_slug = canonical_collection_slug(api, collection_slug)
    api.update_collection_metadata(collection_slug=collection_slug, description=description)
    logger.info(f"Updated collection description: {collection_slug}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a collection overview dataset and update collection metadata."
    )
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_REPO)
    parser.add_argument("--asset-path", type=Path, default=DEFAULT_ASSET)
    parser.add_argument(
        "--collection-slug",
        type=str,
        default=None,
        help=f"HF collection slug (default: {DEFAULT_COLLECTION_SLUG})",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=DEFAULT_COLLECTION_DESCRIPTION,
        help="Plain-text collection description",
    )
    parser.add_argument("--private", action="store_true", help="Create the overview dataset as private")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    collection_slug = resolve_collection_slug(args.collection_slug, disabled=False)
    if not collection_slug:
        logger.error("No collection slug configured.")
        sys.exit(1)

    if not args.asset_path.exists():
        logger.error(f"Asset not found: {args.asset_path}")
        sys.exit(1)

    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN environment variable not set. Ensure you are logged in via `huggingface-cli login`.")

    api = HfApi()
    repo_id = ensure_overview_dataset(
        api,
        args.username,
        dataset_name=args.dataset_name,
        asset_path=args.asset_path,
        private=args.private,
        dry_run=args.dry_run,
    )
    add_overview_to_collection(
        api,
        collection_slug,
        repo_id,
        dry_run=args.dry_run,
    )
    update_collection_description(
        api,
        collection_slug,
        args.description,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
