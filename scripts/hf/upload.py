import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
import logging

from readme import write_readme
from naming import repo_id_for, repo_name, short_model_slug

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IGNORE_PATTERNS = ["*.tmp", "*.log", "__pycache__/*", ".DS_Store", "checkpoint-*/"]
WEIGHT_FILE_MARKERS = (
    "model.safetensors",
    "pytorch_model.bin",
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def is_adapter_checkpoint(concept_dir: Path) -> bool:
    return (concept_dir / "adapter_config.json").exists() or (
        concept_dir / "adapter_model.safetensors"
    ).exists()


def iter_checkpoints(
    root_dir: str,
    methods: list[str] | None = None,
    base_models: list[str] | None = None,
):
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory {root_dir} does not exist.")

    method_filter = {m.lower() for m in methods} if methods else None
    model_filter = set(base_models) if base_models else None

    for method_dir in sorted(root_path.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        if method_filter and method.lower() not in method_filter:
            continue

        for model_dir in sorted(method_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            base_model_raw = model_dir.name
            if model_filter and base_model_raw not in model_filter:
                continue
            base_model_clean = short_model_slug(base_model_raw)

            for concept_dir in sorted(model_dir.iterdir()):
                if not concept_dir.is_dir():
                    continue

                concept = concept_dir.name
                checkpoint_repo_name = repo_name(method, base_model_raw, concept)
                yield {
                    "method": method,
                    "base_model_raw": base_model_raw,
                    "base_model_clean": base_model_clean,
                    "concept": concept,
                    "concept_dir": concept_dir,
                    "repo_name": checkpoint_repo_name,
                }


def repo_has_complete_upload(api: HfApi, repo_id: str) -> bool:
    if not api.repo_exists(repo_id=repo_id, repo_type="model"):
        return False

    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    has_config = "config.json" in files or "adapter_config.json" in files
    has_weights = any(
        name.endswith(".safetensors")
        or name.endswith(".bin")
        or name == "model.safetensors.index.json"
        for name in files
    )
    return has_config and has_weights


DEFAULT_COLLECTION_SLUG = "shirasko/what-did-you-forget-unlearned-models"


def resolve_collection_slug(explicit: str | None = None, disabled: bool = False) -> str | None:
    if disabled:
        return None
    if explicit:
        return explicit
    return os.environ.get("HF_COLLECTION_SLUG", DEFAULT_COLLECTION_SLUG) or None


def canonical_collection_slug(api: HfApi, collection_slug: str) -> str:
    """Return the full collection slug (includes Hub id suffix when present)."""
    return api.get_collection(collection_slug).slug


def add_repo_to_collection(
    api: HfApi,
    repo_id: str,
    collection_slug: str,
    *,
    dry_run: bool = False,
) -> bool:
    if dry_run:
        logger.info(f"[DRY RUN] Would add {repo_id} to collection {collection_slug}")
        return True

    api.add_collection_item(
        collection_slug=collection_slug,
        item_id=repo_id,
        item_type="model",
        exists_ok=True,
    )
    logger.info(f"Added to collection: {repo_id}")
    return True


def sync_checkpoints_to_collection(
    root_dir: str,
    username: str,
    collection_slug: str,
    dry_run: bool = False,
    only_complete: bool = True,
) -> tuple[int, int, int]:
    api = HfApi()
    added = 0
    skipped = 0
    failed = 0

    for checkpoint in iter_checkpoints(root_dir):
        repo_id = repo_id_for(username, checkpoint["repo_name"])

        if only_complete and not repo_has_complete_upload(api, repo_id):
            logger.info(f"[skip] Repo missing or incomplete on Hub: {repo_id}")
            skipped += 1
            continue

        try:
            add_repo_to_collection(api, repo_id, collection_slug, dry_run=dry_run)
            added += 1
        except Exception as exc:
            failed += 1
            logger.error(f"Failed to add {repo_id} to collection: {exc}")

    return added, skipped, failed


def make_existing_public(api: HfApi, username: str, root_dir: str, dry_run: bool = False) -> tuple[int, int]:
    made_public = 0
    skipped = 0

    for checkpoint in iter_checkpoints(root_dir):
        repo_id = repo_id_for(username, checkpoint["repo_name"])
        if not api.repo_exists(repo_id=repo_id, repo_type="model"):
            logger.info(f"[skip] Repo does not exist yet: {repo_id}")
            skipped += 1
            continue

        info = api.repo_info(repo_id=repo_id, repo_type="model")
        if not info.private:
            logger.info(f"[skip] Already public: {repo_id}")
            skipped += 1
            continue

        logger.info(f"Making public: {repo_id}")
        if dry_run:
            made_public += 1
            continue

        if hasattr(api, "update_repo_settings"):
            api.update_repo_settings(repo_id=repo_id, private=False, repo_type="model")
        else:
            api.update_repo_visibility(repo_id=repo_id, private=False, repo_type="model")
        made_public += 1
        logger.info(f"Made public: {repo_id}")

    return made_public, skipped


def upload_checkpoints(
    root_dir: str,
    username: str,
    private: bool = True,
    dry_run: bool = False,
    skip_complete: bool = False,
    collection_slug: str | None = None,
    methods: list[str] | None = None,
    base_models: list[str] | None = None,
) -> tuple[int, int, int]:
    api = HfApi()
    uploaded = 0
    skipped = 0
    failed = 0

    for checkpoint in iter_checkpoints(root_dir, methods=methods, base_models=base_models):
        concept_dir = checkpoint["concept_dir"]
        repo_id = repo_id_for(username, checkpoint["repo_name"])
        method = checkpoint["method"]
        base_model_raw = checkpoint["base_model_raw"]
        base_model_clean = checkpoint["base_model_clean"]
        concept = checkpoint["concept"]

        logger.info(f"Processing: {repo_id} (from {concept_dir})")

        if skip_complete and repo_has_complete_upload(api, repo_id):
            logger.info(f"[skip] Complete upload already exists: {repo_id}")
            if collection_slug:
                try:
                    add_repo_to_collection(api, repo_id, collection_slug, dry_run=dry_run)
                except Exception as e:
                    failed += 1
                    logger.error(f"Failed to add {repo_id} to collection: {e}")
            skipped += 1
            continue

        if dry_run:
            logger.info(f"[DRY RUN] Would upload {concept_dir} to {repo_id} (private={private})")
            if collection_slug:
                add_repo_to_collection(api, repo_id, collection_slug, dry_run=True)
            uploaded += 1
            continue

        try:
            create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
            write_readme(
                concept_dir,
                method=method,
                base_model_raw=base_model_raw,
                base_model_clean=base_model_clean,
                concept=concept,
            )
            logger.info(f"Generated README.md for {repo_id}")

            logger.info(f"Uploading {concept_dir} to {repo_id} (private={private})...")
            upload_folder(
                folder_path=str(concept_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {method} unlearned checkpoint for {concept} on {base_model_clean}",
                ignore_patterns=IGNORE_PATTERNS,
            )
            uploaded += 1
            logger.info(f"Successfully uploaded {repo_id}")
            if collection_slug:
                add_repo_to_collection(api, repo_id, collection_slug)
        except Exception as e:
            failed += 1
            logger.error(f"Failed to upload {repo_id}: {e}")

    return uploaded, skipped, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload unlearned checkpoints to Hugging Face Hub.")
    parser.add_argument("--root_dir", type=str, default="unlearned_checkpoints")
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--public", action="store_false", dest="private", help="Create/upload repos as public")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="Skip checkpoints whose HF repo already contains model weights",
    )
    parser.add_argument(
        "--make-public-existing",
        action="store_true",
        help="Set all existing checkpoint repos under this username to public",
    )
    parser.add_argument(
        "--collection-slug",
        type=str,
        default=None,
        help=f"HF collection slug to add each repo to (default: {DEFAULT_COLLECTION_SLUG})",
    )
    parser.add_argument(
        "--no-collection",
        action="store_true",
        help="Do not add uploaded repos to a Hugging Face collection",
    )
    parser.add_argument(
        "--sync-collection-only",
        action="store_true",
        help="Only add existing Hub repos to the collection (no uploads)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Upload only these unlearning methods (e.g. rmu snmf)",
    )
    parser.add_argument(
        "--base-models",
        nargs="+",
        default=None,
        help="Upload only these base model directory names (e.g. google_gemma-2-2b-it)",
    )
    args = parser.parse_args()

    collection_slug = resolve_collection_slug(args.collection_slug, disabled=args.no_collection)

    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN environment variable not set. Ensure you are logged in via `huggingface-cli login`.")

    exit_code = 0

    if args.sync_collection_only:
        if not collection_slug:
            logger.error("Collection sync requested but no collection slug is configured.")
            sys.exit(1)
        added, skipped, failed = sync_checkpoints_to_collection(
            args.root_dir,
            args.username,
            collection_slug,
            dry_run=args.dry_run,
        )
        logger.info(f"Collection sync summary: added={added}, skipped={skipped}, failed={failed}")
        sys.exit(1 if failed > 0 else 0)

    if args.make_public_existing:
        api = HfApi()
        try:
            made_public, skipped = make_existing_public(api, args.username, args.root_dir, dry_run=args.dry_run)
            logger.info(f"Make-public summary: changed={made_public}, skipped={skipped}")
        except Exception as e:
            logger.error(f"Failed while making repos public: {e}")
            sys.exit(1)

    uploaded, skipped, failed = upload_checkpoints(
        args.root_dir,
        args.username,
        private=args.private,
        dry_run=args.dry_run,
        skip_complete=args.skip_complete,
        collection_slug=collection_slug,
        methods=args.methods,
        base_models=args.base_models,
    )
    logger.info(f"Upload summary: uploaded={uploaded}, skipped={skipped}, failed={failed}")

    if failed > 0:
        exit_code = 1
    sys.exit(exit_code)
