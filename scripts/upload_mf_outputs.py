#!/usr/bin/env python3
"""Upload ``mf_outputs/`` SNMF feature files to a Hugging Face dataset.

Hub paths:

    <model>/{pickles,csvs,interpretations}/rank<R>/seed<S>/<concept>/mlp/...

Existing Hub files are skipped unless ``--overwrite`` is set. Re-run after each
concept finishes training; only new files are committed.

Model repos (used when ``--repo-name`` / ``--repo-id`` are omitted):

    Qwen_Qwen3.5-2B           -> shirasko/qwen-snmf-features
    Qwen_Qwen2.5-3B-Instruct  -> shirasko/qwen2.5-snmf-features

Example:

    python scripts/upload_mf_outputs.py --username shirasko --public \\
        --models Qwen_Qwen3.5-2B

    python scripts/upload_mf_outputs.py --username shirasko --public \\
        --models Qwen_Qwen2.5-3B-Instruct

    python scripts/upload_mf_outputs.py --username shirasko --dry-run \\
        --models Qwen_Qwen3.5-2B

Download:

    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="shirasko/qwen-snmf-features", repo_type="dataset",
                      local_dir="mf_outputs",
                      allow_patterns=["Qwen_Qwen3.5-2B/**"])
    snapshot_download(repo_id="shirasko/qwen2.5-snmf-features", repo_type="dataset",
                      local_dir="mf_outputs",
                      allow_patterns=["Qwen_Qwen2.5-3B-Instruct/**"])
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "mf_outputs"
KINDS = ("pickles", "csvs", "interpretations")
SKIP_SUFFIXES = {".part", ".tmp", ".log"}
SKIP_NAMES = {".ds_store", "thumbs.db"}
COMMIT_BATCH = 32
DEFAULT_MODEL = "Qwen_Qwen3.5-2B"
MODEL_REPOS = {
    "Qwen_Qwen3.5-2B": "qwen-snmf-features",
    "Qwen_Qwen2.5-3B-Instruct": "qwen2.5-snmf-features",
}
README_SPECS = {
    "Qwen_Qwen3.5-2B": {
        "pretty_name": "Qwen3.5 SNMF features for unlearning",
        "hf_model": "Qwen/Qwen3.5-2B",
        "tag": "qwen3.5",
        "config_name": "qwen",
        "sibling": ("Qwen/Qwen2.5-3B-Instruct", "shirasko/qwen2.5-snmf-features"),
    },
    "Qwen_Qwen2.5-3B-Instruct": {
        "pretty_name": "Qwen2.5 SNMF features for unlearning",
        "hf_model": "Qwen/Qwen2.5-3B-Instruct",
        "tag": "qwen2.5",
        "config_name": "qwen25",
        "sibling": ("Qwen/Qwen3.5-2B", "shirasko/qwen-snmf-features"),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--username", default=os.environ.get("HF_USERNAME", "shirasko"),
                   help="HF namespace for the dataset repo (default: %(default)s)")
    p.add_argument("--repo-name", default=None,
                   help="Dataset repo name under --username. Default: inferred "
                        "from --models via MODEL_REPOS.")
    p.add_argument("--repo-id", default=None,
                   help="Full dataset id, e.g. shirasko/qwen-snmf-features. "
                        "Overrides --username/--repo-name.")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="Local mf_outputs directory (default: repo mf_outputs/)")
    p.add_argument("--models", nargs="*", default=[DEFAULT_MODEL],
                   help="Model directory names under mf_outputs/ to upload. "
                        "Empty list means every model dir.")
    p.add_argument("--public", action="store_true", help="Create/keep the dataset public")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-upload files that already exist on the Hub")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-readme", action="store_true")
    p.add_argument("--readme-only", action="store_true",
                   help="Replace README.md on the Hub and skip feature files.")
    return p.parse_args()


def _rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_feature_files(root: Path, models: list[str] | None) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"mf_outputs not found: {root}")

    model_filter = set(models) if models else None
    files: list[Path] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if model_filter is not None and model_dir.name not in model_filter:
            continue
        for kind in KINDS:
            kind_dir = model_dir / kind
            if not kind_dir.is_dir():
                continue
            for path in kind_dir.rglob("*"):
                if not path.is_file():
                    continue
                if path.name.lower() in SKIP_NAMES:
                    continue
                if path.suffix.lower() in SKIP_SUFFIXES:
                    continue
                if any(part == "__pycache__" for part in path.parts):
                    continue
                files.append(path)
    return files


def resolve_repo_id(username: str, repo_id: str | None, repo_name: str | None,
                    models: list[str] | None) -> str:
    if repo_id:
        return repo_id
    if repo_name:
        return f"{username}/{repo_name}"
    keys = models or list(MODEL_REPOS)
    names = {MODEL_REPOS.get(m) for m in keys}
    names.discard(None)
    if len(names) == 1:
        return f"{username}/{next(iter(names))}"
    if not names:
        return f"{username}/{MODEL_REPOS[DEFAULT_MODEL]}"
    raise SystemExit(
        "Models map to different dataset repos; pass --repo-id or --repo-name, "
        f"or upload one model at a time. Got: {keys}"
    )


def dataset_readme(repo_id: str, models: list[str] | None) -> str:
    model_dir = (models[0] if models and len(models) == 1 else DEFAULT_MODEL)
    spec = README_SPECS.get(model_dir, README_SPECS[DEFAULT_MODEL])
    sibling_model, sibling_repo = spec["sibling"]
    title = spec["pretty_name"]
    return f"""---
license: mit
pretty_name: {title}
language:
- en
tags:
- unlearning
- snmf
- qwen
- {spec["tag"]}
size_categories:
- n<1K
configs:
  - config_name: {spec["config_name"]}
    data_files:
      - split: mlp
        path: "{model_dir}/interpretations/rank100/seed42/*/mlp/potential_features.csv"
---

# {title}

MLP Semi-NMF factorizations and (when present) LLM interpretations for
**SNMF concept unlearning** on `{spec["hf_model"]}` (rank 100, seed 42).

These files are the SNMF track only: per-layer MLP directions used to project
concept features out of `up_proj` / `down_proj`. There is no embedding-matrix
factorization in this dataset.

`{sibling_model}` features live in [`{sibling_repo}`](https://huggingface.co/datasets/{sibling_repo}).

Layout (same directory scheme used by the training/unlearning pipeline):

`<model>/{{pickles,csvs,interpretations}}/rank<R>/seed<S>/<concept>/mlp/`

- `pickles/mlp/layer*.pkl` — Semi-NMF fit on MLP activations (one pickle per layer).
- `csvs/mlp/` — per-token feature scores and concept-vs-neutral statistics.
- `interpretations/mlp/` — Gemini-labeled features (`from_activation.csv`,
  `from_projection.csv`) and the selected set `potential_features.csv` used
  by SNMF unlearning.

## Download

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="{repo_id}", repo_type="dataset", local_dir="mf_outputs",
                  allow_patterns=["{model_dir}/**"])
```
"""


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: HF_TOKEN is not set (add it to .env)", file=sys.stderr)
        raise SystemExit(1)

    models = list(args.models) if args.models else None
    repo_id = resolve_repo_id(args.username, args.repo_id, args.repo_name, models)
    root = args.root.resolve()
    local_files = iter_feature_files(root, models)

    print(f"Local root : {root}")
    print(f"Hub dataset: {repo_id}")
    print(f"Models     : {models or '(all)'}")
    print(f"Local files: {len(local_files)}")

    from huggingface_hub import CommitOperationAdd, HfApi, create_repo

    api = HfApi(token=token)
    if not args.dry_run:
        create_repo(
            repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=not args.public,
            token=token,
        )
        if args.public:
            # huggingface_hub>=1.0 replaced update_repo_visibility with update_repo_settings.
            if hasattr(api, "update_repo_settings"):
                api.update_repo_settings(repo_id, private=False, repo_type="dataset")
            else:
                api.update_repo_visibility(repo_id, private=False, repo_type="dataset")

    remote: set[str] = set()
    if api.repo_exists(repo_id, repo_type="dataset"):
        remote = set(api.list_repo_files(repo_id, repo_type="dataset"))
    print(f"Remote files: {len(remote)}")

    to_upload: list[tuple[str, Path]] = []
    skipped = 0
    if not args.readme_only:
        for path in local_files:
            rel = _rel_posix(path, root)
            if rel in remote and not args.overwrite:
                skipped += 1
                continue
            to_upload.append((rel, path))

    readme_rel = "README.md"
    write_readme = (not args.skip_readme) and (
        args.readme_only or args.overwrite or readme_rel not in remote
    )
    if write_readme:
        readme_text = dataset_readme(repo_id, models)
        if args.dry_run:
            to_upload.append((readme_rel, Path("<generated README.md>")))
        else:
            tmp = root / ".ember_features_README.md"
            tmp.write_text(readme_text, encoding="utf-8")
            to_upload.append((readme_rel, tmp))

    print(f"Skip existing: {skipped}")
    print(f"To upload    : {len(to_upload)}")

    if args.dry_run:
        for rel, path in to_upload[:40]:
            print(f"  would upload  {rel}")
        if len(to_upload) > 40:
            print(f"  ... {len(to_upload) - 40} more")
        return

    if not to_upload:
        print("Nothing new to upload.")
        return

    tmp_readme = root / ".ember_features_README.md"
    try:
        for start in range(0, len(to_upload), COMMIT_BATCH):
            batch = to_upload[start:start + COMMIT_BATCH]
            ops = [
                CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(path))
                for rel, path in batch
            ]
            msg = (
                f"Add {len(batch)} mf_outputs files "
                f"({start + 1}-{start + len(batch)}/{len(to_upload)})"
            )
            print(msg)
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=ops,
                commit_message=msg,
            )
    finally:
        if tmp_readme.exists():
            tmp_readme.unlink()

    print(f"Done. https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
