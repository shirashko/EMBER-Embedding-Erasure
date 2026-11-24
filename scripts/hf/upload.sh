#!/usr/bin/env bash
# Upload unlearned checkpoints to Hugging Face Hub.
# Usage:
#   ./scripts/hf/upload.sh <hf_username> [--dry-run] [--public] [--skip-complete]

set -e

HF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HF_DIR}/_env.sh"

if [[ -z "$1" ]]; then
    echo "Usage: $0 <hf_username> [--dry-run] [--public] [--skip-complete] [--make-public-existing]"
    exit 1
fi

USERNAME=$1
shift
EXTRA_ARGS=()
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            EXTRA_ARGS+=(--dry_run)
            echo "[!] Performing a DRY RUN..."
            ;;
        --public)
            EXTRA_ARGS+=(--public)
            ;;
        --skip-complete)
            EXTRA_ARGS+=(--skip-complete)
            ;;
        --make-public-existing)
            EXTRA_ARGS+=(--make-public-existing)
            ;;
        --methods)
            shift
            EXTRA_ARGS+=(--methods "$1")
            ;;
        --base-models)
            shift
            EXTRA_ARGS+=(--base-models "$1")
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [[ "$DRY_RUN" == false ]]; then
    if ! huggingface-cli whoami &> /dev/null && [ -z "$HF_TOKEN" ]; then
        echo "[ERROR] You are not logged in to Hugging Face Hub." >&2
        echo "Please run 'huggingface-cli login' or set the HF_TOKEN environment variable." >&2
        exit 1
    fi
fi

python "${HF_DIR}/upload.py" --username "$USERNAME" "${EXTRA_ARGS[@]}"
