#!/usr/bin/env bash
# Publish the collection overview dataset and update collection metadata.
# Usage:
#   ./scripts/hf/setup_collection.sh shirasko
#   ./scripts/hf/setup_collection.sh shirasko --dry-run

set -e

HF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HF_DIR}/_env.sh"

if [[ -z "$1" ]]; then
    echo "Usage: $0 <hf_username> [--dry-run]"
    exit 1
fi

USERNAME=$1
shift
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            EXTRA_ARGS+=(--dry_run)
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

python "${HF_DIR}/setup_collection.py" --username "$USERNAME" "${EXTRA_ARGS[@]}"
