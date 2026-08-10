#!/usr/bin/env bash
# Shared environment for Hugging Face Hub scripts.
HF_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HF_SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${HF_SCRIPT_DIR}:${PYTHONPATH}"

if [ -f "${REPO_ROOT}/scripts/ember_runner_env.sh" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/scripts/ember_runner_env.sh"
fi

cd "${REPO_ROOT}"
