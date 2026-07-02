#!/usr/bin/env bash
# ==============================================================================
# Environment initialization for EMBER SLURM / shell runners
# Usage: source scripts/ember_runner_env.sh
#
# Configuration layering (first non-empty value wins):
#   1. Variables already exported in the shell
#   2. ${REPO_ROOT}/.env — local secrets and machine-specific paths
#   3. Defaults below — derived from WORKSPACE_ROOT / REPO_ROOT
# ==============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[-] Error: This initialization script must be sourced, not executed directly." >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# 1. Path topology & workspace resolution
# ------------------------------------------------------------------------------
_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${_ENV_SCRIPT_DIR}/.." && pwd)}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

export WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
export EMBER_ERASURE_ROOT="${EMBER_ERASURE_ROOT:-${REPO_ROOT}}"

export CONDA_HOME="${CONDA_HOME:-${WORKSPACE_ROOT}/miniconda3}"
export TARGET_CONDA_ENV="${TARGET_CONDA_ENV:-${WORKSPACE_ROOT}/envs/ember}"

# ------------------------------------------------------------------------------
# 2. Conda bootstrap & environment activation
# ------------------------------------------------------------------------------
_CONDA_EXEC="${CONDA_HOME}/bin/conda"

if [[ -x "$_CONDA_EXEC" ]]; then
    if _CONDA_HOOK="$("$_CONDA_EXEC" shell.bash hook 2>/dev/null)"; then
        eval "$_CONDA_HOOK"
    elif [[ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${CONDA_HOME}/etc/profile.d/conda.sh"
    else
        export PATH="${CONDA_HOME}/bin:${PATH}"
    fi
    unset _CONDA_HOOK

    if ! conda activate "$TARGET_CONDA_ENV" 2>/dev/null; then
        if ! conda activate ember 2>/dev/null; then
            echo "[-] Error: Could not activate required Conda environment (tried: ${TARGET_CONDA_ENV}, ember)." >&2
            return 1
        fi
    fi
else
    echo "[-] Warning: Conda binary unresolved at $_CONDA_EXEC" >&2
fi

# ------------------------------------------------------------------------------
# 3. Cache topologies & runtime variables
# ------------------------------------------------------------------------------
export CACHE_ROOT="${CACHE_ROOT:-${WORKSPACE_ROOT}/hf_cache}"

# Accept either EMBER-style or sibling-repo alias for the shared model hub.
if [[ -n "${SHARED_HF_HUB_CACHE:-}" && -z "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
    export HUGGINGFACE_HUB_CACHE="${SHARED_HF_HUB_CACHE}"
fi

export HF_HOME="${HF_HOME:-${CACHE_ROOT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$TMPDIR"

# Hugging Face / Gemini keys are loaded from .env; ensure HF_TOKEN is visible to hub clients.
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
fi

# Gemini judge: Vertex AI express uses GOOGLE_API_KEY. Keep one env var set so
# google-genai does not warn on every Client() construction.
if [[ -n "${GOOGLE_API_KEY:-}" ]]; then
    export GOOGLE_API_KEY
    unset GEMINI_API_KEY
elif [[ -n "${GEMINI_API_KEY:-}" ]]; then
    export GOOGLE_API_KEY="${GEMINI_API_KEY}"
    unset GEMINI_API_KEY
fi

# SLURM logs: slurm_outputs/<subdir>/ and slurm_errors/<subdir>/ (when SLURM_LOG_SUBDIR is set).
if [[ -n "${SLURM_LOG_SUBDIR:-}" ]]; then
    mkdir -p "${REPO_ROOT}/slurm_outputs/${SLURM_LOG_SUBDIR}" \
             "${REPO_ROOT}/slurm_errors/${SLURM_LOG_SUBDIR}"
fi

# ------------------------------------------------------------------------------
# 4. Python environment invariants
# ------------------------------------------------------------------------------
cd "$REPO_ROOT" || return 1

if [[ -z "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$(pwd)"
else
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
fi

echo "[+] EMBER execution context initialized."
echo "    -> Workspace : $REPO_ROOT"
echo "    -> Active env: ${CONDA_DEFAULT_ENV:-UNRESOLVED}"
echo "    -> HF hub    : ${HUGGINGFACE_HUB_CACHE}"
echo "    -> HF_HOME   : ${HF_HOME}"
if [[ -n "${GOOGLE_API_KEY:-}" ]]; then
    echo "    -> Gemini    : configured (GOOGLE_API_KEY)"
else
    echo "    -> Gemini    : not set (Alpaca/open-QA judge will fail)"
fi
