#!/usr/bin/env bash
# Reproduce one optimal unlearning config from jobs.manifest.tsv.
#
# Usage:
#   ./scripts/reproduce_unlearning.sh [task_index]
#   SLURM_ARRAY_TASK_ID=3 ./scripts/reproduce_unlearning.sh
#
# Prereqs:
#   - configs/reproduce_optimal_configs/jobs.manifest.tsv
#     (generate via: python scripts/generate_reproduce_configs.py)
#   - local MLP features under mf_outputs/ when using SNMF
#   - HF models available via HUGGINGFACE_HUB_CACHE in .env (see .env.example)

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SLURM_LOG_SUBDIR="${SLURM_LOG_SUBDIR:-reproduce_unlearning}"

# shellcheck source=scripts/ember_runner_env.sh
source "${REPO_ROOT}/scripts/ember_runner_env.sh"

_has_gemini_key() {
    [[ -n "${GOOGLE_API_KEY:-}" || -n "${GEMINI_API_KEY:-}" || -n "${GEMINI_API_TOKEN:-}" ]]
}

TASK_ID="${1:-${SLURM_ARRAY_TASK_ID:-0}}"

MANIFEST="${REPO_ROOT}/configs/reproduce_optimal_configs/jobs.manifest.tsv"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing manifest: $MANIFEST" >&2
    echo "Run: python scripts/generate_reproduce_configs.py" >&2
    exit 1
fi

JOB_LINE=$((TASK_ID + 1))
IFS=$'\t' read -r CONFIG CONCEPT TRAIN_EVAL < <(
    awk -F'\t' -v n="$JOB_LINE" 'NR==n {print; exit}' "$MANIFEST"
)
if [[ -z "${CONFIG:-}" ]]; then
    echo "No manifest entry for task ${TASK_ID} (line ${JOB_LINE})" >&2
    exit 1
fi

if [[ ! -f "${REPO_ROOT}/${CONFIG}" ]]; then
    echo "Config not found: ${REPO_ROOT}/${CONFIG}" >&2
    exit 1
fi

if [[ "${TRAIN_EVAL}" == "open" ]]; then
    if [[ "${SKIP_LLM_JUDGE:-0}" == "1" ]]; then
        echo "Error: train_eval=open requires Gemini LLM judging." >&2
        echo "       Refusing --skip-llm-judge / SKIP_LLM_JUDGE=1 for open-mode jobs." >&2
        exit 1
    fi
    if ! _has_gemini_key; then
        echo "Error: train_eval=open requires GOOGLE_API_KEY (or GEMINI_API_KEY)." >&2
        echo "       Add it to ${REPO_ROOT}/.env before running open-mode jobs." >&2
        echo "       Open-QA scoring cannot fall back to MC mode." >&2
        exit 1
    fi
elif ! _has_gemini_key && [[ "${SKIP_LLM_JUDGE:-0}" != "1" ]]; then
    echo "Error: GOOGLE_API_KEY is not set. Add it to ${REPO_ROOT}/.env" >&2
    echo "       (or GEMINI_API_KEY; required for Alpaca/open-QA scoring)." >&2
    echo "       Set SKIP_LLM_JUDGE=1 only for MC-only reproduce jobs." >&2
    exit 1
fi

echo "================================================================"
echo " reproduce_unlearning | job=${SLURM_JOB_ID:-local} task=${TASK_ID}"
echo " Node:     ${SLURMD_NODENAME:-local}"
echo " Repo:     $REPO_ROOT"
echo " Config:   $CONFIG"
echo " Concept:  $CONCEPT"
echo " Eval:     $TRAIN_EVAL"
echo " HF hub:   ${HUGGINGFACE_HUB_CACHE:-<default>}"
echo " HF_HOME:  ${HF_HOME:-<default>}"
echo " Skip LLM: ${SKIP_LLM_JUDGE:-0}"
echo " Overwrite:${REPRODUCE_OVERWRITE:-0}"
echo "================================================================"

EXTRA_ARGS=()
if [[ "${SKIP_LLM_JUDGE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-llm-judge)
fi
if [[ "${REPRODUCE_OVERWRITE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--overwrite)
fi

python -m ember.run_erasure \
    --config "${REPO_ROOT}/${CONFIG}" \
    --concepts "${CONCEPT}" \
    --train-eval "${TRAIN_EVAL}" \
    --features-source local \
    "${EXTRA_ARGS[@]}"

echo "Done: ${CONFIG}"
