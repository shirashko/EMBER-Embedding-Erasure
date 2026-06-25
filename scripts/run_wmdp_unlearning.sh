#!/usr/bin/env bash
# Run one WMDP CRISP unlearning config from configs/wmdp_configs/jobs.manifest.tsv.
#
# Usage:
#   ./scripts/run_wmdp_unlearning.sh [task_index]
#   SLURM_ARRAY_TASK_ID=2 ./scripts/run_wmdp_unlearning.sh
#
# Prereqs:
#   - Prepared corpora: python scripts/prepare_wmdp_corpora.py --domain all
#   - Prepared MCQ eval: python scripts/prepare_wmdp_mcq.py --domain all
#   - HF_TOKEN in .env for gated models (Llama-3.1-8B)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

if [[ -n "${HF_HOME:-}" ]]; then
    export TMPDIR="${TMPDIR:-$HF_HOME/tmp}"
    mkdir -p "$HF_HOME" "$TMPDIR"
fi

TASK_ID="${1:-${SLURM_ARRAY_TASK_ID:-0}}"

MANIFEST="${REPO_ROOT}/configs/wmdp_configs/jobs.manifest.tsv"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing manifest: $MANIFEST" >&2
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

echo "================================================================"
echo " run_wmdp_unlearning | job=${SLURM_JOB_ID:-local} task=${TASK_ID}"
echo " Node:     ${SLURMD_NODENAME:-local}"
echo " Repo:     $REPO_ROOT"
echo " Config:   $CONFIG"
echo " Concept:  $CONCEPT"
echo " Eval:     $TRAIN_EVAL"
echo " HF hub:   ${HUGGINGFACE_HUB_CACHE:-<default>}"
echo " HF_HOME:  ${HF_HOME:-<default>}"
echo "================================================================"

python -m ember.run_erasure \
    --config "${REPO_ROOT}/${CONFIG}" \
    --concepts "${CONCEPT}" \
    --train-eval "${TRAIN_EVAL}" \
    --features-source local

echo "Done: ${CONFIG}"
