#!/usr/bin/env bash
# Full non-EMBER grid search for WMDP-Bio and WMDP-Cyber (RMU, CRISP, SNMF).
# PISCES is omitted. Gemini / LLM judge is skipped throughout.
#
# Prereqs:
#   1. WMDP concepts in data/:
#        python scripts/prepare_wmdp_concepts_for_ember.py --apply
#   2. CRISP: Gemma SAE cache (see slurm/download_crisp_saes.slurm).
#   3. SNMF: local MLP features + interpretations under mf_outputs/:
#        sbatch slurm/train_wmdp_mf_features.slurm
#
# Usage (local, one manifest row):
#   ./scripts/run_wmdp_unlearning.sh 0
#
# Run all manifest jobs sequentially on one machine:
#   RUN_ALL=1 ./scripts/run_wmdp_unlearning.sh
#
# Checkpoints (best HP after grid + validate) are saved under:
#   unlearned_checkpoints/<method>/<model>/<concept>/
# Override with CHECKPOINT_ROOT=... ; use OVERWRITE=1 to replace existing saves.
#
# SLURM array (one GPU job per row):
#   sbatch slurm/run_wmdp_unlearning.slurm
#   sbatch --array=0 slurm/run_wmdp_unlearning.slurm
#
# Manifest columns: config<TAB>concept<TAB>train_eval<TAB>features_source

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SLURM_LOG_SUBDIR="${SLURM_LOG_SUBDIR:-run_wmdp_unlearning}"
export SKIP_LLM_JUDGE=1
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-unlearned_checkpoints}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# shellcheck source=scripts/ember_runner_env.sh
source "${REPO_ROOT}/scripts/ember_runner_env.sh"

MANIFEST="${MANIFEST:-${REPO_ROOT}/configs/wmdp_configs/jobs.manifest.tsv}"
if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing manifest: $MANIFEST" >&2
    exit 1
fi

run_one() {
    local task_id="$1"
    local job_line=$((task_id + 1))
    local config concept train_eval features_source

    IFS=$'\t' read -r config concept train_eval features_source < <(
        awk -F'\t' -v n="$job_line" 'NR==n {print; exit}' "$MANIFEST"
    )

    if [[ -z "${config:-}" ]]; then
        echo "No manifest entry for task ${task_id} (line ${job_line})" >&2
        exit 1
    fi

    if [[ ! -f "${REPO_ROOT}/${config}" ]]; then
        echo "Config not found: ${REPO_ROOT}/${config}" >&2
        exit 1
    fi

    features_source="${features_source:-local}"

    echo "================================================================"
    echo " run_wmdp_unlearning | job=${SLURM_JOB_ID:-local} task=${task_id}"
    echo " Node:            ${SLURMD_NODENAME:-local}"
    echo " Repo:            $REPO_ROOT"
    echo " Config:          $config"
    echo " Concept:         $concept"
    echo " Train eval:      $train_eval"
    echo " Features source: $features_source"
    echo " Skip LLM judge:  1"
    echo " Checkpoint root: $CHECKPOINT_ROOT"
    echo " CUDA alloc conf: $PYTORCH_CUDA_ALLOC_CONF"
    echo "================================================================"

    local extra_args=(--skip-llm-judge --features-source "$features_source"
                      --checkpoint-root "$CHECKPOINT_ROOT")
    if [[ "${OVERWRITE:-0}" == "1" ]]; then
        extra_args+=(--overwrite)
    fi

    python -m ember.run_erasure \
        --config "${REPO_ROOT}/${config}" \
        --concepts "${concept}" \
        --train-eval "${train_eval}" \
        "${extra_args[@]}"

    echo "Done: ${config} | ${concept}"
}

if [[ "${RUN_ALL:-0}" == "1" ]]; then
    total=$(grep -cve '^\s*$' "$MANIFEST" || true)
    for ((i = 0; i < total; i++)); do
        run_one "$i"
    done
else
    TASK_ID="${1:-${SLURM_ARRAY_TASK_ID:-0}}"
    run_one "$TASK_ID"
fi
