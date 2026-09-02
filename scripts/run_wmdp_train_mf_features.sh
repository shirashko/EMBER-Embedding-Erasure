#!/usr/bin/env bash
# Train + interpret MF features for WMDP concepts (SNMF/MLP track by default).
#
# Usage (local, one manifest row):
#   ./scripts/run_wmdp_train_mf_features.sh 0
#
# Run all rows sequentially:
#   RUN_ALL=1 ./scripts/run_wmdp_train_mf_features.sh
#
# SLURM array:
#   sbatch slurm/train_wmdp_mf_features.slurm
#   sbatch --array=0 slurm/train_wmdp_mf_features.slurm
#
# Manifest columns (tab-separated):
#   model_name<TAB>concept<TAB>rank<TAB>seed<TAB>tracks
#
# Notes:
#   - "tracks" is passed to interpret_features (default: mlp).
#   - By default this script trains only MLP features (--skip-embedding).
#     Set TRAIN_SKIP_EMBEDDING=0 to also train the embedding track.

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SLURM_LOG_SUBDIR="${SLURM_LOG_SUBDIR:-train_wmdp_mf_features}"
export OUTDIR="${OUTDIR:-mf_outputs}"
export TRAIN_SKIP_EMBEDDING="${TRAIN_SKIP_EMBEDDING:-1}"
export SKIP_INTERPRET="${SKIP_INTERPRET:-0}"
export RATIO_THRESH="${RATIO_THRESH:-2.0}"
export CONFIDENCE_THRESH="${CONFIDENCE_THRESH:-0.85}"
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash-lite}"
export MAX_WORKERS="${MAX_WORKERS:-5}"
export SAVE_EVERY="${SAVE_EVERY:-50}"

# shellcheck source=scripts/ember_runner_env.sh
source "${REPO_ROOT}/scripts/ember_runner_env.sh"

_default_manifest_1="${REPO_ROOT}/configs/wmdp_configs/jobs.manifest.train_mf_features.tsv"
_default_manifest_2="${REPO_ROOT}/configs/wmdp/jobs.manifest.train_mf_features.tsv"
if [[ -f "$_default_manifest_1" ]]; then
    MANIFEST_DEFAULT="$_default_manifest_1"
elif [[ -f "$_default_manifest_2" ]]; then
    MANIFEST_DEFAULT="$_default_manifest_2"
else
    MANIFEST_DEFAULT="$_default_manifest_1"
fi
MANIFEST="${MANIFEST:-$MANIFEST_DEFAULT}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing manifest: $MANIFEST" >&2
    echo "Expected tab-separated columns:" >&2
    echo "  model_name<TAB>concept<TAB>rank<TAB>seed<TAB>tracks" >&2
    echo "Example rows:" >&2
    echo "  google/gemma-2-2b-it  wmdp-bio    100 42  mlp" >&2
    echo "  google/gemma-2-2b-it  wmdp-cyber  100 42  mlp" >&2
    exit 1
fi

require_gemini() {
    if [[ "${SKIP_INTERPRET}" == "1" ]]; then
        return 0
    fi
    if [[ -z "${GOOGLE_API_KEY:-}" && -z "${GEMINI_API_KEY:-}" && -z "${GEMINI_API_TOKEN:-}" ]]; then
        echo "Error: GOOGLE_API_KEY is not set." >&2
        echo "Interpretation requires Gemini credentials." >&2
        echo "Set SKIP_INTERPRET=1 to train raw features without interpretation." >&2
        exit 1
    fi
}

manifest_line_for_task() {
    local task_id="$1"
    local row_num=$((task_id + 1))
    awk -F'\t' -v n="$row_num" '
        NF && $1 !~ /^[[:space:]]*#/ {
            c++
            if (c == n) {
                print
                exit
            }
        }
    ' "$MANIFEST"
}

manifest_total_rows() {
    awk '
        NF && $1 !~ /^[[:space:]]*#/ { c++ }
        END { print c + 0 }
    ' "$MANIFEST"
}

default_rank_for_model() {
    local model_name="$1"
    if [[ "$model_name" == *"llama"* || "$model_name" == *"Llama"* ]]; then
        echo "200"
    else
        echo "100"
    fi
}

run_one() {
    local task_id="$1"
    local line
    line="$(manifest_line_for_task "$task_id")"

    if [[ -z "${line:-}" ]]; then
        echo "No manifest entry for task ${task_id}" >&2
        exit 1
    fi

    local model_name concept rank seed tracks
    IFS=$'\t' read -r model_name concept rank seed tracks _ <<< "$line"

    model_name="${model_name:-google/gemma-2-2b-it}"
    concept="${concept:-}"
    rank="${rank:-$(default_rank_for_model "$model_name")}"
    seed="${seed:-42}"
    tracks="${tracks:-mlp}"

    if [[ -z "$concept" ]]; then
        echo "Manifest row for task ${task_id} is missing concept." >&2
        exit 1
    fi

    require_gemini

    echo "================================================================"
    echo " run_wmdp_train_mf_features | job=${SLURM_JOB_ID:-local} task=${task_id}"
    echo " Node:              ${SLURMD_NODENAME:-local}"
    echo " Repo:              $REPO_ROOT"
    echo " Manifest:          $MANIFEST"
    echo " Model:             $model_name"
    echo " Concept:           $concept"
    echo " Rank:              $rank"
    echo " Seed:              $seed"
    echo " Outdir:            $OUTDIR"
    echo " Train skip embed:  $TRAIN_SKIP_EMBEDDING"
    echo " Interpret tracks:  $tracks"
    echo " Skip interpret:    $SKIP_INTERPRET"
    echo "================================================================"

    train_args=(
        --concepts "$concept"
        --ranks "$rank"
        --seed "$seed"
        --model-name "$model_name"
        --outdir "$OUTDIR"
    )
    if [[ "${TRAIN_SKIP_EMBEDDING}" == "1" ]]; then
        train_args+=(--skip-embedding)
    fi

    python -m ember.train_mf_features "${train_args[@]}"

    if [[ "${SKIP_INTERPRET}" != "1" ]]; then
        python -m ember.interpret_features \
            --concepts "$concept" \
            --tracks "$tracks" \
            --rank "$rank" \
            --seed "$seed" \
            --model-name "$model_name" \
            --outdir "$OUTDIR" \
            --ratio-thresh "$RATIO_THRESH" \
            --confidence-thresh "$CONFIDENCE_THRESH" \
            --gemini-model "$GEMINI_MODEL" \
            --max-workers "$MAX_WORKERS" \
            --save-every "$SAVE_EVERY"
    fi

    echo "Done: model=${model_name} concept=${concept} rank=${rank} seed=${seed}"
}

if [[ "${RUN_ALL:-0}" == "1" ]]; then
    total="$(manifest_total_rows)"
    if [[ "$total" -le 0 ]]; then
        echo "Manifest has no runnable rows: $MANIFEST" >&2
        exit 1
    fi
    for ((i = 0; i < total; i++)); do
        run_one "$i"
    done
else
    TASK_ID="${1:-${SLURM_ARRAY_TASK_ID:-0}}"
    run_one "$TASK_ID"
fi
