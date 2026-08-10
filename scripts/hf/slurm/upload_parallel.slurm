#!/bin/bash
# Parallel upload of checkpoints to Hugging Face (4 concurrent CPU jobs).
# Submit from repo root:
#   sbatch scripts/hf/slurm/upload_parallel.slurm
#
# Split: one array task per (method, base_model):
#   0 -> rmu  / google_gemma-2-2b-it
#   1 -> rmu  / meta-llama_Llama-3.1-8B-Instruct
#   2 -> snmf / google_gemma-2-2b-it
#   3 -> snmf / meta-llama_Llama-3.1-8B-Instruct

#SBATCH --job-name=hf_upload_par
#SBATCH --output=slurm_outputs/hf_upload/%A_%a.out
#SBATCH --error=slurm_errors/hf_upload/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-3
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --chdir=/home/morg/students/rashkovits/EMBER-Embedding-Erasure

set -euo pipefail

REPO_ROOT="/home/morg/students/rashkovits/EMBER-Embedding-Erasure"
HF_USERNAME="${HF_USERNAME:-shirasko}"
HF_DIR="${REPO_ROOT}/scripts/hf"

METHODS=(rmu rmu snmf snmf)
BASE_MODELS=(
    google_gemma-2-2b-it
    meta-llama_Llama-3.1-8B-Instruct
    google_gemma-2-2b-it
    meta-llama_Llama-3.1-8B-Instruct
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
METHOD="${METHODS[$TASK_ID]}"
BASE_MODEL="${BASE_MODELS[$TASK_ID]}"

mkdir -p slurm_outputs/hf_upload slurm_errors/hf_upload

source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate ember

export HF_HOME="${HF_HOME:-/home/morg/students/rashkovits/hf_cache}"
export TMPDIR="${TMPDIR:-$HF_HOME/tmp}"
mkdir -p "$HF_HOME" "$TMPDIR"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. Add it to ${REPO_ROOT}/.env" >&2
    exit 1
fi

export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
export PYTHONPATH="${HF_DIR}:${PYTHONPATH}"

echo "================================================================"
echo " hf_upload_par | Node: ${SLURMD_NODENAME:-local} | Job: ${SLURM_JOB_ID:-local} | Task: ${TASK_ID}"
echo " Method:     ${METHOD}"
echo " Base model: ${BASE_MODEL}"
echo " Username:   ${HF_USERNAME}"
echo "================================================================"

exec python "${HF_DIR}/upload.py" \
    --username "$HF_USERNAME" \
    --public \
    --skip-complete \
    --methods "$METHOD" \
    --base-models "$BASE_MODEL"
