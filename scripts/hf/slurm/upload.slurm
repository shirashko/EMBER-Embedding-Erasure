#!/bin/bash
# Upload unlearned checkpoints to Hugging Face Hub (sequential CPU job).
# Submit from repo root:
#   sbatch scripts/hf/slurm/upload.slurm
#
# Optional environment overrides before submit:
#   export HF_USERNAME=shirasko
#   export UPLOAD_PUBLIC=1
#   export UPLOAD_SKIP_COMPLETE=1
#   export UPLOAD_METHODS="rmu snmf"
#   export UPLOAD_BASE_MODELS="google_gemma-2-2b-it"

#SBATCH --job-name=hf_upload
#SBATCH --output=slurm_outputs/hf_upload/%j.out
#SBATCH --error=slurm_errors/hf_upload/%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --chdir=/home/morg/students/rashkovits/EMBER-Embedding-Erasure

set -euo pipefail

REPO_ROOT="/home/morg/students/rashkovits/EMBER-Embedding-Erasure"
HF_USERNAME="${HF_USERNAME:-shirasko}"
HF_DIR="${REPO_ROOT}/scripts/hf"

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

UPLOAD_ARGS=(--username "$HF_USERNAME")
if [[ "${UPLOAD_PUBLIC:-}" == "1" ]]; then
    UPLOAD_ARGS+=(--public)
fi
if [[ "${UPLOAD_SKIP_COMPLETE:-}" == "1" ]]; then
    UPLOAD_ARGS+=(--skip-complete)
fi
if [[ -n "${UPLOAD_METHODS:-}" ]]; then
    # shellcheck disable=SC2206
    UPLOAD_ARGS+=(--methods ${UPLOAD_METHODS})
fi
if [[ -n "${UPLOAD_BASE_MODELS:-}" ]]; then
    # shellcheck disable=SC2206
    UPLOAD_ARGS+=(--base-models ${UPLOAD_BASE_MODELS})
fi

echo "================================================================"
echo " hf_upload | Node: ${SLURMD_NODENAME:-local} | Job: ${SLURM_JOB_ID:-local}"
echo " Username:  ${HF_USERNAME}"
echo " Args:      ${UPLOAD_ARGS[*]}"
echo "================================================================"

exec python "${HF_DIR}/upload.py" "${UPLOAD_ARGS[@]}"
