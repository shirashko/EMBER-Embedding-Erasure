#!/bin/bash
# Upload unlearned checkpoints to Hugging Face Hub (sequential CPU job).
# Submit from repo root:
#   mkdir -p slurm_outputs/hf_upload slurm_errors/hf_upload
#   sbatch scripts/hf/slurm/upload.slurm
#
# Environment (sbatch forwards the submitting shell's environment):
#   EMBER_ROOT  repository checkout (default: the sbatch submit directory)
#   CONDA_ENV   conda env name or prefix to activate (default: ember)
#   CONDA_HOME  conda install prefix (default: see scripts/ember_runner_env.sh)
#   HF_TOKEN    set in EMBER_ROOT/.env
# Pass your mail address at submit time (#SBATCH lines cannot expand variables):
#   --mail-user="${SLURM_MAIL_USER}" --mail-type=END,FAIL
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

set -euo pipefail

REPO_ROOT="${EMBER_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export REPO_ROOT
HF_USERNAME="${HF_USERNAME:-shirasko}"
HF_DIR="${REPO_ROOT}/scripts/hf"
cd "${REPO_ROOT}"

mkdir -p slurm_outputs/hf_upload slurm_errors/hf_upload

# Loads .env, activates conda and sets HF_HOME / TMPDIR (defaults in that script).
export TARGET_CONDA_ENV="${CONDA_ENV:-ember}"
# shellcheck source=scripts/ember_runner_env.sh
source "${REPO_ROOT}/scripts/ember_runner_env.sh"

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
