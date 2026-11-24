#!/bin/bash
# Generic SLURM array job for EMBER job manifests: one manifest row per array task.
#
# The job reads row SLURM_ARRAY_TASK_ID (0-based) of MANIFEST, logs it, and hands the
# task index to RUNNER, which executes that row.
#
# Environment (sbatch forwards the submitting shell's environment):
#   MANIFEST         required. jobs.manifest*.tsv, absolute or relative to EMBER_ROOT.
#   RUNNER           required. Runner script for that manifest, relative to EMBER_ROOT:
#                      scripts/reproduce_unlearning.sh        reproduce_optimal_configs/jobs.manifest*.tsv
#                      scripts/run_wmdp_unlearning.sh         WMDP, Qwen and experiment unlearning manifests
#                      scripts/run_wmdp_train_mf_features.sh  *train_mf_features.tsv manifests
#   EMBER_ROOT       repository checkout. Defaults to the sbatch submit directory.
#   CONDA_ENV        conda env name or prefix to activate (default: see ember_runner_env.sh).
#   CONDA_HOME       conda install prefix (default: see ember_runner_env.sh).
#   CHECKPOINT_ROOT, OVERWRITE, SKIP_LLM_JUDGE, REPRODUCE_OVERWRITE, ...
#                    forwarded unchanged to the runner.
# Secrets and machine-specific paths (HF_TOKEN, GOOGLE_API_KEY, HF caches) belong in
# EMBER_ROOT/.env, which the runners load via scripts/ember_runner_env.sh.
#
# Cluster-specific values cannot be set in #SBATCH lines (no variable expansion
# there). Pass them at submit time, e.g. from your own SLURM_PARTITION,
# SLURM_ACCOUNT and SLURM_MAIL_USER shell variables:
#
#   mkdir -p slurm_outputs slurm_errors
#   RUNNER=scripts/run_wmdp_unlearning.sh \
#   MANIFEST=configs/qwen_configs/jobs.manifest.crisp.tsv \
#     sbatch --array=0-9 \
#       --partition="${SLURM_PARTITION}" --account="${SLURM_ACCOUNT}" \
#       --mail-user="${SLURM_MAIL_USER}" --mail-type=END,FAIL \
#       scripts/slurm/ember_array.slurm
#
# Resources below are defaults; override per submission (e.g. --time=24:00:00,
# --constraint=<gpu type>, --mem=80G --cpus-per-task=16 for MF-feature training).

#SBATCH --job-name=ember_array
#SBATCH --output=slurm_outputs/%x_%A_%a.out
#SBATCH --error=slurm_errors/%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

EMBER_ROOT="${EMBER_ROOT:-${SLURM_SUBMIT_DIR:-}}"
: "${EMBER_ROOT:?set EMBER_ROOT to the repository checkout}"
: "${MANIFEST:?set MANIFEST to a jobs.manifest*.tsv file}"
: "${RUNNER:?set RUNNER to the runner script for this manifest}"
: "${SLURM_ARRAY_TASK_ID:?submit as an array job (sbatch --array=...)}"

cd "${EMBER_ROOT}"
export REPO_ROOT="${EMBER_ROOT}"
if [[ "${MANIFEST}" != /* ]]; then
    MANIFEST="${EMBER_ROOT}/${MANIFEST}"
fi
export MANIFEST
if [[ -n "${CONDA_ENV:-}" ]]; then
    export TARGET_CONDA_ENV="${CONDA_ENV}"
fi
export SLURM_LOG_SUBDIR="${SLURM_LOG_SUBDIR:-${SLURM_JOB_NAME:-ember_array}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing manifest: ${MANIFEST}" >&2
    exit 1
fi
if [[ ! -x "${EMBER_ROOT}/${RUNNER}" ]]; then
    echo "Runner not found or not executable: ${EMBER_ROOT}/${RUNNER}" >&2
    exit 1
fi

# reproduce_unlearning.sh reads its manifest from REPRODUCE_MANIFEST.
if [[ "${RUNNER}" == *reproduce_unlearning.sh ]]; then
    export REPRODUCE_MANIFEST="${MANIFEST}"
fi

ROW="$(awk -v n="$((SLURM_ARRAY_TASK_ID + 1))" 'NR == n { print; exit }' "${MANIFEST}")"
if [[ -z "${ROW}" ]]; then
    echo "No manifest row for task ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
    exit 1
fi

echo "[ember_array] job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID} node=${SLURMD_NODENAME:-local}"
echo "[ember_array] runner:   ${RUNNER}"
echo "[ember_array] manifest: ${MANIFEST}"
echo "[ember_array] row:      ${ROW}"

exec "${EMBER_ROOT}/${RUNNER}" "${SLURM_ARRAY_TASK_ID}"
