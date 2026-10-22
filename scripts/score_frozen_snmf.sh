#!/usr/bin/env bash
# Score a frozen SNMF dictionary from a YAML config, then judge the shortlist.
#
# The YAML must set every hyperparameter. See
# configs/frozen_snmf/llama3_1_8b_it_rank200_init_random_mlp_intermediate.yaml.
#
# Submit from the repo root:
#   mkdir -p slurm_outputs/score_frozen_snmf slurm_errors/score_frozen_snmf
#   sbatch scripts/score_frozen_snmf.sh
#   sbatch scripts/score_frozen_snmf.sh configs/frozen_snmf/other.yaml
#
# Or run locally:
#   ./scripts/score_frozen_snmf.sh
# ==============================================================================
#SBATCH --job-name=snmf_score
#SBATCH --output=slurm_outputs/score_frozen_snmf/%j.out
#SBATCH --error=slurm_errors/score_frozen_snmf/%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${_SCRIPT_DIR}/.." && pwd)}}"
export SLURM_LOG_SUBDIR="${SLURM_LOG_SUBDIR:-score_frozen_snmf}"

# shellcheck source=scripts/ember_runner_env.sh
source "${REPO_ROOT}/scripts/ember_runner_env.sh"

CONFIG="${1:-configs/frozen_snmf/llama3_1_8b_it_rank200_init_random_mlp_intermediate.yaml}"
if [[ "${CONFIG}" != /* ]]; then
    CONFIG="${REPO_ROOT}/${CONFIG}"
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 1
fi

SKIP_INTERPRET="$(python - "${CONFIG}" <<'PY'
import sys
import yaml
payload = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print("1" if payload.get("skip_interpret") else "0")
PY
)"
if [[ "${SKIP_INTERPRET}" != "1" ]]; then
    if [[ -z "${GOOGLE_API_KEY:-}" && -z "${GEMINI_API_KEY:-}" && -z "${GEMINI_API_TOKEN:-}" ]]; then
        echo "Gemini key missing. Set GOOGLE_API_KEY, GEMINI_API_KEY, or GEMINI_API_TOKEN in ${REPO_ROOT}/.env" >&2
        echo "Or set skip_interpret: true in ${CONFIG} to score without the membership judge." >&2
        exit 1
    fi
fi

echo "================================================================"
echo " score_frozen_snmf | job=${SLURM_JOB_ID:-local}"
echo " Node:   ${SLURMD_NODENAME:-local}"
echo " Repo:   ${REPO_ROOT}"
echo " Config: ${CONFIG}"
echo "================================================================"

python -m ember.score_frozen_snmf --config "${CONFIG}"
