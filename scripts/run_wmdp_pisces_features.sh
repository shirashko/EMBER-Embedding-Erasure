#!/usr/bin/env bash
# Run WMDP PISCES feature-finding command, then upsert .concept to JSON.
#
# This wrapper is intentionally command-driven because external/PISCES/feature_finder.py
# does not expose a stable one-shot CLI for all required inputs.
#
# Required env vars:
#   CONCEPT               e.g. wmdp-bio / wmdp-cyber
#
# Optional env vars:
#   FEATURES_JSON         target features JSON (default: data/pisces_concept_features_gemma.json)
#   CONCEPT_FILE          path to produced .concept JSON file
#                         (default: pisces_outputs/<concept>.concept)
#   FEATURE_FINDER_CMD    shell command that generates CONCEPT_FILE
#                         (default: scripts/build_wmdp_pisces_concept.py driver)
#   UPSERT_BACKUP         1/0 (default: 1) write .bak before update
#   SKIP_UPSERT           1/0 (default: 0) skip JSON update stage
#
# Example:
#   CONCEPT=wmdp-bio \
#   CONCEPT_FILE=/home/.../pisces_outputs/wmdp-bio.concept \
#   FEATURE_FINDER_CMD='python /home/.../my_wmdp_bio_feature_finder_driver.py' \
#   ./scripts/run_wmdp_pisces_features.sh

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SLURM_LOG_SUBDIR="${SLURM_LOG_SUBDIR:-run_wmdp_pisces_features}"
export FEATURES_JSON="${FEATURES_JSON:-${REPO_ROOT}/data/pisces_concept_features_gemma.json}"
export UPSERT_BACKUP="${UPSERT_BACKUP:-1}"
export SKIP_UPSERT="${SKIP_UPSERT:-0}"

# shellcheck source=scripts/ember_runner_env.sh
source "${REPO_ROOT}/scripts/ember_runner_env.sh"

if [[ -z "${CONCEPT:-}" ]]; then
    echo "Error: CONCEPT is required (e.g., wmdp-bio)." >&2
    exit 1
fi
export CONCEPT_FILE="${CONCEPT_FILE:-${REPO_ROOT}/pisces_outputs/${CONCEPT}.concept}"
if [[ -z "${FEATURE_FINDER_CMD:-}" ]]; then
    FEATURE_FINDER_CMD="python ${REPO_ROOT}/scripts/build_wmdp_pisces_concept.py --concept ${CONCEPT} --output ${CONCEPT_FILE}"
fi

echo "================================================================"
echo " run_wmdp_pisces_features | job=${SLURM_JOB_ID:-local}"
echo " Node:              ${SLURMD_NODENAME:-local}"
echo " Repo:              ${REPO_ROOT}"
echo " Concept:           ${CONCEPT}"
echo " Concept file:      ${CONCEPT_FILE}"
echo " Features JSON:     ${FEATURES_JSON}"
echo " Skip upsert:       ${SKIP_UPSERT}"
echo " Feature cmd:       ${FEATURE_FINDER_CMD}"
echo "================================================================"

# Run caller-provided feature-finder command.
eval "${FEATURE_FINDER_CMD}"

if [[ ! -f "${CONCEPT_FILE}" ]]; then
    echo "Error: expected concept file not found after feature finder run: ${CONCEPT_FILE}" >&2
    exit 1
fi

if [[ "${SKIP_UPSERT}" == "1" ]]; then
    echo "Skipping JSON upsert (SKIP_UPSERT=1)."
    exit 0
fi

UPSERT_ARGS=(
    --features-json "${FEATURES_JSON}"
    --concept-file "${CONCEPT_FILE}"
    --concept-name "${CONCEPT}"
)
if [[ "${UPSERT_BACKUP}" == "1" ]]; then
    UPSERT_ARGS+=(--backup)
fi

python "${REPO_ROOT}/scripts/upsert_pisces_features.py" "${UPSERT_ARGS[@]}"
echo "Done: ${CONCEPT}"
