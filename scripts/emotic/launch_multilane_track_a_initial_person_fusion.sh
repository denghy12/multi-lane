#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
FULL_RUN="${FULL_RUN:?FULL_RUN is required}"
VALIDATION_FUSION_SUMMARY="${VALIDATION_FUSION_SUMMARY:?VALIDATION_FUSION_SUMMARY is required}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT}/output/emotic_track_a_initial_person_fusion}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_initial_person_fusion}"
CONTROL_DIR="${CONTROL_DIR:-${OUTPUT_BASE}/control}"
BATCH_ID="${BATCH_ID:-initial_person_fixed_fusion_seed0_$(date +%Y%m%d_%H%M%S)}"
PERSON_RUN_ID="${BATCH_ID}_initial_person"
PERSON_RUN="${OUTPUT_BASE}/${PERSON_RUN_ID}"
FUSION_OUTPUT="${CONTROL_DIR}/${BATCH_ID}_fixed_fusion.json"
LAUNCH_LOG="${LOG_DIR}/${BATCH_ID}_launcher.log"

[[ -d "${FULL_RUN}" ]] || { echo "Missing Full source run: ${FULL_RUN}" >&2; exit 2; }
[[ -f "${VALIDATION_FUSION_SUMMARY}" ]] || {
  echo "Missing validation fusion summary: ${VALIDATION_FUSION_SUMMARY}" >&2
  exit 2
}
[[ ! -e "${PERSON_RUN}" ]] || { echo "Person run already exists: ${PERSON_RUN}" >&2; exit 2; }
[[ ! -e "${FUSION_OUTPUT}" ]] || { echo "Fusion output already exists: ${FUSION_OUTPUT}" >&2; exit 2; }
mkdir -p "${OUTPUT_BASE}" "${CONTROL_DIR}" "${LOG_DIR}"

exec > >(tee "${LAUNCH_LOG}") 2>&1

echo "Initial Person fixed fusion batch: batch_id=${BATCH_ID} gpu=${GPU} full_run=${FULL_RUN} person_run=${PERSON_RUN} validation_selection=${VALIDATION_FUSION_SUMMARY} rule=probability/full0.80/person0.20/threshold0.5 test_search=off"

GPU="${GPU}" \
RUN_ID="${PERSON_RUN_ID}" \
OUTPUT_BASE="${OUTPUT_BASE}" \
LOG_DIR="${LOG_DIR}" \
bash scripts/emotic/run_multilane_track_a_initial_person_score_test.sh

python_bin="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
"${python_bin}" -m multi_lane.track_a.fuse_initial_person_fixed_test_scores \
  --full-run "${FULL_RUN}" \
  --person-run "${PERSON_RUN}" \
  --validation-fusion-summary "${VALIDATION_FUSION_SUMMARY}" \
  --output "${FUSION_OUTPUT}"

echo "INITIAL_PERSON_FIXED_FUSION_BATCH_COMPLETE batch_id=${BATCH_ID} person_run=${PERSON_RUN} fusion_output=${FUSION_OUTPUT}"
