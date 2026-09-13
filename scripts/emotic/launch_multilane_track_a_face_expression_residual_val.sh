#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
EXPECTED_SHA256="${EXPECTED_SHA256:?EXPECTED_SHA256 is required}"
GPU="${GPU:-}"
GPU_CANDIDATES=( ${GPU_CANDIDATES:-0 1 2 3 4 5 6 7} )
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
BATCH_ID="${BATCH_ID:-face_expression_residual_seed0_$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_expression_residual_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_face_expression_residual/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_face_expression_residual/${BATCH_ID}"
FULL_RUN="${FULL_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_full_anchor}"
PERSON_RUN="${PERSON_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_person_letterbox}"
ANCHOR_FACE="${ANCHOR_FACE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_endpoint_val_v0.1/face_endpoint_val_seed0_20260909_1320/face_endpoint_val_seed0_20260909_1320_face}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
RUN_ID="${BATCH_ID}_hybrid_residual"
RUN_ROOT="${RESULT_BASE}/${RUN_ID}"

[[ ! -e "${RESULT_BASE}" && ! -e "${CONTROL_DIR}" ]] || { echo "Batch output exists" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean worktree required" >&2; exit 2; }
for source in "${FULL_RUN}" "${PERSON_RUN}" "${ANCHOR_FACE}"; do
  [[ -f "${source}/seed_summary.json" ]] || { echo "Missing source: ${source}" >&2; exit 2; }
done
while [[ -z "${GPU}" ]]; do
  for candidate in "${GPU_CANDIDATES[@]}"; do
    free_mib="$(nvidia-smi -i "${candidate}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${free_mib}" -ge "${MIN_FREE_MIB}" ]]; then
      GPU="${candidate}"
      break
    fi
  done
  if [[ -z "${GPU}" ]]; then
    echo "Waiting for a GPU with at least ${MIN_FREE_MIB} MiB free"
    sleep "${WAIT_SECONDS}"
  fi
done
free_mib="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
[[ "${free_mib}" -ge "${MIN_FREE_MIB}" ]] || { echo "GPU ${GPU} has only ${free_mib} MiB free" >&2; exit 2; }
mkdir -p "${RESULT_BASE}" "${CONTROL_DIR}/status" "${LOG_DIR}"
printf 'variant\tgpu\tseed\trun\ttest_access\n' > "${CONTROL_DIR}/manifest.tsv"
printf 'clip_plus_expression_residual\t%s\t0\t%s\tforbidden\n' "${GPU}" "${RUN_ROOT}" >> "${CONTROL_DIR}/manifest.tsv"

set +e
GPU="${GPU}" RUN_ID="${RUN_ID}" OUTPUT_BASE="${RESULT_BASE}" \
  EXPECTED_SHA256="${EXPECTED_SHA256}" PYTHON="${PYTHON}" \
  FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" LOG_DIR="${LOG_DIR}" \
  bash scripts/emotic/run_multilane_track_a_face_expression_residual_val.sh
code=$?
set -e
printf '%s\n' "${code}" > "${CONTROL_DIR}/status/hybrid_residual.exit_code"
[[ "${code}" -eq 0 ]] || { printf 'failed\n' > "${CONTROL_DIR}/batch_status.txt"; exit "${code}"; }

"${PYTHON}" -m multi_lane.track_a.compare_face_expression_validation \
  --full-run "${FULL_RUN}" --person-run "${PERSON_RUN}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" --anchor "clip_face=${ANCHOR_FACE}" \
  --candidate "clip_expression_residual=${RUN_ROOT}" \
  --output "${CONTROL_DIR}/validation_summary.json" 2>&1 | tee "${LOG_DIR}/summary.log"
printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "FACE_EXPRESSION_RESIDUAL_BATCH_COMPLETE batch=${BATCH_ID} result=${RESULT_BASE}"
