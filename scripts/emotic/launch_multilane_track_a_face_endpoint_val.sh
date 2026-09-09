#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
FULL_RUN="${FULL_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_full_anchor}"
PERSON_RUN="${PERSON_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_person_letterbox}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
READY_CHECKS="${READY_CHECKS:-2}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
BATCH_ID="${BATCH_ID:-face_endpoint_val_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_endpoint_val_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_face_endpoint_val/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_face_endpoint_val"
FACE_RUN_ID="${BATCH_ID}_face"
FACE_RUN="${OUTPUT_BASE}/${FACE_RUN_ID}"

[[ -z "$(git status --porcelain)" ]] || { echo "Launcher requires a clean Git worktree" >&2; exit 2; }
for path in "${FULL_RUN}" "${PERSON_RUN}"; do
  [[ -f "${path}/seed_summary.json" ]] || { echo "Missing reusable validation run: ${path}" >&2; exit 2; }
done
[[ -f "${FACE_MANIFEST_ROOT}/audit_summary.json" ]] || { echo "Missing Face manifest: ${FACE_MANIFEST_ROOT}" >&2; exit 2; }
mkdir -p "${OUTPUT_BASE}" "${CONTROL_DIR}" "${LOG_DIR}"

selected_gpu=""
last_candidate=""
consecutive=0
while [[ -z "${selected_gpu}" ]]; do
  candidate=""
  while IFS=', ' read -r gpu free_mib utilization; do
    if [[ "${free_mib}" -ge "${MIN_FREE_MIB}" && "${utilization}" -le "${MAX_UTILIZATION}" ]]; then
      candidate="${gpu}"
      break
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
  if [[ -n "${candidate}" ]]; then
    if [[ "${candidate}" == "${last_candidate}" ]]; then consecutive=$((consecutive + 1)); else consecutive=1; fi
    last_candidate="${candidate}"
    if [[ "${consecutive}" -ge "${READY_CHECKS}" ]]; then selected_gpu="${candidate}"; break; fi
  else
    consecutive=0
    last_candidate=""
  fi
  echo "Waiting for Face endpoint GPU: candidate=${candidate:-none} consecutive=${consecutive}/${READY_CHECKS}"
  sleep "${WAIT_SECONDS}"
done

printf 'batch_id\tface_gpu\tfull_run\tperson_run\tface_run\tbeta_grid\ttest_access\n' > "${CONTROL_DIR}/manifest.tsv"
printf '%s\t%s\t%s\t%s\t%s\t0,0.05,0.10,0.20\tforbidden\n' "${BATCH_ID}" "${selected_gpu}" "${FULL_RUN}" "${PERSON_RUN}" "${FACE_RUN}" >> "${CONTROL_DIR}/manifest.tsv"

GPU="${selected_gpu}" RUN_ID="${FACE_RUN_ID}" OUTPUT_BASE="${OUTPUT_BASE}" \
  PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
  FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" LOG_DIR="${LOG_DIR}" \
  bash scripts/emotic/run_multilane_track_a_face_endpoint_val.sh

"${PYTHON}" -m multi_lane.track_a.fuse_face_endpoint_validation \
  --full-run "${FULL_RUN}" \
  --person-run "${PERSON_RUN}" \
  --face-run "${FACE_RUN}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --output "${CONTROL_DIR}/fusion_summary.json"
printf '0\n' > "${CONTROL_DIR}/launcher.exit_code"
printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "FACE_ENDPOINT_BATCH_COMPLETE batch=${BATCH_ID} face_run=${FACE_RUN} fusion=${CONTROL_DIR}/fusion_summary.json"
