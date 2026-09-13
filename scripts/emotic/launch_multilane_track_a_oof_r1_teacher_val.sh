#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
GPU="${GPU:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-2500}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
OOF_ROOT="${OOF_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_three_view_oof_v0.1/three_view_oof_seed0_20260910_160250/oof_sources}"
BATCH_ID="${BATCH_ID:-oof_r1_teacher_seed0_$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_oof_r1_teacher_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_oof_r1_teacher/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_oof_r1_teacher/${BATCH_ID}"
ANCHOR_RUN="${ANCHOR_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_oof_distillation_v0.1/oof_distillation_seed0_20260913_230500/oof_distillation_seed0_20260913_230500_D0}"
PERSON_RUN="${PERSON_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_person_letterbox}"
FACE_RUN="${FACE_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_endpoint_val_v0.1/face_endpoint_val_seed0_20260909_1320/face_endpoint_val_seed0_20260909_1320_face}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"

[[ -z "$(git status --porcelain)" ]] || { echo "Launcher requires a clean worktree" >&2; exit 2; }
[[ ! -e "${RESULT_BASE}" && ! -e "${CONTROL_DIR}" ]] || { echo "Batch destination exists" >&2; exit 2; }
for source in "${DATA_ROOT}" "${CLIP_CHECKPOINT}" "${OOF_ROOT}" "${ANCHOR_RUN}" \
  "${PERSON_RUN}" "${FACE_RUN}" "${FACE_MANIFEST_ROOT}"; do
  [[ -e "${source}" ]] || { echo "Missing source: ${source}" >&2; exit 2; }
done
while true; do
  free_mib="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if (( free_mib >= MIN_FREE_MIB )); then break; fi
  echo "Waiting for GPU${GPU}: free=${free_mib}MiB required=${MIN_FREE_MIB}MiB"
  sleep "${WAIT_SECONDS}"
done
mkdir -p "${RESULT_BASE}" "${CONTROL_DIR}/status" "${LOG_DIR}"
run_id="${BATCH_ID}_D3"
printf 'variant\tgpu\tseed\trun\ttest_access\nD3\t%s\t0\t%s\tforbidden\n' \
  "${GPU}" "${RESULT_BASE}/${run_id}" > "${CONTROL_DIR}/manifest.tsv"

set +e
GPU="${GPU}" VARIANT=D3 RUN_ID="${run_id}" OUTPUT_BASE="${RESULT_BASE}" \
  LOG_DIR="${LOG_DIR}" PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" \
  CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" OOF_ROOT="${OOF_ROOT}" \
  FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" \
  bash scripts/emotic/run_multilane_track_a_oof_distillation_val.sh \
  > "${LOG_DIR}/${run_id}.launcher.log" 2>&1
rc=$?
set -e
printf '%s\n' "${rc}" > "${CONTROL_DIR}/status/D3.exit_code"
(( rc == 0 )) || exit "${rc}"

"${PYTHON}" -m multi_lane.track_a.compare_oof_r1_teacher_validation \
  --anchor "D0=${ANCHOR_RUN}" --candidate "D3=${RESULT_BASE}/${run_id}" \
  --person-run "${PERSON_RUN}" --face-run "${FACE_RUN}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --output "${CONTROL_DIR}/validation_summary.json" \
  2>&1 | tee "${LOG_DIR}/comparison.log"
printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "OOF_R1_TEACHER_BATCH_COMPLETE batch=${BATCH_ID} result=${RESULT_BASE}"
