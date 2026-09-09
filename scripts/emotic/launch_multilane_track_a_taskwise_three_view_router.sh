#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
GPU="${GPU:-0}"
BATCH_ID="${BATCH_ID:-taskwise_three_view_router_$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_taskwise_three_view_router_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_taskwise_three_view_router/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_taskwise_three_view_router/${BATCH_ID}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
VALIDATION_FACE_MANIFEST="${VALIDATION_FACE_MANIFEST:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
TEST_FACE_MANIFEST="${TEST_FACE_MANIFEST:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}"
SELECTION="${SELECTION:-/mnt/haoyuan/workspace/multi-lane-main-three-view-router-validation/output/emotic_track_a_three_view_router/three_view_router_seed0_20260909_163655/router_selection_r1_centered/validation_selection.json}"
FIXED_TEST_SUMMARY="${FIXED_TEST_SUMMARY:-/mnt/haoyuan/workspace/multi-lane-main-fixed-three-view-seed012-test/output/emotic_track_a_fixed_three_view_test/fixed_three_view_seed012_test_20260909_213052/fixed_three_view_seed012_test_summary.json}"
DESCRIPTOR_ROOT="${RESULT_BASE}/test_descriptors"
OUTPUT="${CONTROL_DIR}/taskwise_three_view_router_evaluation.json"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"

[[ -z "$(git status --porcelain)" ]] || { echo "Launcher requires a clean worktree" >&2; exit 2; }
[[ ! -e "${RESULT_BASE}" && ! -e "${CONTROL_DIR}" ]] || { echo "Batch destination exists" >&2; exit 2; }
for path in "${DATA_ROOT}/CVPR17_Annotations.mat" "${CLIP_CHECKPOINT}" \
  "${VALIDATION_FACE_MANIFEST}/audit_summary.json" "${TEST_FACE_MANIFEST}/audit_summary.json" \
  "${SELECTION}" "${FIXED_TEST_SUMMARY}"; do
  [[ -f "${path}" ]] || { echo "Missing required input: ${path}" >&2; exit 2; }
done
mkdir -p "${RESULT_BASE}" "${CONTROL_DIR}/status" "${LOG_DIR}"

cat > "${CONTROL_DIR}/protocol.txt" <<EOF
Reuse all saved experts and Router states; no expert or Router training.
Validation selects one existing prior candidate within R2 and R3 after task-lane reassembly.
Router k may alter only task-k classes; old lane weights and normalization remain frozen.
R1/R2/R3 test is an exploratory diagnostic requested by the user; test never selects a candidate.
EOF

consecutive=0
while (( consecutive < 2 )); do
  free_mib="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  utilization="$(nvidia-smi -i "${GPU}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
  echo "GPU ${GPU} free_mib=${free_mib} utilization=${utilization}"
  if (( free_mib >= MIN_FREE_MIB && utilization <= MAX_UTILIZATION )); then
    consecutive=$((consecutive + 1))
  else
    consecutive=0
  fi
  if (( consecutive < 2 )); then sleep 30; fi
done

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.export_three_view_test_descriptors \
  --data-root "${DATA_ROOT}" \
  --face-manifest-root "${TEST_FACE_MANIFEST}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output-dir "${DESCRIPTOR_ROOT}" \
  --device cuda --batch-size 128 --workers 2 \
  2>&1 | tee "${LOG_DIR}/test_descriptors.log"
printf '0\n' > "${CONTROL_DIR}/status/test_descriptors.exit_code"

"${PYTHON}" -m multi_lane.track_a.evaluate_taskwise_three_view_router \
  --selection "${SELECTION}" \
  --data-root "${DATA_ROOT}" \
  --validation-face-manifest "${VALIDATION_FACE_MANIFEST}" \
  --test-face-manifest "${TEST_FACE_MANIFEST}" \
  --test-descriptor-root "${DESCRIPTOR_ROOT}" \
  --fixed-test-summary "${FIXED_TEST_SUMMARY}" \
  --output "${OUTPUT}" \
  2>&1 | tee "${LOG_DIR}/taskwise_evaluation.log"
printf '0\n' > "${CONTROL_DIR}/status/taskwise_evaluation.exit_code"
printf 'complete_validation_selected_test_diagnostic_only\n' > "${CONTROL_DIR}/batch_status.txt"
echo "TASKWISE_THREE_VIEW_ROUTER_BATCH_COMPLETE batch=${BATCH_ID} output=${OUTPUT}"
