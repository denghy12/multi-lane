#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:?FACE_MANIFEST_ROOT is required}"
VALIDATION_SELECTION="${VALIDATION_SELECTION:-/mnt/haoyuan/workspace/multi-lane-main-three-view-router-validation/output/emotic_track_a_three_view_router/three_view_router_seed0_20260909_163655/router_selection_r1_centered/validation_selection.json}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2}"
[[ ${#GPU_LIST[@]} -eq 3 ]] || { echo "Exactly three GPUs are required" >&2; exit 2; }
[[ "${GPU_LIST[0]}" != "${GPU_LIST[1]}" && "${GPU_LIST[0]}" != "${GPU_LIST[2]}" && "${GPU_LIST[1]}" != "${GPU_LIST[2]}" ]] || { echo "GPUs must be distinct" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Launcher requires a clean worktree" >&2; exit 2; }

BATCH_ID="${BATCH_ID:-fixed_three_view_seed012_test_$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_fixed_three_view_test_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_fixed_three_view_test/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_fixed_three_view_test/${BATCH_ID}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"

FULL_RUNS=(
  /mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_formal_v0.1/image_token_layer1_full_person_letterbox_formal_test_seed0_20260903_145816/image_token_layer1_full_person_letterbox_formal_test_seed0_20260903_145816_full_anchor
  /mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_seed_confirmation_v0.1/dual_view_locked_formal_seed12_20260903_174938/seed1/dual_view_locked_formal_seed12_20260903_174938_seed1_full_anchor
  /mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_seed_confirmation_v0.1/dual_view_locked_formal_seed12_20260903_174938/seed2/dual_view_locked_formal_seed12_20260903_174938_seed2_full_anchor
)
PERSON_RUNS=(
  /mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_formal_v0.1/image_token_layer1_full_person_letterbox_formal_test_seed0_20260903_145816/image_token_layer1_full_person_letterbox_formal_test_seed0_20260903_145816_person_letterbox
  /mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_seed_confirmation_v0.1/dual_view_locked_formal_seed12_20260903_174938/seed1/dual_view_locked_formal_seed12_20260903_174938_seed1_person_letterbox
  /mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_seed_confirmation_v0.1/dual_view_locked_formal_seed12_20260903_174938/seed2/dual_view_locked_formal_seed12_20260903_174938_seed2_person_letterbox
)

[[ ! -e "${RESULT_BASE}" && ! -e "${CONTROL_DIR}" ]] || { echo "Batch destination exists" >&2; exit 2; }
[[ -f "${VALIDATION_SELECTION}" ]] || { echo "Missing validation lock" >&2; exit 2; }
for path in "${FULL_RUNS[@]}" "${PERSON_RUNS[@]}"; do
  [[ -f "${path}/seed_summary.json" ]] || { echo "Missing reusable source: ${path}" >&2; exit 2; }
done
mkdir -p "${RESULT_BASE}/face_sources" "${CONTROL_DIR}/status" "${LOG_DIR}"

cat > "${CONTROL_DIR}/protocol.txt" <<EOF
Held-out test evaluation; exactly one validation-locked rule; no test search.
Reliable Face: [Full,Person,Face]=[0.64,0.16,0.20].
Invalid/unreliable Face: [0.80,0.20,0]. Threshold=0.5.
Face sources: seed0/1/2, full train, 8 tasks x 30 epochs, no checkpoints.
EOF

consecutive=0
while (( consecutive < 2 )); do
  ready=1
  for gpu in "${GPU_LIST[@]}"; do
    free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    echo "GPU ${gpu} free_mib=${free_mib} utilization=${utilization}"
    if (( free_mib < MIN_FREE_MIB || utilization > MAX_UTILIZATION )); then ready=0; fi
  done
  if (( ready )); then consecutive=$((consecutive + 1)); else consecutive=0; fi
  if (( consecutive < 2 )); then sleep 30; fi
done

pids=()
face_runs=()
for seed in 0 1 2; do
  run_id="${BATCH_ID}_seed${seed}_face"
  face_runs+=("${RESULT_BASE}/face_sources/${run_id}")
  GPU="${GPU_LIST[$seed]}" SEED="${seed}" RUN_ID="${run_id}" \
    OUTPUT_BASE="${RESULT_BASE}/face_sources" LOG_DIR="${LOG_DIR}" \
    FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" PYTHON="${PYTHON}" \
    bash scripts/emotic/run_multilane_track_a_face_fixed_test_source.sh &
  pids+=("$!")
done

failed=0
for seed in 0 1 2; do
  if wait "${pids[$seed]}"; then rc=0; else rc=$?; failed=1; fi
  printf '%s\n' "${rc}" > "${CONTROL_DIR}/status/seed${seed}.exit_code"
done
if (( failed )); then
  printf 'face_source_failed\n' > "${CONTROL_DIR}/batch_status.txt"
  exit 1
fi

"${PYTHON}" -m multi_lane.track_a.fuse_fixed_three_view_test \
  --full-runs "${FULL_RUNS[@]}" \
  --person-runs "${PERSON_RUNS[@]}" \
  --face-runs "${face_runs[@]}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --validation-selection "${VALIDATION_SELECTION}" \
  --output "${CONTROL_DIR}/fixed_three_view_seed012_test_summary.json" \
  2>&1 | tee "${LOG_DIR}/fixed_fusion.log"
printf 'complete_no_test_search\n' > "${CONTROL_DIR}/batch_status.txt"
echo "FIXED_THREE_VIEW_TEST_BATCH_COMPLETE batch=${BATCH_ID} summary=${CONTROL_DIR}/fixed_three_view_seed012_test_summary.json"
