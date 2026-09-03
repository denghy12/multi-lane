#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"
[[ ${#GPU_LIST[@]} -eq 4 ]] || { echo "Four GPUs are required" >&2; exit 2; }
[[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l)" -eq 4 ]] || { echo "GPUs must be distinct" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean Git worktree required" >&2; exit 2; }
BATCH_ID="${BATCH_ID:-dual_view_locked_formal_seed12_$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_seed_confirmation_v0.1/${BATCH_ID}}"
LOG_DIR="${ROOT}/logs/emotic_track_a_dual_view_seed_confirmation/${BATCH_ID}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_dual_view_seed_confirmation/${BATCH_ID}"
SEED0_SUMMARY="${SEED0_SUMMARY:-/mnt/haoyuan/workspace/multi-lane-main-dual-view-formal-test/output/emotic_track_a_dual_view_formal_test/image_token_layer1_full_person_letterbox_formal_test_seed0_20260903_145816/fixed_test_fusion_summary.json}"
[[ -f "${SEED0_SUMMARY}" ]] || { echo "Missing completed seed0 summary" >&2; exit 2; }
[[ ! -e "${RESULT_BASE}" ]] || { echo "Batch output already exists; refusing duplicate run" >&2; exit 2; }
mkdir -p "${RESULT_BASE}" "${LOG_DIR}" "${CONTROL_DIR}"

run_seed() {
  local seed="$1" gpu_a="$2" gpu_b="$3"
  if SEED="${seed}" GPUS="${gpu_a} ${gpu_b}" BATCH_ID="${BATCH_ID}_seed${seed}" \
    OUTPUT_BASE="${RESULT_BASE}/seed${seed}" LOG_DIR="${LOG_DIR}" PYTHON="${PYTHON}" \
    bash scripts/emotic/launch_multilane_track_a_dual_view_formal_test_2gpu.sh; then
    printf '0\n' > "${CONTROL_DIR}/seed${seed}.exit_code"
  else
    printf '1\n' > "${CONTROL_DIR}/seed${seed}.exit_code"
    return 1
  fi
}

run_seed 1 "${GPU_LIST[0]}" "${GPU_LIST[1]}" &
seed1_pid=$!
run_seed 2 "${GPU_LIST[2]}" "${GPU_LIST[3]}" &
seed2_pid=$!
failed=0
if ! wait "${seed1_pid}"; then failed=1; fi
if ! wait "${seed2_pid}"; then failed=1; fi
if (( failed )); then echo "At least one seed failed; outputs preserved" >&2; exit 1; fi

"${PYTHON}" -m multi_lane.track_a.summarize_dual_view_seeds \
  --fusion-summaries "${SEED0_SUMMARY}" \
    "${ROOT}/output/emotic_track_a_dual_view_formal_test/${BATCH_ID}_seed1/fixed_test_fusion_summary.json" \
    "${ROOT}/output/emotic_track_a_dual_view_formal_test/${BATCH_ID}_seed2/fixed_test_fusion_summary.json" \
  --output "${CONTROL_DIR}/formal_seed_summary.json"
printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "DUAL_VIEW_SEED12_COMPLETE batch=${BATCH_ID} result=${RESULT_BASE} control=${CONTROL_DIR}"
