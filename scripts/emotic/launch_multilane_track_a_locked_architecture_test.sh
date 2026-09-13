#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
C2_SELECTION="${C2_SELECTION:-/mnt/haoyuan/workspace/multi-lane-main-class-aware-oof-stacking/output/emotic_track_a_class_aware_oof/class_aware_oof_seed0_20260911_002/validation_selection.json}"
C2_SELECTION_SHA256="${C2_SELECTION_SHA256:-3fdab45e10dd68975c51b2f318d767af0392e661d7b10d281b65db1f9d3edae0}"
TEST_DESCRIPTORS="${TEST_DESCRIPTORS:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_taskwise_three_view_router_v0.1/taskwise_three_view_router_20260909_233435/test_descriptors}"
R1_SUMMARY="${R1_SUMMARY:-/mnt/haoyuan/workspace/multi-lane-main-fixed-three-view-seed012-test/output/emotic_track_a_fixed_three_view_test/fixed_three_view_seed012_test_20260909_213052/fixed_three_view_seed012_test_summary.json}"
BATCH_ID="${BATCH_ID:-locked_architecture_test_$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_locked_architecture_test_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_locked_architecture_test/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_locked_architecture_test/${BATCH_ID}"
MIN_FREE_MIB="${MIN_FREE_MIB:-9000}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6}"

[[ ${#GPU_LIST[@]} -eq 7 ]] || { echo "Exactly seven GPU IDs are required" >&2; exit 2; }
[[ $(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l | tr -d ' ') -eq 7 ]] || { echo "GPU IDs must be distinct" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Launcher requires a clean worktree" >&2; exit 2; }
[[ ! -e "${RESULT_BASE}" && ! -e "${CONTROL_DIR}" ]] || { echo "Batch destination exists" >&2; exit 2; }
for path in "${C2_SELECTION}" "${TEST_DESCRIPTORS}/test_descriptors.npz" "${R1_SUMMARY}"; do
  [[ -f "${path}" ]] || { echo "Missing locked input: ${path}" >&2; exit 2; }
done
for gpu in "${GPU_LIST[@]}"; do
  free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  echo "GPU ${gpu} free_mib=${free_mib}"
  (( free_mib >= MIN_FREE_MIB )) || { echo "GPU ${gpu} has less than ${MIN_FREE_MIB} MiB free" >&2; exit 2; }
done
mkdir -p "${RESULT_BASE}" "${CONTROL_DIR}/status" "${LOG_DIR}"
printf '%s\n' \
  'Validation-locked exploratory held-out test comparison; no test tuning.' \
  'Six independent training runs: J0/A2 x seed0/1/2; C2 reuses one locked seed0-OOF stacker.' \
  'All J0/A2 runs: 8 tasks x 30 epochs, AMP/TF32 on, test score dumps on, checkpoints off.' \
  > "${CONTROL_DIR}/protocol.txt"

methods=(J0 J0 J0 A2 A2 A2)
seeds=(0 1 2 0 1 2)
pids=()
j0_runs=()
a2_runs=()
for index in 0 1 2 3 4 5; do
  method="${methods[$index]}"
  seed="${seeds[$index]}"
  run_id="${BATCH_ID}_${method}_seed${seed}_test"
  run_path="${RESULT_BASE}/${run_id}"
  if [[ "${method}" == J0 ]]; then j0_runs+=("${run_path}"); else a2_runs+=("${run_path}"); fi
  GPU="${GPU_LIST[$index]}" SEED="${seed}" METHOD="${method}" RUN_ID="${run_id}" \
    OUTPUT_BASE="${RESULT_BASE}" LOG_DIR="${LOG_DIR}" DATA_ROOT="${DATA_ROOT}" \
    FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" PYTHON="${PYTHON}" \
    bash scripts/emotic/run_multilane_track_a_locked_architecture_test.sh \
    > "${LOG_DIR}/${method}_seed${seed}.launcher.log" 2>&1 &
  pids+=("$!")
done

c2_output="${CONTROL_DIR}/C2_locked_test_summary.json"
CUDA_VISIBLE_DEVICES="${GPU_LIST[6]}" "${PYTHON}" -m multi_lane.track_a.evaluate_class_aware_oof_test \
  --selection "${C2_SELECTION}" \
  --expected-selection-sha256 "${C2_SELECTION_SHA256}" \
  --data-root "${DATA_ROOT}" \
  --test-face-manifest "${FACE_MANIFEST_ROOT}" \
  --test-descriptor-root "${TEST_DESCRIPTORS}" \
  --fixed-test-summary "${R1_SUMMARY}" \
  --output "${c2_output}" \
  --device cuda \
  > "${LOG_DIR}/C2_evaluation.log" 2>&1 &
c2_pid="$!"

failed=0
for index in 0 1 2 3 4 5; do
  if wait "${pids[$index]}"; then rc=0; else rc=$?; failed=1; fi
  printf '%s\n' "${rc}" > "${CONTROL_DIR}/status/${methods[$index]}_seed${seeds[$index]}.exit_code"
done
if wait "${c2_pid}"; then c2_rc=0; else c2_rc=$?; failed=1; fi
printf '%s\n' "${c2_rc}" > "${CONTROL_DIR}/status/C2.exit_code"
if (( failed )); then
  printf 'failed\n' > "${CONTROL_DIR}/batch_status.txt"
  exit 1
fi

"${PYTHON}" -m multi_lane.track_a.summarize_locked_architecture_test \
  --j0-runs "${j0_runs[@]}" \
  --a2-runs "${a2_runs[@]}" \
  --r1-summary "${R1_SUMMARY}" \
  --c2-summary "${c2_output}" \
  --output "${CONTROL_DIR}/locked_architecture_test_summary.json" \
  2>&1 | tee "${LOG_DIR}/summary.log"
printf 'complete_no_test_search\n' > "${CONTROL_DIR}/batch_status.txt"
echo "LOCKED_ARCHITECTURE_TEST_BATCH_COMPLETE batch=${BATCH_ID} summary=${CONTROL_DIR}/locked_architecture_test_summary.json"
