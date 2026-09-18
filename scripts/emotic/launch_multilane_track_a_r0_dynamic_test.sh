#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
GPU_R0="${GPU_R0:-0}"
GPU_D1="${GPU_D1:-1}"
BATCH_ID="${BATCH_ID:-r0_dynamic_fusion_test_$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-${ROOT}/output/emotic_track_a_r0_dynamic_test/${BATCH_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_r0_dynamic_test/${BATCH_ID}}"
CONTROL_DIR="${CONTROL_DIR:-${ROOT}/output/emotic_track_a_r0_dynamic_test/${BATCH_ID}/control}"

[[ "${GPU_R0}" =~ ^[0-9]+$ && "${GPU_D1}" =~ ^[0-9]+$ ]] || {
  echo "GPU_R0 and GPU_D1 must be numeric" >&2; exit 2;
}
[[ "${GPU_R0}" != "${GPU_D1}" ]] || { echo "GPUs must be distinct" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Launcher requires a clean Git worktree" >&2; exit 2; }
[[ ! -e "${RESULT_BASE}" ]] || { echo "Result directory already exists" >&2; exit 2; }

mkdir -p "${RESULT_BASE}" "${LOG_DIR}" "${CONTROL_DIR}/status"
cat > "${CONTROL_DIR}/protocol.txt" <<EOF
Exploratory held-out test comparison requested after shared-base R0 validation.
R0: fixed reliability-masked Full/Person/Face feature fusion.
D1: task-local hidden-16 sample-level masked-softmax Full/Person/Face Router,
    initialized at the same fixed prior and trained jointly from the train split.
Both: current shared-base commit, seed0, FP32 with TF32 enabled, 8 tasks,
30 epochs/task, batch64, Adam reset/cosine, frozen CLIP, no checkpoints.
Test is evaluated once with no test-side tuning or weight search.
EOF

pids=()
methods=(R0 D1)
gpus=("${GPU_R0}" "${GPU_D1}")
for i in 0 1; do
  method="${methods[$i]}"
  run_id="${BATCH_ID}_${method}_seed0_test"
  GPU="${gpus[$i]}" METHOD="${method}" RUN_ID="${run_id}" \
    OUTPUT_BASE="${RESULT_BASE}" LOG_DIR="${LOG_DIR}" PYTHON="${PYTHON}" \
    bash scripts/emotic/run_multilane_track_a_r0_dynamic_test.sh \
    > "${LOG_DIR}/${method}.launcher.log" 2>&1 &
  pids+=("$!")
done

failed=0
for i in 0 1; do
  if wait "${pids[$i]}"; then rc=0; else rc=$?; failed=1; fi
  printf '%s\n' "${rc}" > "${CONTROL_DIR}/status/${methods[$i]}.exit_code"
done
if (( failed )); then
  printf 'failed\n' > "${CONTROL_DIR}/batch_status.txt"
  exit 1
fi

printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "R0_DYNAMIC_FUSION_TEST_BATCH_COMPLETE batch=${BATCH_ID} output=${RESULT_BASE}"
