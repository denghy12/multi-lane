#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export RUN_ID="${RUN_ID:-formal_seed0_20261020_01}"
LOG_ROOT="logs/joint_view_summary/${RUN_ID}"
[[ ! -e "${LOG_ROOT}" && ! -e "output/joint_view_summary/${RUN_ID}" ]] || { echo "Batch already exists" >&2; exit 2; }
mkdir -p "${LOG_ROOT}"
trap 'batch_exit=$?; echo "${batch_exit}" > "${LOG_ROOT}/batch_exit_status.txt"' EXIT
run_lane() {
  local gpu="$1"; shift
  local arm
  for arm in "$@"; do
    GPU="${gpu}" ARM="${arm}" bash scripts/emotic/run_joint_view_summary.sh
  done
}
run_lane 5 baseline_noaux dense > "${LOG_ROOT}/gpu5.log" 2>&1 &
lane5=$!
run_lane 6 fixed moe > "${LOG_ROOT}/gpu6.log" 2>&1 &
lane6=$!
status5=0; status6=0
wait "${lane5}" || status5=$?
wait "${lane6}" || status6=$?
echo "gpu5_exit=${status5} gpu6_exit=${status6}" > "${LOG_ROOT}/lane_status.txt"
[[ "${status5}" == 0 && "${status6}" == 0 ]] || exit 1
echo "JOINT_SUMMARY_BATCH_COMPLETE ${RUN_ID}"
