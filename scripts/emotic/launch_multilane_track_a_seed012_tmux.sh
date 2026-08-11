#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
SESSION="${SESSION:-emotic_multilane_track_a_seed012}"
RUN_ID="${RUN_ID:-multi_lane_main_track_a_seed012_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT}/logs/emotic_track_a/${RUN_ID}"
mkdir -p "${LOG_DIR}"
tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
}
command="cd $(printf '%q' "${ROOT}") && RUN_ID=$(printf '%q' "${RUN_ID}") bash scripts/emotic/launch_multilane_track_a_seed012.sh 2>&1 | tee $(printf '%q' "${LOG_DIR}/launcher.log"); code=\${PIPESTATUS[0]}; echo MULTI_LANE_LAUNCH_EXIT_CODE=\$code; exec bash"
tmux new-session -d -s "${SESSION}" -n seed012 "${command}"
echo "Started ${SESSION} with run id ${RUN_ID}"
echo "Log: ${LOG_DIR}/launcher.log"
