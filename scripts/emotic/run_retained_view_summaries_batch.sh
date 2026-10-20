#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
RUN_ID="${RUN_ID:-pilot_seed0_20261020_01}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
LOG="logs/retained_view_summaries/${RUN_ID}"
[[ ! -e "$LOG" ]] || exit 2
mkdir -p "$LOG"
trap 'status=$?; echo "$status" > "$LOG/batch_exit_status.txt"' EXIT
lane5() {
 for arm in baseline_noaux fixed; do
  GPU=5 ARM="$arm" RUN_ID="$RUN_ID" bash scripts/emotic/run_retained_view_summaries.sh
 done
}
lane6() {
 GPU=6 ARM=retained RUN_ID="$RUN_ID" bash scripts/emotic/run_retained_view_summaries.sh
}
lane5 > "$LOG/lane5.log" 2>&1 & p5=$!
lane6 > "$LOG/lane6.log" 2>&1 & p6=$!
s5=0; s6=0
wait "$p5" || s5=$?
wait "$p6" || s6=$?
echo "GPU5=$s5 GPU6=$s6" > "$LOG/lane_exit_status.txt"
[[ "$s5" == 0 && "$s6" == 0 ]] || exit 1
expected=3
[[ -z "${SMOKE_UPDATES:-}" ]] || expected=2
for arm in baseline_noaux fixed retained; do
 OMP_NUM_THREADS=1 "$PYTHON" -m multi_lane.track_a.verify_joint_summary \
  "output/retained_view_summaries/$RUN_ID/$arm" --expected-tasks "$expected" \
  --baseline "output/retained_view_summaries/$RUN_ID/baseline_noaux/train" \
  > "$LOG/$arm/paired_verification.log" 2>&1
done
