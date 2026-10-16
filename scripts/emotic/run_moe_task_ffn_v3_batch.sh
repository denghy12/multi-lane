#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BATCH_ID="${BATCH_ID:-seed0_20261016_01}"
for mode in dynamic shared_fixed view_fixed; do
  echo "START_${mode}"
  ROUTE_MODE="${mode}" RUN_ID="formal_${mode}_${BATCH_ID}" \
    bash scripts/emotic/run_moe_task_ffn_v3.sh
  echo "DONE_${mode}"
done
echo "MOE_TASK_FFN_V3_BATCH_COMPLETE"
