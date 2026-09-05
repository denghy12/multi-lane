#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_GROUP="${RUN_GROUP:?RUN_GROUP is required}"
REPORTING_SPLIT="${REPORTING_SPLIT:-test}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a gpus <<< "${GPU_LIST}"
[[ ${#gpus[@]} -eq 4 ]] || { echo "GPU_LIST must contain exactly four GPUs" >&2; exit 2; }

modes=(disabled disabled person_patches person_patches)
crops=(legacy target_aware legacy target_aware)
names=(legacy_disabled target_aware_disabled legacy_person_patches target_aware_person_patches)
pids=()

cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for index in 0 1 2 3; do
  GPU="${gpus[$index]}" MODE="${modes[$index]}" \
    FULL_CROP_MODE="${crops[$index]}" SEED=0 \
    RUN_ID="${RUN_GROUP}_${names[$index]}_seed0_${REPORTING_SPLIT}" \
    REPORTING_SPLIT="${REPORTING_SPLIT}" \
    bash scripts/emotic/run_multilane_track_a_person_conditioned_val.sh &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ ${status} -eq 0 ]] || { echo "SELECTOR_PATCH_2X2_FAILED" >&2; exit 1; }
echo "SELECTOR_PATCH_2X2_COMPLETE group=${RUN_GROUP} split=${REPORTING_SPLIT}"
