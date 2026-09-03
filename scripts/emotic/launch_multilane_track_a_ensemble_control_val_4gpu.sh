#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"
[[ ${#GPU_LIST[@]} -eq 4 ]] || { echo "Four GPUs required" >&2; exit 2; }
[[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l)" -eq 4 ]] || { echo "GPUs must be distinct" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean worktree required" >&2; exit 2; }
BATCH_ID="${BATCH_ID:-fixed_ensemble_control_val_seed12_$(date +%Y%m%d_%H%M%S)}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_ensemble_control/${BATCH_ID}"
OUTPUT_BASE="${CONTROL_DIR}/runs"
LOG_DIR="${ROOT}/logs/emotic_track_a_ensemble_control/${BATCH_ID}"
SEED0_ID=image_token_layer1_full_person_letterbox_val_seed0_20260903_134720
SEED0_BASE="/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/${SEED0_ID}"
FULL0="${SEED0_BASE}/${SEED0_ID}_full_anchor"
PERSON0="${SEED0_BASE}/${SEED0_ID}_person_letterbox"
[[ ! -e "${CONTROL_DIR}" ]] || { echo "Batch already exists; refusing overwrite" >&2; exit 2; }
"${PYTHON}" -m multi_lane.track_a.compare_validation_ensembles --audit-only --full-runs "${FULL0}" --person-runs "${PERSON0}"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}" "${CONTROL_DIR}/status"
printf 'seed\tview\tgpu\trun_id\n' > "${CONTROL_DIR}/manifest.tsv"
index=0
for seed in 1 2; do
  for view in full person; do
    printf '%s\t%s\t%s\t%s_seed%s_%s\n' "$seed" "$view" "${GPU_LIST[$index]}" "$BATCH_ID" "$seed" "$view" >> "${CONTROL_DIR}/manifest.tsv"
    index=$((index + 1))
  done
done
consecutive=0
while (( consecutive < 2 )); do
  ready=1
  for gpu in "${GPU_LIST[@]}"; do
    free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    utilization="$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    echo "GPU $gpu free_mib=$free_mib utilization=$utilization"
    if (( free_mib < 18000 || utilization > 10 )); then ready=0; fi
  done
  if (( ready )); then consecutive=$((consecutive + 1)); else consecutive=0; fi
  if (( consecutive < 2 )); then sleep 30; fi
done
run_view() {
  local seed="$1" view="$2" gpu="$3" rc=0
  SEED="$seed" VIEW="$view" GPU="$gpu" RUN_ID="${BATCH_ID}_seed${seed}_${view}" \
    OUTPUT_BASE="$OUTPUT_BASE" LOG_DIR="$LOG_DIR" PYTHON="$PYTHON" \
    bash scripts/emotic/run_multilane_track_a_dual_view_val.sh || rc=$?
  printf '%s\n' "$rc" > "${CONTROL_DIR}/status/seed${seed}_${view}.exit_code"
  return "$rc"
}
pids=()
index=0
for seed in 1 2; do
  for view in full person; do
    run_view "$seed" "$view" "${GPU_LIST[$index]}" &
    pids+=("$!")
    index=$((index + 1))
  done
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then echo "Training failed; outputs preserved" >&2; exit 1; fi
"${PYTHON}" -m multi_lane.track_a.compare_validation_ensembles \
  --full-runs "$FULL0" "${OUTPUT_BASE}/${BATCH_ID}_seed1_full" "${OUTPUT_BASE}/${BATCH_ID}_seed2_full" \
  --person-runs "$PERSON0" "${OUTPUT_BASE}/${BATCH_ID}_seed1_person" "${OUTPUT_BASE}/${BATCH_ID}_seed2_person" \
  --output "${CONTROL_DIR}/ensemble_comparison.json"
printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "ENSEMBLE_CONTROL_VALIDATION_COMPLETE batch=${BATCH_ID}"
