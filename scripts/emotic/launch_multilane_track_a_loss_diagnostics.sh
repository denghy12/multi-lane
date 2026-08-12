#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPUS_TEXT="${GPUS:-1 2}"
read -r -a GPU_LIST <<< "${GPUS_TEXT}"
LOSS_MODES=(legacy_full_zero current_only)
[[ "${#GPU_LIST[@]}" -eq "${#LOSS_MODES[@]}" ]] || { echo "GPUS must contain exactly two device ids" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Loss diagnostics require a clean Git worktree" >&2; exit 2; }

BATCH_ID="${BATCH_ID:-training_loss_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_loss_diagnostics_v0.3/${BATCH_ID}}"
mkdir -p "${OUTPUT_BASE}"

pids=()
run_roots=()
for index in "${!LOSS_MODES[@]}"; do
  loss_mode="${LOSS_MODES[$index]}"
  gpu="${GPU_LIST[$index]}"
  run_id="${BATCH_ID}_disabled_${loss_mode}"
  echo "Starting disabled loss=${loss_mode} on physical GPU ${gpu}"
  SEED=0 GPU="${gpu}" METHOD_MODE=disabled \
    TRAINING_LOSS_MODE="${loss_mode}" INPUT_NORMALIZATION=none TRAIN_CROP_MIN=0.05 \
    RUN_ID="${run_id}" OUTPUT_BASE="${OUTPUT_BASE}" \
    bash scripts/emotic/run_multilane_track_a_adapter_diagnostic.sh &
  pids+=("$!")
  run_roots+=("${OUTPUT_BASE}/${run_id}")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
[[ "${failed}" == "0" ]] || { echo "At least one loss diagnostic failed" >&2; exit 1; }
"${PYTHON:-/opt/conda/envs/ddp/bin/python}" -m multi_lane.track_a.summarize_validation_diagnostics \
  --run "legacy=${run_roots[0]}" \
  --run "current_only=${run_roots[1]}" \
  --vary-field training_loss_mode \
  --vary-field training_loss_reduction_classes \
  --vary-field training_loss_current_only_gradient_multiplier_vs_legacy \
  --output "${OUTPUT_BASE}/loss_diagnostics_summary.json"
echo "MULTI_LANE_LOSS_DIAGNOSTICS_COMPLETE ${OUTPUT_BASE}"
