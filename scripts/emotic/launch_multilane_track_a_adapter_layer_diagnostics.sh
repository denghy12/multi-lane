#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

: "${INPUT_NORMALIZATION:?Set INPUT_NORMALIZATION to the selected none or clip preprocessing}"
: "${TRAIN_CROP_MIN:?Set TRAIN_CROP_MIN to the selected crop minimum}"
: "${TRAINING_LOSS_MODE:?Set TRAINING_LOSS_MODE to the selected loss}"
: "${BASELINE_RUN:?Set BASELINE_RUN to the selected disabled preprocessing run root}"
[[ "${TRAINING_LOSS_MODE}" == "legacy_full_zero" || "${TRAINING_LOSS_MODE}" == "current_only" ]] || { echo "Invalid selected training loss mode" >&2; exit 2; }

GPUS_TEXT="${GPUS:-1 2 3}"
read -r -a GPU_LIST <<< "${GPUS_TEXT}"
LAYERS=(5 8 11)
[[ "${#GPU_LIST[@]}" -eq "${#LAYERS[@]}" ]] || { echo "GPUS must contain exactly three device ids" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Layer diagnostics require a clean Git worktree" >&2; exit 2; }

BATCH_ID="${BATCH_ID:-adapter_layer_position_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_adapter_layer_position_v0.3/${BATCH_ID}}"
mkdir -p "${OUTPUT_BASE}"

pids=()
run_roots=()
for index in "${!LAYERS[@]}"; do
  layer="${LAYERS[$index]}"
  gpu="${GPU_LIST[$index]}"
  run_id="${BATCH_ID}_task_lane_independent_layer${layer}"
  echo "Starting zero-based layer ${layer} on physical GPU ${gpu}"
  SEED=0 GPU="${gpu}" BOTTLENECK=64 ADAPTER_LR=0.0004 \
    ADAPTER_TASK_INIT=independent ADAPTER_LAYERS="${layer}" \
    TRAINING_LOSS_MODE="${TRAINING_LOSS_MODE}" \
    INPUT_NORMALIZATION="${INPUT_NORMALIZATION}" \
    TRAIN_CROP_MIN="${TRAIN_CROP_MIN}" \
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
[[ "${failed}" == "0" ]] || { echo "At least one layer diagnostic failed" >&2; exit 1; }
for index in "${!LAYERS[@]}"; do
  layer="${LAYERS[$index]}"
  "${PYTHON:-/opt/conda/envs/ddp/bin/python}" -m multi_lane.track_a.compare_validation \
    --baseline-run "${BASELINE_RUN}" \
    --candidate-run "${run_roots[$index]}" \
    --candidate-task-init independent \
    --output "${OUTPUT_BASE}/layer${layer}_paired_validation.json"
done
echo "MULTI_LANE_ADAPTER_LAYER_DIAGNOSTICS_COMPLETE ${OUTPUT_BASE}"
