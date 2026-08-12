#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

: "${TRAINING_LOSS_MODE:?Set TRAINING_LOSS_MODE to the selected legacy_full_zero or current_only loss}"
[[ "${TRAINING_LOSS_MODE}" == "legacy_full_zero" || "${TRAINING_LOSS_MODE}" == "current_only" ]] || { echo "Invalid selected training loss mode" >&2; exit 2; }

GPUS_TEXT="${GPUS:-1 2 3 4}"
read -r -a GPU_LIST <<< "${GPUS_TEXT}"
NORMALIZATIONS=(none clip none clip)
CROP_MINS=(0.05 0.05 0.50 0.50)
[[ "${#GPU_LIST[@]}" -eq "${#NORMALIZATIONS[@]}" ]] || { echo "GPUS must contain exactly four device ids" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Preprocessing diagnostics require a clean Git worktree" >&2; exit 2; }

BATCH_ID="${BATCH_ID:-preprocessing_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_preprocessing_diagnostics_v0.3/${BATCH_ID}}"
mkdir -p "${OUTPUT_BASE}"

pids=()
run_arguments=()
for index in "${!NORMALIZATIONS[@]}"; do
  normalization="${NORMALIZATIONS[$index]}"
  crop_min="${CROP_MINS[$index]}"
  gpu="${GPU_LIST[$index]}"
  run_id="${BATCH_ID}_disabled_${TRAINING_LOSS_MODE}_${normalization}_crop${crop_min}"
  label="${normalization}_crop${crop_min}"
  echo "Starting disabled loss=${TRAINING_LOSS_MODE} normalization=${normalization} crop_min=${crop_min} on physical GPU ${gpu}"
  SEED=0 GPU="${gpu}" METHOD_MODE=disabled \
    TRAINING_LOSS_MODE="${TRAINING_LOSS_MODE}" \
    INPUT_NORMALIZATION="${normalization}" TRAIN_CROP_MIN="${crop_min}" \
    RUN_ID="${run_id}" OUTPUT_BASE="${OUTPUT_BASE}" \
    bash scripts/emotic/run_multilane_track_a_adapter_diagnostic.sh &
  pids+=("$!")
  run_arguments+=(--run "${label}=${OUTPUT_BASE}/${run_id}")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
[[ "${failed}" == "0" ]] || { echo "At least one preprocessing diagnostic failed" >&2; exit 1; }
"${PYTHON:-/opt/conda/envs/ddp/bin/python}" -m multi_lane.track_a.summarize_validation_diagnostics \
  "${run_arguments[@]}" \
  --vary-field input_normalization \
  --vary-field input_normalization_mean \
  --vary-field input_normalization_std \
  --vary-field train_crop_scale \
  --output "${OUTPUT_BASE}/preprocessing_diagnostics_summary.json"
echo "MULTI_LANE_PREPROCESSING_DIAGNOSTICS_COMPLETE ${OUTPUT_BASE}"
