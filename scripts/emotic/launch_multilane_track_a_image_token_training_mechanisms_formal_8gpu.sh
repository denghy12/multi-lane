#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPUS_TEXT="${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_LIST <<< "${GPUS_TEXT}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-18000}"
GPU_READY_CHECKS="${GPU_READY_CHECKS:-2}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-30}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"

[[ "${#GPU_LIST[@]}" -eq 8 ]] || { echo "Training-mechanism search requires exactly eight GPUs" >&2; exit 2; }
[[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l | tr -d ' ')" -eq 8 ]] || {
  echo "Training-mechanism search requires eight distinct GPUs" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || { echo "Training-mechanism search requires a clean Git worktree" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }

BATCH_ID="${BATCH_ID:-image_token_asl_layer1_training_mechanisms_formal_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_image_token_training_mechanisms_formal_v0.1/${BATCH_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_image_token_training_mechanisms_formal}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_image_token_training_mechanisms_formal/${BATCH_ID}"
STATUS_DIR="${CONTROL_DIR}/launcher_status"
MANIFEST="${CONTROL_DIR}/search_manifest.tsv"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}" "${STATUS_DIR}"

labels=(
  control_epochs30
  updates0900 updates1200 updates1500 updates1800 updates2100 updates2400 updates2700
  reg_residual_p01 reg_residual_p03 reg_residual_p10
  reg_cosine_p01 reg_cosine_p03 reg_cosine_p10
  gate_init0025 gate_init0050 gate_init0100 gate_init0200
)
stages=(
  stage1
  stage1 stage1 stage1 stage1 stage1 stage1 stage1
  stage2 stage2 stage2
  stage2 stage2 stage2
  stage3 stage3 stage3 stage3
)
budget_modes=(
  epochs
  updates updates updates updates updates updates updates
  epochs epochs epochs
  epochs epochs epochs
  epochs epochs epochs epochs
)
updates=(
  0
  900 1200 1500 1800 2100 2400 2700
  0 0 0
  0 0 0
  0 0 0 0
)
regularizations=(
  none
  none none none none none none none
  residual_ratio residual_ratio residual_ratio
  feature_cosine feature_cosine feature_cosine
  none none none none
)
fractions=(
  0
  0 0 0 0 0 0 0
  0.01 0.03 0.10
  0.01 0.03 0.10
  0 0 0 0
)
gate_modes=(
  fixed
  fixed fixed fixed fixed fixed fixed fixed
  fixed fixed fixed
  fixed fixed fixed
  learnable learnable learnable learnable
)
gate_initial_scales=(
  0.1
  0.1 0.1 0.1 0.1 0.1 0.1 0.1
  0.1 0.1 0.1
  0.1 0.1 0.1
  0.025 0.05 0.1 0.2
)
gpu_slots=(0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1)

for values_name in stages budget_modes updates regularizations fractions gate_modes gate_initial_scales gpu_slots; do
  declare -n values_ref="${values_name}"
  [[ "${#values_ref[@]}" -eq "${#labels[@]}" ]] || {
    echo "Internal training-mechanism array length mismatch: ${values_name}" >&2
    exit 2
  }
done

printf 'index\tlabel\tstage\tbudget_mode\tupdates_per_task\tregularization\tregularization_fraction\tgate_mode\tgate_initial_scale\tgpu\treporting\n' > "${MANIFEST}"
for index in "${!labels[@]}"; do
  gpu="${GPU_LIST[${gpu_slots[$index]}]}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\ttest\n' \
    "${index}" "${labels[$index]}" "${stages[$index]}" \
    "${budget_modes[$index]}" "${updates[$index]}" \
    "${regularizations[$index]}" "${fractions[$index]}" \
    "${gate_modes[$index]}" "${gate_initial_scales[$index]}" \
    "${gpu}" >> "${MANIFEST}"
done

wait_for_all_gpus() {
  local consecutive=0
  while (( consecutive < GPU_READY_CHECKS )); do
    local ready=1
    for gpu in "${GPU_LIST[@]}"; do
      free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
      if (( free_mib < GPU_MIN_FREE_MIB )); then ready=0; fi
      echo "GPU ${gpu}: free_mib=${free_mib} required=${GPU_MIN_FREE_MIB}"
    done
    if (( ready == 1 )); then
      consecutive=$((consecutive + 1))
      echo "All GPUs ready ${consecutive}/${GPU_READY_CHECKS}"
    else
      consecutive=0
      echo "Waiting for all GPUs before atomic 18-run launch"
    fi
    if (( consecutive < GPU_READY_CHECKS )); then sleep "${GPU_WAIT_SECONDS}"; fi
  done
}

wait_for_all_gpus

run_one() {
  local index="$1"
  local label="${labels[$index]}"
  local gpu="${GPU_LIST[${gpu_slots[$index]}]}"
  local run_id="${BATCH_ID}_${label}"
  if GPU="${gpu}" RUN_ID="${run_id}" OUTPUT_BASE="${OUTPUT_BASE}" \
    BUDGET_MODE="${budget_modes[$index]}" \
    UPDATES_PER_TASK="${updates[$index]}" \
    REGULARIZATION="${regularizations[$index]}" \
    REGULARIZATION_FRACTION="${fractions[$index]}" \
    GATE_MODE="${gate_modes[$index]}" \
    GATE_INITIAL_SCALE="${gate_initial_scales[$index]}" \
    PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" \
    CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" LOG_DIR="${LOG_DIR}" \
    bash scripts/emotic/run_multilane_track_a_image_token_training_mechanism_formal.sh; then
    printf '0\n' > "${STATUS_DIR}/${label}.exit_code"
  else
    printf '1\n' > "${STATUS_DIR}/${label}.exit_code"
    return 1
  fi
}

pids=()
for index in "${!labels[@]}"; do
  run_one "${index}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if (( failed != 0 )); then
  echo "At least one training-mechanism run failed; completed results were preserved" >&2
  exit 1
fi

printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "MULTI_LANE_IMAGE_TOKEN_TRAINING_MECHANISMS_COMPLETE batch_id=${BATCH_ID} output=${OUTPUT_BASE} manifest=${MANIFEST}"
