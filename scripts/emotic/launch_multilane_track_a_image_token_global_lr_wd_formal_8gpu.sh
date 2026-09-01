#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPUS_TEXT="${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_LIST <<< "${GPUS_TEXT}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-12000}"
GPU_READY_CHECKS="${GPU_READY_CHECKS:-2}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-30}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"

[[ "${#GPU_LIST[@]}" -eq 8 ]] || { echo "Global LR/WD search requires exactly eight GPUs" >&2; exit 2; }
[[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l | tr -d ' ')" -eq 8 ]] || {
  echo "Global LR/WD search requires eight distinct GPUs" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || { echo "Global LR/WD search requires a clean Git worktree" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }

BATCH_ID="${BATCH_ID:-image_token_asl_layer1_global_lr_wd_formal_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_image_token_asl_layer1_global_lr_wd_formal_v0.1/${BATCH_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_image_token_global_lr_wd_formal}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_image_token_global_lr_wd_formal/${BATCH_ID}"
STATUS_DIR="${CONTROL_DIR}/launcher_status"
MANIFEST="${CONTROL_DIR}/search_manifest.tsv"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}" "${STATUS_DIR}"

# Source LR is multiplied by train_batch_size/reference_batch_size = 64/256.
labels=(
  lr0125_wd0_anchor
  lr0140_wd3e6
  lr0140_wd1e5
  lr0145_wd3e6
  lr0145_wd1e5
  lr0150_wd3e6
  lr0150_wd1e5
  lr0155_wd3e6
  lr0155_wd1e5
)
source_lrs=(0.05 0.056 0.056 0.058 0.058 0.06 0.06 0.062 0.062)
actual_lrs=(0.0125 0.014 0.014 0.0145 0.0145 0.015 0.015 0.0155 0.0155)
adapter_weight_decays=(0 0.000003 0.00001 0.000003 0.00001 0.000003 0.00001 0.000003 0.00001)
gpu_slots=(0 1 2 3 4 5 6 7 0)

for values_name in source_lrs actual_lrs adapter_weight_decays gpu_slots; do
  declare -n values_ref="${values_name}"
  [[ "${#values_ref[@]}" -eq "${#labels[@]}" ]] || {
    echo "Internal global LR/WD array length mismatch: ${values_name}" >&2
    exit 2
  }
done

printf 'index\tlabel\tsource_lr\tactual_main_lr\tadapter_lr\tadapter_weight_decay\tgpu\treporting\n' > "${MANIFEST}"
for index in "${!labels[@]}"; do
  gpu="${GPU_LIST[${gpu_slots[$index]}]}"
  printf '%s\t%s\t%s\t%s\t0.0004\t%s\t%s\ttest\n' \
    "${index}" "${labels[$index]}" "${source_lrs[$index]}" \
    "${actual_lrs[$index]}" "${adapter_weight_decays[$index]}" \
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
      echo "Waiting for all GPUs before atomic global LR/WD launch"
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
    BOTTLENECK_DIMS="32 32 32 32 32 32 32 32" \
    SOURCE_LR="${source_lrs[$index]}" \
    ADAPTER_WEIGHT_DECAY="${adapter_weight_decays[$index]}" \
    PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" \
    CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" LOG_DIR="${LOG_DIR}" \
    bash scripts/emotic/run_multilane_track_a_image_token_stage34_formal.sh; then
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
  echo "At least one global LR/WD run failed; completed results were preserved" >&2
  exit 1
fi

printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "MULTI_LANE_IMAGE_TOKEN_GLOBAL_LR_WD_COMPLETE batch_id=${BATCH_ID} output=${OUTPUT_BASE} manifest=${MANIFEST}"
