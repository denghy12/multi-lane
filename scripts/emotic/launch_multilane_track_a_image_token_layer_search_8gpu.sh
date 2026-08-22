#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPUS_TEXT="${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_LIST <<< "${GPUS_TEXT}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-8000}"
GPU_MAX_UTIL_PERCENT="${GPU_MAX_UTIL_PERCENT:-10}"
GPU_READY_CHECKS="${GPU_READY_CHECKS:-2}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"

[[ "${#GPU_LIST[@]}" -eq 8 ]] || { echo "Layer search requires exactly 8 GPUs" >&2; exit 2; }
for gpu in "${GPU_LIST[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "GPU identifiers must be nonnegative integers" >&2; exit 2; }
done
[[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l | tr -d ' ')" -eq 8 ]] || {
  echo "Layer search requires 8 distinct GPUs" >&2
  exit 2
}
[[ "${GPU_READY_CHECKS}" =~ ^[1-9][0-9]*$ ]] || { echo "GPU_READY_CHECKS must be positive" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Layer search requires a clean Git worktree" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }

BATCH_ID="${BATCH_ID:-image_token_layer_search_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT}/output/emotic_image_token_tuning/layer_search/${BATCH_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_image_token_tuning/layer_search}"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"
MANIFEST="${OUTPUT_BASE}/search_manifest.tsv"
STATUS_DIR="${OUTPUT_BASE}/launcher_status"
mkdir -p "${STATUS_DIR}"

labels=(disabled_bce)
adapter_modes=(disabled)
routings=(joint_bce)
layers=(8)
for layer in $(seq 0 11); do
  labels+=("layer${layer}_bce" "layer${layer}_asl")
  adapter_modes+=(image_token image_token)
  routings+=(joint_bce adapter_asl)
  layers+=("${layer}" "${layer}")
done

printf 'index\tlabel\tadapter_mode\tloss_routing\tlayer\tbottleneck\tadapter_lr\tresidual_scale\tactivation\ttask_init\tprecision\tgpu\n' > "${MANIFEST}"
for index in "${!labels[@]}"; do
  gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
  printf '%s\t%s\t%s\t%s\t%s\t32\t0.0004\t0.1\trelu\tindependent\tfp32\t%s\n' \
    "${index}" "${labels[$index]}" "${adapter_modes[$index]}" \
    "${routings[$index]}" "${layers[$index]}" "${gpu}" >> "${MANIFEST}"
done

gpu_snapshot() {
  local gpu="$1"
  nvidia-smi -i "${gpu}" \
    --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits
}

wait_for_gpu_ready() {
  local gpu="$1"
  local purpose="$2"
  local consecutive=0
  local snapshot free_mib utilization
  while (( consecutive < GPU_READY_CHECKS )); do
    snapshot="$(gpu_snapshot "${gpu}")"
    IFS=',' read -r free_mib utilization <<< "${snapshot}"
    free_mib="${free_mib// /}"
    utilization="${utilization// /}"
    if (( free_mib >= GPU_MIN_FREE_MIB && utilization <= GPU_MAX_UTIL_PERCENT )); then
      consecutive=$((consecutive + 1))
      echo "GPU ${gpu} ready ${consecutive}/${GPU_READY_CHECKS} for ${purpose}: free_mib=${free_mib} utilization=${utilization}%"
    else
      consecutive=0
      echo "GPU ${gpu} waiting for ${purpose}: free_mib=${free_mib} utilization=${utilization}%"
    fi
    if (( consecutive < GPU_READY_CHECKS )); then
      sleep "${GPU_WAIT_SECONDS}"
    fi
  done
}

wait_for_any_gpu_ready() {
  local -a consecutive=()
  local index gpu snapshot free_mib utilization
  for index in "${!GPU_LIST[@]}"; do
    consecutive[$index]=0
  done
  while true; do
    for index in "${!GPU_LIST[@]}"; do
      gpu="${GPU_LIST[$index]}"
      snapshot="$(gpu_snapshot "${gpu}")"
      IFS=',' read -r free_mib utilization <<< "${snapshot}"
      free_mib="${free_mib// /}"
      utilization="${utilization// /}"
      if (( free_mib >= GPU_MIN_FREE_MIB && utilization <= GPU_MAX_UTIL_PERCENT )); then
        consecutive[$index]=$((consecutive[$index] + 1))
        echo "GPU ${gpu} ready ${consecutive[$index]}/${GPU_READY_CHECKS} for global smoke: free_mib=${free_mib} utilization=${utilization}%" >&2
      else
        consecutive[$index]=0
        echo "GPU ${gpu} waiting for global smoke: free_mib=${free_mib} utilization=${utilization}%" >&2
      fi
      if (( consecutive[$index] >= GPU_READY_CHECKS )); then
        printf '%s\n' "${gpu}"
        return 0
      fi
    done
    sleep "${GPU_WAIT_SECONDS}"
  done
}

SMOKE_GPU="$(wait_for_any_gpu_ready)"
CUDA_VISIBLE_DEVICES="${SMOKE_GPU}" "${PYTHON}" -m multi_lane.track_a.smoke \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --adapter-mode image_token \
  --adapter-bottleneck-dim 32 \
  --adapter-layer-indices 8 \
  --adapter-task-init independent \
  --loss-routing adapter_asl \
  --no-amp
echo "IMAGE_TOKEN_LAYER_SEARCH_FP32_SMOKE_COMPLETE gpu=${SMOKE_GPU}"

run_gpu_lane() {
  local lane_index="$1"
  local gpu="${GPU_LIST[$lane_index]}"
  local job_index label run_id lane_failed=0
  for ((job_index=lane_index; job_index<${#labels[@]}; job_index+=${#GPU_LIST[@]})); do
    label="${labels[$job_index]}"
    run_id="${BATCH_ID}_${label}"
    wait_for_gpu_ready "${gpu}" "${label}"
    echo "Starting ${label} on physical GPU ${gpu} in FP32"
    if SEED=0 GPU="${gpu}" ADAPTER_MODE="${adapter_modes[$job_index]}" \
      LOSS_ROUTING="${routings[$job_index]}" ADAPTER_LAYERS="${layers[$job_index]}" \
      BOTTLENECK=32 ADAPTER_LR=0.0004 ADAPTER_SCALE=0.1 \
      ADAPTER_ACTIVATION=relu ADAPTER_TASK_INIT=independent \
      ASL_GAMMA_NEG=9.8 ASL_GAMMA_POS=0.0 ASL_CLIP=0.05 \
      NO_AMP=1 PYTHON="${PYTHON}" CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
      DATA_ROOT="${DATA_ROOT}" RUN_ID="${run_id}" \
      OUTPUT_BASE="${OUTPUT_BASE}" LOG_DIR="${LOG_DIR}" \
      bash scripts/emotic/run_multilane_track_a_image_token_hparam_val.sh; then
      printf '0\n' > "${STATUS_DIR}/${label}.exit_code"
    else
      printf '1\n' > "${STATUS_DIR}/${label}.exit_code"
      lane_failed=1
    fi
  done
  return "${lane_failed}"
}

pids=()
for lane_index in "${!GPU_LIST[@]}"; do
  run_gpu_lane "${lane_index}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "At least one layer-search run failed; completed runs were preserved" >&2
  exit 1
fi

summary_arguments=()
for label in "${labels[@]}"; do
  summary_arguments+=(--run "${label}=${OUTPUT_BASE}/${BATCH_ID}_${label}")
done
"${PYTHON}" -m multi_lane.track_a.summarize_image_token_layer_search \
  "${summary_arguments[@]}" \
  --output "${OUTPUT_BASE}/layer_search_summary.json"
echo "MULTI_LANE_IMAGE_TOKEN_LAYER_SEARCH_COMPLETE ${OUTPUT_BASE}"
