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

[[ "${#GPU_LIST[@]}" -eq 8 ]] || { echo "Capacity/LR search requires exactly 8 GPUs" >&2; exit 2; }
for gpu in "${GPU_LIST[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "GPU identifiers must be nonnegative integers" >&2; exit 2; }
done
[[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l | tr -d ' ')" -eq 8 ]] || {
  echo "Capacity/LR search requires 8 distinct GPUs" >&2
  exit 2
}
[[ "${GPU_READY_CHECKS}" =~ ^[1-9][0-9]*$ ]] || { echo "GPU_READY_CHECKS must be positive" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Capacity/LR search requires a clean Git worktree" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }

BATCH_ID="${BATCH_ID:-image_token_asl_capacity_lr_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT}/output/emotic_image_token_tuning/asl_capacity_lr/${BATCH_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_image_token_tuning/asl_capacity_lr}"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"
MANIFEST="${OUTPUT_BASE}/search_manifest.tsv"
STATUS_DIR="${OUTPUT_BASE}/launcher_status"
mkdir -p "${STATUS_DIR}"

# The first eight jobs occupy all GPUs. GPUs 0--4 then continue with jobs 8--12.
labels=(
  disabled_bce
  b32_lr4e4_asl
  b8_lr1e4_asl
  b8_lr2e4_asl
  b8_lr4e4_asl
  b16_lr1e4_asl
  b16_lr2e4_asl
  b16_lr4e4_asl
  b32_lr1e4_asl
  b32_lr2e4_asl
  b64_lr1e4_asl
  b64_lr2e4_asl
  b64_lr4e4_asl
)
adapter_modes=(disabled image_token image_token image_token image_token image_token image_token image_token image_token image_token image_token image_token image_token)
routings=(joint_bce adapter_asl adapter_asl adapter_asl adapter_asl adapter_asl adapter_asl adapter_asl adapter_asl adapter_asl adapter_asl adapter_asl adapter_asl)
bottlenecks=(32 32 8 8 8 16 16 16 32 32 64 64 64)
adapter_lrs=(0.0004 0.0004 0.0001 0.0002 0.0004 0.0001 0.0002 0.0004 0.0001 0.0002 0.0001 0.0002 0.0004)
for values_name in adapter_modes routings bottlenecks adapter_lrs; do
  declare -n values_ref="${values_name}"
  [[ "${#values_ref[@]}" -eq "${#labels[@]}" ]] || {
    echo "Internal search array length mismatch: ${values_name}" >&2
    exit 2
  }
done

printf 'index\tlabel\tadapter_mode\tloss_routing\tlayer\tbottleneck\tadapter_lr\tresidual_scale\tactivation\ttask_init\tamp\ttf32\tgpu\n' > "${MANIFEST}"
for index in "${!labels[@]}"; do
  gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
  printf '%s\t%s\t%s\t%s\t8\t%s\t%s\t0.1\trelu\tindependent\toff\toff\t%s\n' \
    "${index}" "${labels[$index]}" "${adapter_modes[$index]}" \
    "${routings[$index]}" "${bottlenecks[$index]}" \
    "${adapter_lrs[$index]}" "${gpu}" >> "${MANIFEST}"
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
    if (( consecutive < GPU_READY_CHECKS )); then sleep "${GPU_WAIT_SECONDS}"; fi
  done
}

wait_for_gpu_ready "${GPU_LIST[1]}" "b64 Adapter-ASL strict-FP32 smoke"
CUDA_VISIBLE_DEVICES="${GPU_LIST[1]}" "${PYTHON}" -m multi_lane.track_a.smoke \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --adapter-mode image_token \
  --adapter-bottleneck-dim 64 \
  --adapter-layer-indices 8 \
  --adapter-task-init independent \
  --loss-routing adapter_asl \
  --no-amp \
  --no-tf32
echo "IMAGE_TOKEN_ASL_CAPACITY_LR_STRICT_FP32_SMOKE_COMPLETE gpu=${GPU_LIST[1]}"

run_gpu_lane() {
  local lane_index="$1"
  local gpu="${GPU_LIST[$lane_index]}"
  local job_index label run_id lane_failed=0
  for ((job_index=lane_index; job_index<${#labels[@]}; job_index+=${#GPU_LIST[@]})); do
    label="${labels[$job_index]}"
    run_id="${BATCH_ID}_${label}"
    wait_for_gpu_ready "${gpu}" "${label}"
    echo "Starting ${label} on physical GPU ${gpu} with AMP off and TF32 off"
    if SEED=0 GPU="${gpu}" ADAPTER_MODE="${adapter_modes[$job_index]}" \
      LOSS_ROUTING="${routings[$job_index]}" ADAPTER_LAYERS=8 \
      BOTTLENECK="${bottlenecks[$job_index]}" \
      ADAPTER_LR="${adapter_lrs[$job_index]}" ADAPTER_SCALE=0.1 \
      ADAPTER_ACTIVATION=relu ADAPTER_TASK_INIT=independent \
      ASL_GAMMA_NEG=9.8 ASL_GAMMA_POS=0.0 ASL_CLIP=0.05 \
      NO_AMP=1 NO_TF32=1 PYTHON="${PYTHON}" \
      CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" DATA_ROOT="${DATA_ROOT}" \
      RUN_ID="${run_id}" OUTPUT_BASE="${OUTPUT_BASE}" LOG_DIR="${LOG_DIR}" \
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
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" != "0" ]]; then
  echo "At least one capacity/LR run failed; completed runs were preserved" >&2
  exit 1
fi

summary_arguments=()
for label in "${labels[@]}"; do
  summary_arguments+=(--run "${label}=${OUTPUT_BASE}/${BATCH_ID}_${label}")
done
"${PYTHON}" -m multi_lane.track_a.summarize_image_token_asl_capacity_lr \
  "${summary_arguments[@]}" \
  --output "${OUTPUT_BASE}/capacity_lr_summary.json"
echo "MULTI_LANE_IMAGE_TOKEN_ASL_CAPACITY_LR_COMPLETE ${OUTPUT_BASE}"
