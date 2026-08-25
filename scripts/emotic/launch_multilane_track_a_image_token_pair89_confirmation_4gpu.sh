#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPUS_TEXT="${GPUS:-0 1 2 3}"
read -r -a GPU_LIST <<< "${GPUS_TEXT}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-8000}"
GPU_MAX_UTIL_PERCENT="${GPU_MAX_UTIL_PERCENT:-10}"
GPU_READY_CHECKS="${GPU_READY_CHECKS:-2}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"

[[ "${#GPU_LIST[@]}" -eq 4 ]] || { echo "Pair8/9 confirmation requires exactly 4 GPUs" >&2; exit 2; }
for gpu in "${GPU_LIST[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "GPU identifiers must be nonnegative integers" >&2; exit 2; }
done
[[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l | tr -d ' ')" -eq 4 ]] || {
  echo "Pair8/9 confirmation requires 4 distinct GPUs" >&2
  exit 2
}
[[ "${GPU_READY_CHECKS}" =~ ^[1-9][0-9]*$ ]] || { echo "GPU_READY_CHECKS must be positive" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Pair8/9 confirmation requires a clean Git worktree" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }

BATCH_ID="${BATCH_ID:-image_token_pair89_confirmation_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT}/output/emotic_image_token_tuning/pair89_confirmation/${BATCH_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_image_token_tuning/pair89_confirmation}"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"
MANIFEST="${OUTPUT_BASE}/confirmation_manifest.tsv"
STATUS_DIR="${OUTPUT_BASE}/launcher_status"
mkdir -p "${STATUS_DIR}"

labels=(disabled_bce single8_bce single9_bce pair8_9_bce)
adapter_modes=(disabled image_token image_token image_token)
layer_specs=(8 8 9 "8 9")

printf 'index\tlabel\tadapter_mode\tzero_based_layers\tloss\tbottleneck\tadapter_lr\tresidual_scale\tactivation\ttask_init\tprecision\tgpu\n' > "${MANIFEST}"
for index in "${!labels[@]}"; do
  printf '%s\t%s\t%s\t%s\tjoint_bce\t32\t0.0004\t0.1\trelu\tindependent\tfp32\t%s\n' \
    "${index}" "${labels[$index]}" "${adapter_modes[$index]}" \
    "${layer_specs[$index]}" "${GPU_LIST[$index]}" >> "${MANIFEST}"
done

wait_for_gpu_ready() {
  local gpu="$1"
  local purpose="$2"
  local consecutive=0
  local snapshot free_mib utilization
  while (( consecutive < GPU_READY_CHECKS )); do
    snapshot="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)"
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

wait_for_gpu_ready "${GPU_LIST[3]}" "pair8_9 BCE FP32 smoke"
CUDA_VISIBLE_DEVICES="${GPU_LIST[3]}" "${PYTHON}" -m multi_lane.track_a.smoke \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --adapter-mode image_token \
  --adapter-bottleneck-dim 32 \
  --adapter-layer-indices 8 9 \
  --adapter-task-init independent \
  --loss-routing joint_bce \
  --no-amp
echo "IMAGE_TOKEN_PAIR89_CONFIRMATION_FP32_SMOKE_COMPLETE gpu=${GPU_LIST[3]}"

run_one() {
  local index="$1"
  local gpu="${GPU_LIST[$index]}"
  local label="${labels[$index]}"
  local run_id="${BATCH_ID}_${label}"
  wait_for_gpu_ready "${gpu}" "${label}"
  echo "Starting ${label} layers=${layer_specs[$index]} on physical GPU ${gpu}"
  if SEED=0 GPU="${gpu}" ADAPTER_MODE="${adapter_modes[$index]}" \
    LOSS_ROUTING=joint_bce ADAPTER_LAYERS="${layer_specs[$index]}" \
    BOTTLENECK=32 ADAPTER_LR=0.0004 ADAPTER_SCALE=0.1 \
    ADAPTER_ACTIVATION=relu ADAPTER_TASK_INIT=independent \
    NO_AMP=1 PYTHON="${PYTHON}" CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
    DATA_ROOT="${DATA_ROOT}" RUN_ID="${run_id}" \
    OUTPUT_BASE="${OUTPUT_BASE}" LOG_DIR="${LOG_DIR}" \
    bash scripts/emotic/run_multilane_track_a_image_token_hparam_val.sh; then
    printf '0\n' > "${STATUS_DIR}/${label}.exit_code"
  else
    printf '1\n' > "${STATUS_DIR}/${label}.exit_code"
    return 1
  fi
}

pids=()
for index in "${!labels[@]}"; do run_one "${index}" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
if [[ "${failed}" != "0" ]]; then
  echo "At least one pair8/9 confirmation run failed" >&2
  exit 1
fi

summary_arguments=()
for label in "${labels[@]}"; do
  summary_arguments+=(--run "${label}=${OUTPUT_BASE}/${BATCH_ID}_${label}")
done
"${PYTHON}" -m multi_lane.track_a.summarize_image_token_pair89_confirmation \
  "${summary_arguments[@]}" \
  --output "${OUTPUT_BASE}/pair89_confirmation_summary.json"
echo "MULTI_LANE_IMAGE_TOKEN_PAIR89_CONFIRMATION_COMPLETE ${OUTPUT_BASE}"
