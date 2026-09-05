#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_GROUP="${RUN_GROUP:?RUN_GROUP is required}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:?CLIP_CHECKPOINT is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-./datasets/EMOTIC}"
REPORTING_SPLIT="${REPORTING_SPLIT:-test}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"

[[ -z "$(git status --porcelain)" ]] || { echo "Requires a clean Git worktree" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" && -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || {
  echo "Missing CLIP checkpoint or EMOTIC annotations" >&2; exit 2;
}

while true; do
  idle=()
  while IFS=',' read -r index used utilization; do
    index="${index// /}"
    used="${used// /}"
    utilization="${utilization// /}"
    if (( used <= MAX_USED_MIB && utilization <= MAX_UTILIZATION )); then
      idle+=("${index}")
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits)
  if (( ${#idle[@]} >= 4 )); then
    selected=("${idle[@]:0:4}")
    break
  fi
  echo "WAITING_FOR_FOUR_IDLE_GPUS timestamp=$(date -Iseconds) idle_count=${#idle[@]}"
  sleep "${POLL_SECONDS}"
done

echo "SELECTED_GPUS=${selected[*]} timestamp=$(date -Iseconds)"
common_smoke=(
  --clip-checkpoint "${CLIP_CHECKPOINT}"
  --adapter-mode image_token --adapter-layer-indices 1
  --adapter-bottleneck-dim 32 --adapter-residual-scale 0.1
  --adapter-task-init independent --loss-routing adapter_asl
  --selector-condition-layers 1
)
CUDA_VISIBLE_DEVICES="${selected[0]}" "${PYTHON}" -m multi_lane.track_a.smoke \
  "${common_smoke[@]}" --selector-conditioning disabled
CUDA_VISIBLE_DEVICES="${selected[0]}" "${PYTHON}" -m multi_lane.track_a.smoke \
  "${common_smoke[@]}" --selector-conditioning person_patches

export CLIP_CHECKPOINT PYTHON DATA_ROOT REPORTING_SPLIT
export GPU_LIST="${selected[0]},${selected[1]},${selected[2]},${selected[3]}"
export RUN_GROUP
bash scripts/emotic/launch_multilane_track_a_selector_patch_2x2.sh
