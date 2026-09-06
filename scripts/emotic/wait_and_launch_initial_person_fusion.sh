#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

FULL_RUN="${FULL_RUN:?FULL_RUN is required}"
VALIDATION_FUSION_SUMMARY="${VALIDATION_FUSION_SUMMARY:?VALIDATION_FUSION_SUMMARY is required}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:?CLIP_CHECKPOINT is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_USED_MIB="${MAX_USED_MIB:-2000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-2}"

[[ -z "$(git status --porcelain)" ]] || { echo "Requires a clean Git worktree" >&2; exit 2; }
[[ -d "${FULL_RUN}" ]] || { echo "Missing Full source run: ${FULL_RUN}" >&2; exit 2; }
[[ -f "${VALIDATION_FUSION_SUMMARY}" ]] || {
  echo "Missing validation fusion summary: ${VALIDATION_FUSION_SUMMARY}" >&2; exit 2;
}
[[ -f "${CLIP_CHECKPOINT}" && -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || {
  echo "Missing CLIP checkpoint or EMOTIC annotations" >&2; exit 2;
}

selected=""
consecutive=0
while (( consecutive < REQUIRED_IDLE_CHECKS )); do
  candidate=""
  while IFS=',' read -r index used utilization; do
    index="${index// /}"
    used="${used// /}"
    utilization="${utilization// /}"
    if (( used <= MAX_USED_MIB && utilization <= MAX_UTILIZATION )); then
      candidate="${index}"
      break
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits)

  if [[ -n "${candidate}" && ( -z "${selected}" || "${candidate}" == "${selected}" ) ]]; then
    selected="${candidate}"
    consecutive=$((consecutive + 1))
    echo "IDLE_GPU_CHECK timestamp=$(date -Iseconds) gpu=${selected} consecutive=${consecutive}/${REQUIRED_IDLE_CHECKS}"
  else
    selected="${candidate}"
    consecutive=0
    echo "WAITING_FOR_IDLE_GPU timestamp=$(date -Iseconds) candidate=${selected:-none}"
  fi
  if (( consecutive < REQUIRED_IDLE_CHECKS )); then
    sleep "${POLL_SECONDS}"
  fi
done

echo "SELECTED_GPU=${selected} timestamp=$(date -Iseconds)"
CUDA_VISIBLE_DEVICES="${selected}" "${PYTHON}" -m multi_lane.track_a.smoke \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --adapter-mode image_token \
  --adapter-bottleneck-dim 32 \
  --adapter-layer-indices 1 \
  --adapter-task-init independent \
  --adapter-residual-scale 0.1 \
  --loss-routing adapter_asl

export GPU="${selected}" FULL_RUN VALIDATION_FUSION_SUMMARY CLIP_CHECKPOINT PYTHON DATA_ROOT
bash scripts/emotic/launch_multilane_track_a_initial_person_fusion.sh
