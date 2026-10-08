#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BATCH_ID="${BATCH_ID:-post_task_router_seed0_val_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-1 2}"
read -r -a GPUS <<< "${GPU_LIST}"
[[ "${#GPUS[@]}" -eq 2 && "${GPUS[0]}" != "${GPUS[1]}" ]] || {
  echo "GPU_LIST must contain two distinct GPU indices" >&2; exit 2;
}
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PYTHON="${PYTHON:-python}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-./models/clip/ViT-B-16.pt}"
SMOKE_DIR="${ROOT}/logs/emotic_track_a_post_task_router_val/${BATCH_ID}"
mkdir -p "${SMOKE_DIR}"
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint" >&2; exit 2; }

echo "Waiting for GPU ${GPUS[0]} with ${MIN_FREE_MIB} MiB free before real ViT smoke"
while true; do
  free_mib="$(nvidia-smi -i "${GPUS[0]}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free_mib >= MIN_FREE_MIB )) && break
  sleep "${POLL_SECONDS}"
done
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" -m multi_lane.track_a.smoke \
  --clip-checkpoint "${CLIP_CHECKPOINT}" --view-fusion fixed_three_view \
  --parax-mode post_task_router --parax-layer-indices 10 \
  --parax-rank 32 --parax-num-experts 3 --parax-router-hidden 16 \
  --parax-initialization zero_output --parax-residual-scale 0.001 \
  --parax-output-scale-mode fixed --parax-trainable-components router \
  --parax-freeze-center-after-task0 --loss-routing joint_bce --no-amp \
  > "${SMOKE_DIR}/real_vit_smoke.log" 2>&1 || {
    tail -60 "${SMOKE_DIR}/real_vit_smoke.log" >&2
    exit 1
  }
grep -q MULTI_LANE_TRACK_A_SMOKE_OK "${SMOKE_DIR}/real_vit_smoke.log"
date -Is > "${SMOKE_DIR}/real_vit_smoke_passed.txt"
echo "Real ViT smoke passed; starting paired task0-2 validation"
BATCH_ID="${BATCH_ID}" GPU_LIST="${GPU_LIST}" MIN_FREE_MIB="${MIN_FREE_MIB}" \
  POLL_SECONDS="${POLL_SECONDS}" PYTHON="${PYTHON}" \
  CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
  bash scripts/emotic/launch_multilane_track_a_post_task_router_val.sh
