#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

SEED="${SEED:?SEED is required}"
GPU="${GPU:?GPU is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"

mkdir -p "${ROOT}/logs/emotic_track_a/$(basename "${RUN_ROOT}")"
LOG_PATH="${ROOT}/logs/emotic_track_a/$(basename "${RUN_ROOT}")/seed${SEED}.log"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed "${SEED}" \
  --data-root "${DATA_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output-root "${RUN_ROOT}/seed${SEED}" \
  --train-batch-size 64 \
  --eval-batch-size 64 \
  --workers 2 \
  --threshold 0.5 \
  2>&1 | tee "${LOG_PATH}"
