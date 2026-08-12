#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

SEED=0
GPU="${GPU:?GPU is required}"
RUN_ID="${RUN_ID:-image_token_adapter_b32_layer8_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_image_token_adapter_formal_v0.1}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
LOG_DIR="${ROOT}/logs/emotic_track_a_image_token_adapter_formal"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Formal run requires a clean Git worktree" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}"

echo "Image-token Adapter formal run: seed=0 gpu=${GPU} tasks=8 epochs=30 batch=64 adapter=image_token layer=8 bottleneck=32 adapter_lr=0.0004 task_init=independent loss=legacy_full_zero normalization=clip crop=0.05-1.0 reporting=test"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed "${SEED}" \
  --data-root "${DATA_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output-root "${RUN_ROOT}" \
  --epochs 30 \
  --train-batch-size 64 \
  --eval-batch-size 64 \
  --workers 2 \
  --threshold 0.5 \
  --source-learning-rate 0.05 \
  --source-reference-batch-size 256 \
  --weight-decay 0.0 \
  --temperature 1.0 \
  --training-loss-mode legacy_full_zero \
  --input-mode full \
  --input-normalization clip \
  --train-crop-scale 0.05 1.0 \
  --adapter-mode image_token \
  --adapter-bottleneck-dim 32 \
  --adapter-layer-indices 8 \
  --adapter-residual-scale 0.1 \
  --adapter-activation relu \
  --adapter-learning-rate 0.0004 \
  --adapter-task-init independent \
  --max-tasks 8 \
  --reporting-split test \
  2>&1 | tee "${LOG_PATH}"

echo "IMAGE_TOKEN_ADAPTER_FORMAL_SEED0_COMPLETE run_root=${RUN_ROOT} log=${LOG_PATH}"
