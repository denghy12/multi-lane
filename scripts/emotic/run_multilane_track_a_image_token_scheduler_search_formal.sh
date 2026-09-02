#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
SCHEDULER_MODE="${SCHEDULER_MODE:?SCHEDULER_MODE is required}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0}"
WARMUP_RATIO="${WARMUP_RATIO:-0}"
MULTISTEP_MILESTONES="${MULTISTEP_MILESTONES:-0.6 0.85}"
MULTISTEP_GAMMA="${MULTISTEP_GAMMA:-0.1}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_image_token_scheduler_search_formal}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

read -r -a MILESTONE_VALUES <<< "${MULTISTEP_MILESTONES}"
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Formal run requires a clean Git worktree" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}"

echo "Image-token scheduler search: run_id=${RUN_ID} seed=0 gpu=${GPU} tasks=8 epochs=30 batch=64 scheduler=${SCHEDULER_MODE} min_lr_ratio=${MIN_LR_RATIO} warmup_ratio=${WARMUP_RATIO} multistep=${MULTISTEP_MILESTONES} gamma=${MULTISTEP_GAMMA} layer=1 bottleneck=32 main_lr=0.0125 adapter_lr=0.0004 adapter_weight_decay=0 scale=0.1 activation=relu main_loss=bce adapter_loss=asl9.8/0/0.05 amp=on tf32=on reporting=test checkpoints=disabled"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed 0 \
  --data-root "${DATA_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output-root "${RUN_ROOT}" \
  --epochs 30 \
  --scheduler-mode "${SCHEDULER_MODE}" \
  --scheduler-min-lr-ratio "${MIN_LR_RATIO}" \
  --scheduler-warmup-ratio "${WARMUP_RATIO}" \
  --scheduler-multistep-milestones "${MILESTONE_VALUES[@]}" \
  --scheduler-multistep-gamma "${MULTISTEP_GAMMA}" \
  --train-batch-size 64 \
  --eval-batch-size 64 \
  --workers 2 \
  --threshold 0.5 \
  --source-learning-rate 0.05 \
  --source-reference-batch-size 256 \
  --weight-decay 0.0 \
  --temperature 1.0 \
  --training-loss-mode legacy_full_zero \
  --loss-routing adapter_asl \
  --asl-gamma-neg 9.8 \
  --asl-gamma-pos 0.0 \
  --asl-clip 0.05 \
  --asl-eps 1e-8 \
  --no-save-checkpoints \
  --input-mode full \
  --input-normalization clip \
  --train-crop-scale 0.05 1.0 \
  --adapter-mode image_token \
  --adapter-bottleneck-dim 32 \
  --adapter-layer-indices 1 \
  --adapter-residual-scale 0.1 \
  --adapter-residual-gate-mode fixed \
  --adapter-activation relu \
  --adapter-learning-rate 0.0004 \
  --adapter-weight-decay 0 \
  --adapter-task-init independent \
  --adapter-regularization none \
  --max-tasks 8 \
  --reporting-split test \
  2>&1 | tee "${LOG_PATH}"

echo "IMAGE_TOKEN_SCHEDULER_SEARCH_FORMAL_COMPLETE run_root=${RUN_ROOT} log=${LOG_PATH}"
