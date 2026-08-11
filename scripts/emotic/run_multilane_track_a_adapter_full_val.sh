#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

SEED="${SEED:-0}"
GPU="${GPU:-0}"
BOTTLENECK="${BOTTLENECK:-64}"
ADAPTER_LR="${ADAPTER_LR:-0.0004}"
ADAPTER_TASK_INIT="${ADAPTER_TASK_INIT:-copy_previous}"
RUN_ID="${RUN_ID:-task_lane_adapter_full_val_seed${SEED}_b${BOTTLENECK}_lr${ADAPTER_LR}_${ADAPTER_TASK_INIT}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_task_lane_adapter_warmstart_val_v0.2}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
LOG_DIR="${ROOT}/logs/emotic_track_a_adapter_warmstart_val"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}"

echo "Full validation run: seed=${SEED} gpu=${GPU} bottleneck=${BOTTLENECK} adapter_lr=${ADAPTER_LR} task_init=${ADAPTER_TASK_INIT}"
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
  --input-mode full \
  --adapter-mode task_lane \
  --adapter-bottleneck-dim "${BOTTLENECK}" \
  --adapter-layer-indices 11 \
  --adapter-residual-scale 0.1 \
  --adapter-activation relu \
  --adapter-learning-rate "${ADAPTER_LR}" \
  --adapter-task-init "${ADAPTER_TASK_INIT}" \
  --max-tasks 8 \
  --reporting-split val \
  2>&1 | tee "${LOG_PATH}"
