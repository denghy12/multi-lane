#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

SEED=0
GPU="${GPU:?GPU is required}"
LOSS_ROUTING="${LOSS_ROUTING:?LOSS_ROUTING is required}"
case "${LOSS_ROUTING}" in
  model_asl|adapter_asl|both_asl) ;;
  *) echo "Expected LOSS_ROUTING=model_asl, adapter_asl, or both_asl" >&2; exit 2 ;;
esac

RUN_ID="${RUN_ID:-image_token_${LOSS_ROUTING}_b32_layer8_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_image_token_asl_formal_v0.1}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
LOG_DIR="${ROOT}/logs/emotic_track_a_image_token_asl_formal"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Formal run requires a clean Git worktree" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}"

echo "Image-token ASL formal run: routing=${LOSS_ROUTING} seed=0 gpu=${GPU} tasks=8 epochs=30 batch=64 adapter=image_token layer=8 bottleneck=32 adapter_lr=0.0004 ASL=gamma_neg9.8/gamma_pos0/clip0.05/eps1e-8 loss_view=legacy_full_zero normalization=clip crop=0.05-1.0 reporting=test"
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
  --loss-routing "${LOSS_ROUTING}" \
  --asl-gamma-neg 9.8 \
  --asl-gamma-pos 0.0 \
  --asl-clip 0.05 \
  --asl-eps 1e-8 \
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

echo "IMAGE_TOKEN_ASL_FORMAL_SEED0_COMPLETE routing=${LOSS_ROUTING} run_root=${RUN_ROOT} log=${LOG_PATH}"
