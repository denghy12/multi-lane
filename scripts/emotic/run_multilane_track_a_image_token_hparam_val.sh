#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

SEED="${SEED:-0}"
GPU="${GPU:?GPU is required}"
LOSS_ROUTING="${LOSS_ROUTING:-adapter_asl}"
ASL_GAMMA_NEG="${ASL_GAMMA_NEG:-9.8}"
ASL_GAMMA_POS="${ASL_GAMMA_POS:-0.0}"
ASL_CLIP="${ASL_CLIP:-0.05}"
ASL_EPS="${ASL_EPS:-1e-8}"
ADAPTER_LAYERS="${ADAPTER_LAYERS:-8}"
BOTTLENECK="${BOTTLENECK:-32}"
ADAPTER_LR="${ADAPTER_LR:-0.0004}"
ADAPTER_SCALE="${ADAPTER_SCALE:-0.1}"
ADAPTER_ACTIVATION="${ADAPTER_ACTIVATION:-relu}"
ADAPTER_TASK_INIT="${ADAPTER_TASK_INIT:-independent}"
EPOCHS="${EPOCHS:-30}"
SOURCE_LR="${SOURCE_LR:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
LOG_DIR="${ROOT}/logs/emotic_track_a_image_token_hparam_val"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

read -r -a ADAPTER_LAYER_ARGS <<< "${ADAPTER_LAYERS}"
[[ "${SEED}" == "0" || "${SEED}" == "1" || "${SEED}" == "2" ]] || { echo "SEED must be 0, 1, or 2" >&2; exit 2; }
[[ "${LOSS_ROUTING}" == "joint_bce" || "${LOSS_ROUTING}" == "adapter_asl" ]] || { echo "Hyperparameter validation permits only joint_bce or adapter_asl" >&2; exit 2; }
[[ "${ADAPTER_ACTIVATION}" == "relu" || "${ADAPTER_ACTIVATION}" == "gelu" ]] || { echo "Invalid Adapter activation" >&2; exit 2; }
[[ "${ADAPTER_TASK_INIT}" == "independent" || "${ADAPTER_TASK_INIT}" == "copy_previous" ]] || { echo "Invalid Adapter task initialization" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Hyperparameter validation requires a clean Git worktree" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}"

echo "Image-token hyperparameter validation: seed=${SEED} gpu=${GPU} routing=${LOSS_ROUTING} gamma_neg=${ASL_GAMMA_NEG} gamma_pos=${ASL_GAMMA_POS} clip=${ASL_CLIP} layers=${ADAPTER_LAYERS} bottleneck=${BOTTLENECK} adapter_lr=${ADAPTER_LR} scale=${ADAPTER_SCALE} activation=${ADAPTER_ACTIVATION} task_init=${ADAPTER_TASK_INIT} epochs=${EPOCHS} source_lr=${SOURCE_LR} weight_decay=${WEIGHT_DECAY} temperature=${TEMPERATURE} reporting=val checkpoints=disabled"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed "${SEED}" \
  --data-root "${DATA_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output-root "${RUN_ROOT}" \
  --epochs "${EPOCHS}" \
  --train-batch-size 64 \
  --eval-batch-size 64 \
  --workers 2 \
  --threshold 0.5 \
  --source-learning-rate "${SOURCE_LR}" \
  --source-reference-batch-size 256 \
  --weight-decay "${WEIGHT_DECAY}" \
  --temperature "${TEMPERATURE}" \
  --training-loss-mode legacy_full_zero \
  --loss-routing "${LOSS_ROUTING}" \
  --asl-gamma-neg "${ASL_GAMMA_NEG}" \
  --asl-gamma-pos "${ASL_GAMMA_POS}" \
  --asl-clip "${ASL_CLIP}" \
  --asl-eps "${ASL_EPS}" \
  --no-save-checkpoints \
  --input-mode full \
  --input-normalization clip \
  --train-crop-scale 0.05 1.0 \
  --adapter-mode image_token \
  --adapter-bottleneck-dim "${BOTTLENECK}" \
  --adapter-layer-indices "${ADAPTER_LAYER_ARGS[@]}" \
  --adapter-residual-scale "${ADAPTER_SCALE}" \
  --adapter-activation "${ADAPTER_ACTIVATION}" \
  --adapter-learning-rate "${ADAPTER_LR}" \
  --adapter-task-init "${ADAPTER_TASK_INIT}" \
  --max-tasks 8 \
  --reporting-split val \
  2>&1 | tee "${LOG_PATH}"

echo "IMAGE_TOKEN_HPARAM_VAL_COMPLETE run_root=${RUN_ROOT} log=${LOG_PATH}"
