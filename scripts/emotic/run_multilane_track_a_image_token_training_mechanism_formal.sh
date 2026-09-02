#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
BUDGET_MODE="${BUDGET_MODE:?BUDGET_MODE is required}"
UPDATES_PER_TASK="${UPDATES_PER_TASK:-0}"
REGULARIZATION="${REGULARIZATION:-none}"
REGULARIZATION_FRACTION="${REGULARIZATION_FRACTION:-0}"
GATE_MODE="${GATE_MODE:-fixed}"
GATE_INITIAL_SCALE="${GATE_INITIAL_SCALE:-0.1}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_image_token_training_mechanisms_formal}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

[[ "${BUDGET_MODE}" == "epochs" || "${BUDGET_MODE}" == "updates" ]] || {
  echo "BUDGET_MODE must be epochs or updates" >&2
  exit 2
}
if [[ "${BUDGET_MODE}" == "updates" ]]; then
  [[ "${UPDATES_PER_TASK}" =~ ^[1-9][0-9]*$ ]] || {
    echo "UPDATES_PER_TASK must be a positive integer for update budgets" >&2
    exit 2
  }
fi
[[ "${REGULARIZATION}" == "none" || "${REGULARIZATION}" == "residual_ratio" || "${REGULARIZATION}" == "feature_cosine" ]] || {
  echo "Unknown REGULARIZATION mode" >&2
  exit 2
}
[[ "${GATE_MODE}" == "fixed" || "${GATE_MODE}" == "learnable" ]] || {
  echo "GATE_MODE must be fixed or learnable" >&2
  exit 2
}
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Formal run requires a clean Git worktree" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}"

budget_args=(--epochs 30)
if [[ "${BUDGET_MODE}" == "updates" ]]; then
  budget_args+=(--optimizer-updates-per-task "${UPDATES_PER_TASK}")
fi

echo "Training-mechanism formal run: run_id=${RUN_ID} seed=0 gpu=${GPU} tasks=8 budget_mode=${BUDGET_MODE} epochs=30 updates_per_task=${UPDATES_PER_TASK} batch=64 layer=1 bottleneck=32 main_lr=0.0125 adapter_lr=0.0004 adapter_weight_decay=0 scale_or_gate_init=${GATE_INITIAL_SCALE} gate_mode=${GATE_MODE} regularization=${REGULARIZATION} regularization_fraction=${REGULARIZATION_FRACTION} activation=relu main_loss=bce adapter_loss=asl9.8/0/0.05 amp=on tf32=on reporting=test checkpoints=disabled"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed 0 \
  --data-root "${DATA_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output-root "${RUN_ROOT}" \
  "${budget_args[@]}" \
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
  --adapter-residual-scale "${GATE_INITIAL_SCALE}" \
  --adapter-residual-gate-mode "${GATE_MODE}" \
  --adapter-activation relu \
  --adapter-learning-rate 0.0004 \
  --adapter-weight-decay 0 \
  --adapter-task-init independent \
  --adapter-regularization "${REGULARIZATION}" \
  --adapter-regularization-fraction "${REGULARIZATION_FRACTION}" \
  --adapter-regularization-calibration-updates 30 \
  --max-tasks 8 \
  --reporting-split test \
  2>&1 | tee "${LOG_PATH}"

echo "IMAGE_TOKEN_TRAINING_MECHANISM_FORMAL_COMPLETE run_root=${RUN_ROOT} log=${LOG_PATH}"
