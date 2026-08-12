#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

SEED="${SEED:-0}"
GPU="${GPU:-0}"
METHOD_MODE="${METHOD_MODE:-task_lane}"
BOTTLENECK="${BOTTLENECK:-64}"
ADAPTER_LR="${ADAPTER_LR:-0.0004}"
ADAPTER_TASK_INIT="${ADAPTER_TASK_INIT:-independent}"
ADAPTER_LAYERS="${ADAPTER_LAYERS:-11}"
TRAINING_LOSS_MODE="${TRAINING_LOSS_MODE:-current_only}"
INPUT_NORMALIZATION="${INPUT_NORMALIZATION:-none}"
TRAIN_CROP_MIN="${TRAIN_CROP_MIN:-0.05}"
DIAGNOSTIC_TAG="${DIAGNOSTIC_TAG:-adapter_diagnostic}"
LAYER_TAG="$(tr ' ' '-' <<< "${ADAPTER_LAYERS}")"
RUN_ID="${RUN_ID:-${DIAGNOSTIC_TAG}_${METHOD_MODE}_seed${SEED}_layers${LAYER_TAG}_${TRAINING_LOSS_MODE}_${INPUT_NORMALIZATION}_crop${TRAIN_CROP_MIN}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_adapter_diagnostics_v0.3}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
LOG_DIR="${ROOT}/logs/emotic_track_a_adapter_diagnostics"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

read -r -a ADAPTER_LAYER_ARGS <<< "${ADAPTER_LAYERS}"
[[ "${SEED}" == "0" ]] || { echo "Diagnostics are restricted to seed0" >&2; exit 2; }
[[ "${METHOD_MODE}" == "disabled" || "${METHOD_MODE}" == "task_lane" ]] || { echo "Invalid method mode" >&2; exit 2; }
if [[ "${METHOD_MODE}" == "task_lane" ]]; then
  [[ "${ADAPTER_TASK_INIT}" == "independent" ]] || { echo "Position diagnostics require independent Adapter initialization" >&2; exit 2; }
fi
[[ "${TRAINING_LOSS_MODE}" == "legacy_full_zero" || "${TRAINING_LOSS_MODE}" == "current_only" ]] || { echo "Invalid training loss mode" >&2; exit 2; }
[[ "${INPUT_NORMALIZATION}" == "none" || "${INPUT_NORMALIZATION}" == "clip" ]] || { echo "Invalid input normalization" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Diagnostic run requires a clean Git worktree" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}"

echo "Adapter diagnostic: method=${METHOD_MODE} seed=${SEED} gpu=${GPU} layers=${ADAPTER_LAYERS} bottleneck=${BOTTLENECK} adapter_lr=${ADAPTER_LR} task_init=${ADAPTER_TASK_INIT} loss=${TRAINING_LOSS_MODE} normalization=${INPUT_NORMALIZATION} crop_min=${TRAIN_CROP_MIN}"
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
  --training-loss-mode "${TRAINING_LOSS_MODE}" \
  --input-mode full \
  --input-normalization "${INPUT_NORMALIZATION}" \
  --train-crop-scale "${TRAIN_CROP_MIN}" 1.0 \
  --adapter-mode "${METHOD_MODE}" \
  --adapter-bottleneck-dim "${BOTTLENECK}" \
  --adapter-layer-indices "${ADAPTER_LAYER_ARGS[@]}" \
  --adapter-residual-scale 0.1 \
  --adapter-activation relu \
  --adapter-learning-rate "${ADAPTER_LR}" \
  --adapter-task-init "${ADAPTER_TASK_INIT}" \
  --max-tasks 8 \
  --reporting-split val \
  2>&1 | tee "${LOG_PATH}"
