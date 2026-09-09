#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
SEED="${SEED:?SEED is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:?FACE_MANIFEST_ROOT is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_fixed_three_view_test}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${FACE_MANIFEST_ROOT}/manifests/test.jsonl" ]] || { echo "Missing test Face manifest" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Formal test requires a clean worktree" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}"

echo "Locked Face formal test: seed=${SEED} gpu=${GPU} EMOTIC 8 tasks x30 epochs batch64 main_lr=0.0125 cosine min0 no warmup image-token layer1 b32 adapter_lr=4e-4 scale0.1 relu independent main=BCE adapter=ASL9.8/0/0.05 AMP/TF32=on full-train test scores=on checkpoints=off"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed "${SEED}" \
  --data-root "${DATA_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output-root "${RUN_ROOT}" \
  --epochs 30 \
  --scheduler-mode cosine \
  --scheduler-min-lr-ratio 0 \
  --scheduler-warmup-ratio 0 \
  --train-batch-size 64 \
  --eval-batch-size 64 \
  --workers 2 \
  --threshold 0.5 \
  --source-learning-rate 0.05 \
  --source-reference-batch-size 256 \
  --weight-decay 0 \
  --temperature 1 \
  --training-loss-mode legacy_full_zero \
  --loss-routing adapter_asl \
  --asl-gamma-neg 9.8 \
  --asl-gamma-pos 0 \
  --asl-clip 0.05 \
  --asl-eps 1e-8 \
  --no-save-checkpoints \
  --input-mode face_crop \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --save-evaluation-scores \
  --evaluation-score-purpose fixed_test_fusion \
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

echo "FACE_FIXED_TEST_SOURCE_COMPLETE seed=${SEED} run_root=${RUN_ROOT}"
