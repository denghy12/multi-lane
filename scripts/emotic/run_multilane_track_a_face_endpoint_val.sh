#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_face_endpoint_val}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }
[[ -f "${FACE_MANIFEST_ROOT}/audit_summary.json" ]] || { echo "Missing Face audit: ${FACE_MANIFEST_ROOT}" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Face validation requires a clean Git worktree" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}"

echo "Face endpoint validation: run_id=${RUN_ID} seed=0 gpu=${GPU} dataset=EMOTIC input=face_crop manifest=${FACE_MANIFEST_ROOT} train_filter=valid_and_nonambiguous transform=margin15_letterbox224_flip_train normalization=clip tasks=8 epochs=30 batch=64 scheduler=cosine eta_min=0 warmup=0 layer=1 bottleneck=32 main_lr=0.0125 adapter_lr=0.0004 scale=0.1 relu independent main_loss=bce adapter_loss=asl9.8/0/0.05 amp=on tf32=on reporting=val scores=on checkpoints=off test=forbidden"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed 0 \
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
  --evaluation-score-purpose validation_search \
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
  --reporting-split val \
  2>&1 | tee "${LOG_PATH}"

echo "FACE_ENDPOINT_VALIDATION_RUN_COMPLETE run_root=${RUN_ROOT} log=${LOG_PATH}"
