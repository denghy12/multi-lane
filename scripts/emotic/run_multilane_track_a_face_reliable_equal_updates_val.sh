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
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_face_equal_updates}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"

[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean worktree required" >&2; exit 2; }
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

echo "Face equal-update validation: variant=reliable_m15 seed=0 gpu=${GPU} dataset=EMOTIC train_filter=valid_nonambiguous_shortside24_score0.6 margin=0.15 letterbox224 tasks=8 updates=[1920,1620,360,3570,1710,960,240,600] batch=64 cosine_per_task_min0_nowarmup layer1_b32_adapterLR4e-4_scale0.1_relu_independent mainBCE_adapterASL9.8_0_0.05 AMP_TF32 val_scores no_checkpoint test_forbidden"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed 0 --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output-root "${RUN_ROOT}" --epochs 30 --optimizer-updates-by-task \
  1920 1620 360 3570 1710 960 240 600 --scheduler-mode cosine \
  --scheduler-min-lr-ratio 0 --scheduler-warmup-ratio 0 \
  --train-batch-size 64 --eval-batch-size 64 --workers 2 --threshold 0.5 \
  --source-learning-rate 0.05 --source-reference-batch-size 256 --weight-decay 0 \
  --temperature 1 --training-loss-mode legacy_full_zero --loss-routing adapter_asl \
  --asl-gamma-neg 9.8 --asl-gamma-pos 0 --asl-clip 0.05 --asl-eps 1e-8 \
  --no-save-checkpoints --input-mode face_crop --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --face-crop-margin 0.15 --face-min-training-short-side 24 \
  --face-min-training-score 0.6 --face-color-jitter-strength 0 \
  --face-color-jitter-probability 0 \
  --save-evaluation-scores --evaluation-score-purpose validation_search \
  --input-normalization clip --train-crop-scale 0.05 1.0 \
  --adapter-mode image_token --adapter-bottleneck-dim 32 --adapter-layer-indices 1 \
  --adapter-residual-scale 0.1 --adapter-residual-gate-mode fixed \
  --adapter-activation relu --adapter-learning-rate 0.0004 --adapter-weight-decay 0 \
  --adapter-task-init independent --adapter-regularization none \
  --max-tasks 8 --reporting-split val \
  2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"

echo "FACE_RELIABLE_EQUAL_UPDATES_COMPLETE run=${RUN_ROOT}"
