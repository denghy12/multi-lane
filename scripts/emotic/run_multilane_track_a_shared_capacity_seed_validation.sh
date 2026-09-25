#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
SEED="${SEED:?SEED is required}"
METHOD="${METHOD:?METHOD is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_shared_capacity_multiseed_validation}"
ADAPTER_RESIDUAL_SCALE="${ADAPTER_RESIDUAL_SCALE:-0.03}"

case "${SEED}" in
  1|2) ;;
  *) echo "This replication is locked to seed 1 or 2" >&2; exit 2 ;;
esac
case "${METHOD}" in
  A0-shared-b32-scale003) shared_bottleneck=32; expected_parameters=49952 ;;
  A-cap-shared-b97-scale003) shared_bottleneck=97; expected_parameters=149857 ;;
  *) echo "Unknown METHOD=${METHOD}" >&2; exit 2 ;;
esac

RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" && -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || {
  echo "Missing CLIP checkpoint or EMOTIC annotations" >&2
  exit 2
}
[[ -f "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" \
  && -f "${FACE_MANIFEST_ROOT}/manifests/val.jsonl" ]] || {
  echo "Missing train/validation Face manifests" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "Validation requires a clean Git worktree" >&2
  exit 2
}
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

echo "Shared capacity multiseed validation: method=${METHOD} seed=${SEED} shared_b${shared_bottleneck} residual_scale=${ADAPTER_RESIDUAL_SCALE} parameters_per_task=${expected_parameters} tasks=0-7 epochs=30/task validation-only test-forbidden"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 "${PYTHON}" -m multi_lane.track_a.runner \
  --seed "${SEED}" --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" --output-root "${RUN_ROOT}" \
  --epochs 30 --max-tasks 8 --scheduler-mode cosine --scheduler-min-lr-ratio 0 \
  --scheduler-warmup-ratio 0 --train-batch-size 64 --eval-batch-size 64 --workers 2 \
  --threshold 0.5 --source-learning-rate 0.05 --source-reference-batch-size 256 \
  --weight-decay 0 --temperature 1 --training-loss-mode legacy_full_zero \
  --loss-routing adapter_asl --asl-gamma-neg 9.8 --asl-gamma-pos 0 --asl-clip 0.05 --asl-eps 1e-8 \
  --no-save-checkpoints --input-mode full --input-normalization clip --train-crop-scale 0.05 1.0 \
  --full-crop-mode legacy --person-crop-margin 0.15 --person-transform-mode letterbox \
  --person-color-jitter-strength 0 --person-color-jitter-probability 0 \
  --view-fusion fixed_three_view --view-fusion-hidden-dim 16 --view-fusion-learning-rate 0.0004 \
  --view-auxiliary-loss-weight 0.1 --view-gradient-routing joint --view-evaluation-diagnostics \
  --save-evaluation-scores --evaluation-score-purpose validation_search \
  --adapter-mode image_token --adapter-bottleneck-dim "${shared_bottleneck}" \
  --adapter-view-mode shared --adapter-view-bottleneck-dim 0 --adapter-layer-indices 1 \
  --adapter-residual-scale "${ADAPTER_RESIDUAL_SCALE}" --adapter-residual-gate-mode fixed \
  --adapter-activation relu --adapter-learning-rate 0.0004 --adapter-weight-decay 0 \
  --adapter-task-init independent --adapter-regularization none --parax-mode disabled \
  --reporting-split val 2>&1 | tee "${LOG_PATH}"
echo "SHARED_CAPACITY_MULTISEED_VALIDATION_COMPLETE method=${METHOD} seed=${SEED} run=${RUN_ROOT}"
