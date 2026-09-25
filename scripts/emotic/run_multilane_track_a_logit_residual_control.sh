#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
GPU="${GPU:?GPU is required}"
METHOD="${METHOD:?METHOD is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
LOG_PATH="${LOG_PATH:?LOG_PATH is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
case "${METHOD}" in
  B0-paired) FUSION=fixed_three_view ;;
  L-post-logit) FUSION=logit_residual_three_view ;;
  *) echo "Unknown METHOD=${METHOD}" >&2; exit 2 ;;
esac
RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
[[ ! -e "${RUN_ROOT}" ]] || { echo "Output already exists: ${RUN_ROOT}" >&2; exit 2; }
for path in "${CLIP_CHECKPOINT}" "${DATA_ROOT}/CVPR17_Annotations.mat" "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" "${FACE_MANIFEST_ROOT}/manifests/val.jsonl"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
mkdir -p "$(dirname "${LOG_PATH}")"
echo "Logit residual control method=${METHOD} seed=0 gpu=${GPU} tasks=3 epochs=30 batch=64 main_lr=0.0125 calibration_lr=0.0004 fusion=${FUSION} residual_scale=0.1 auxiliary_loss=0.1 parax=disabled adapter=disabled validation_only checkpoint=off test=forbidden AMP=on TF32=on"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed 0 --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" --output-root "${RUN_ROOT}" \
  --epochs 30 --max-tasks 3 --scheduler-mode cosine --scheduler-min-lr-ratio 0 \
  --scheduler-warmup-ratio 0 --train-batch-size 64 --eval-batch-size 64 --workers 2 \
  --threshold 0.5 --source-learning-rate 0.05 --source-reference-batch-size 256 \
  --weight-decay 0 --temperature 1 --training-loss-mode legacy_full_zero --loss-routing joint_bce \
  --no-save-checkpoints --input-mode full --input-normalization clip --train-crop-scale 0.05 1.0 \
  --full-crop-mode legacy --person-crop-margin 0.15 --person-transform-mode letterbox \
  --person-color-jitter-strength 0 --person-color-jitter-probability 0 \
  --view-fusion "${FUSION}" --view-fusion-hidden-dim 16 --view-fusion-learning-rate 0.0004 \
  --view-residual-scale 0.1 --view-auxiliary-loss-weight 0.1 --view-gradient-routing joint \
  --view-evaluation-diagnostics --save-evaluation-scores --evaluation-score-purpose validation_search \
  --adapter-mode disabled --parax-mode disabled --reporting-split val \
  2>&1 | tee "${LOG_PATH}"
