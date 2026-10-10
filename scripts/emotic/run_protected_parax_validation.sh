#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
METHOD="${METHOD:?Set METHOD}"
RUN_ID="${RUN_ID:?Set RUN_ID}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-./datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-./models/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-./output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/emotic_protected_parax_val}"
LOG_ROOT="${LOG_ROOT:-./logs/emotic_protected_parax_val}"
case "${METHOD}" in
  baseline) mode=disabled ;;
  frozen_pool) mode=post_task_protected_frozen ;;
  fresh_only) mode=post_task_protected_fresh ;;
  reuse_old) mode=post_task_protected_reuse ;;
  *) echo "Unknown method: ${METHOD}" >&2; exit 2 ;;
esac
extra=()
if [[ "${METHOD}" != baseline ]]; then
  extra+=(--protected-parax-residual-ablation)
  if [[ "${STAGED_TRAINING:-0}" == 1 ]]; then
    extra+=(--protected-parax-staged-training --parax-calibration-epochs "${CALIBRATION_EPOCHS:-5}" --parax-calibration-consistency-weight 0.1)
  fi
fi
[[ "${UPDATES_PER_TASK:-0}" == 0 ]] || extra+=(--optimizer-updates-per-task "${UPDATES_PER_TASK}")
for path in "${DATA_ROOT}/CVPR17_Annotations.mat" "${CLIP_CHECKPOINT}" \
  "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" "${FACE_MANIFEST_ROOT}/manifests/val.jsonl"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
run_root="${OUTPUT_ROOT}/${RUN_ID}/${METHOD}"
[[ ! -e "${run_root}" ]] || { echo "Output already exists: ${run_root}" >&2; exit 2; }
mkdir -p "${LOG_ROOT}/${RUN_ID}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --protected-parax-paired-audit --seed 0 --training-protocol incremental \
  --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" --output-root "${run_root}" \
  --epochs "${EPOCHS:-30}" --max-tasks 3 --scheduler-mode cosine \
  --scheduler-min-lr-ratio 0 --scheduler-warmup-ratio 0 \
  --train-batch-size 64 --eval-batch-size 64 --workers 2 \
  --threshold 0.5 --source-learning-rate 0.05 --source-reference-batch-size 256 \
  --weight-decay 0 --temperature 1 --training-loss-mode legacy_full_zero --loss-routing joint_bce \
  --no-save-checkpoints --save-compact-checkpoints --input-mode full --input-normalization clip \
  --train-crop-scale 0.05 1.0 --full-crop-mode legacy \
  --person-crop-margin 0.15 --person-transform-mode letterbox \
  --person-color-jitter-strength 0 --person-color-jitter-probability 0 \
  --view-fusion fixed_three_view --view-classifier-mode shared_post_fusion \
  --view-fusion-hidden-dim 16 --view-fusion-learning-rate 0.0004 \
  --view-auxiliary-loss-weight 0.1 --view-gradient-routing joint \
  --view-evaluation-diagnostics --save-evaluation-scores --save-view-evaluation-scores \
  --evaluation-score-purpose validation_search --adapter-mode disabled \
  --adapter-learning-rate 0.0004 --adapter-weight-decay 0 \
  --parax-mode "${mode}" --parax-layer-indices 0 --parax-rank 32 --parax-num-experts 6 \
  --parax-router-hidden 16 --parax-initialization zero_output \
  --parax-residual-scale 1 --parax-output-scale-mode fixed --parax-smooth-ratio-bound 0.02 \
  --parax-trainable-components all --parax-residual-ratio-cap 0 \
  --selector-mode shared --prompt-mode shared --num-selectors 10 \
  --amp-initial-scale 1024 --amp-growth-interval 1000000000 \
  --reporting-split val "${extra[@]}" 2>&1 | tee "${LOG_ROOT}/${RUN_ID}/${METHOD}.log"
