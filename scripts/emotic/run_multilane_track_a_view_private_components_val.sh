#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
GPU="${GPU:?GPU must be set}"
METHOD="${METHOD:?METHOD must be set}"
RUN_ID="${RUN_ID:?RUN_ID must be set}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE must be set}"
LOG_DIR="${LOG_DIR:?LOG_DIR must be set}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-0}"
NO_AMP="${NO_AMP:-0}"

precision_args=()
if [[ "${NO_AMP}" == "1" ]]; then
  precision_args+=(--no-amp)
fi

case "${METHOD}" in
  P0) PROMPT_MODE=shared       ADAPTER_VIEW_MODE=shared      ADAPTER_VIEW_DIM=0  CLASSIFIER_MODE=shared_post_fusion ;;
  P1) PROMPT_MODE=view_specific ADAPTER_VIEW_MODE=shared      ADAPTER_VIEW_DIM=0  CLASSIFIER_MODE=shared_post_fusion ;;
  P2) PROMPT_MODE=shared       ADAPTER_VIEW_MODE=independent ADAPTER_VIEW_DIM=32 CLASSIFIER_MODE=shared_post_fusion ;;
  P3) PROMPT_MODE=view_specific ADAPTER_VIEW_MODE=independent ADAPTER_VIEW_DIM=32 CLASSIFIER_MODE=shared_post_fusion ;;
  H0) PROMPT_MODE=shared       ADAPTER_VIEW_MODE=shared      ADAPTER_VIEW_DIM=0  CLASSIFIER_MODE=shared_per_view ;;
  H1) PROMPT_MODE=shared       ADAPTER_VIEW_MODE=shared      ADAPTER_VIEW_DIM=0  CLASSIFIER_MODE=private_per_view ;;
  H2) PROMPT_MODE=view_specific ADAPTER_VIEW_MODE=independent ADAPTER_VIEW_DIM=32 CLASSIFIER_MODE=shared_per_view ;;
  H3) PROMPT_MODE=view_specific ADAPTER_VIEW_MODE=independent ADAPTER_VIEW_DIM=32 CLASSIFIER_MODE=private_per_view ;;
  *) echo "Unknown METHOD=${METHOD}" >&2; exit 2 ;;
esac

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed 0 \
  --data-root /mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC \
  --clip-checkpoint /mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt \
  --face-manifest-root /mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908 \
  --output-root "${OUTPUT_BASE}/${RUN_ID}" \
  --epochs 30 --max-tasks 8 \
  --scheduler-mode cosine --scheduler-min-lr-ratio 0 --scheduler-warmup-ratio 0 \
  --train-batch-size 64 --eval-batch-size 64 --workers 2 --threshold 0.5 \
  --source-learning-rate 0.05 --source-reference-batch-size 256 \
  --weight-decay 0 --temperature 1 \
  --training-loss-mode legacy_full_zero --loss-routing adapter_asl \
  --asl-gamma-neg 9.8 --asl-gamma-pos 0 --asl-clip 0.05 --asl-eps 1e-8 \
  --no-save-checkpoints --input-mode full --input-normalization clip \
  --train-crop-scale 0.05 1.0 --full-crop-mode legacy \
  --person-crop-margin 0.15 --person-transform-mode letterbox \
  --person-color-jitter-strength 0 --person-color-jitter-probability 0 \
  --view-fusion fixed_three_view --view-fusion-hidden-dim 16 \
  --view-fusion-learning-rate 0.0004 --view-auxiliary-loss-weight 0.1 \
  --view-gradient-routing joint --view-gradient-audit \
  --view-path-gradient-audit-epochs 0 14 29 --view-path-gradient-audit-batches 3 \
  --view-evaluation-diagnostics --save-evaluation-scores \
  --save-view-evaluation-scores --evaluation-score-purpose validation_search \
  --adapter-mode image_token --adapter-bottleneck-dim 32 \
  --adapter-view-bottleneck-dim "${ADAPTER_VIEW_DIM}" \
  --adapter-view-mode "${ADAPTER_VIEW_MODE}" \
  --adapter-layer-indices 1 --adapter-residual-scale 0.1 \
  --adapter-residual-gate-mode fixed --adapter-activation relu \
  --adapter-learning-rate 0.0004 --adapter-weight-decay 0 \
  --adapter-task-init independent --adapter-regularization none \
  --selector-mode view_specific --prompt-mode "${PROMPT_MODE}" --num-selectors 10 \
  --gradient-clip-norm "${GRADIENT_CLIP_NORM}" \
  --view-classifier-mode "${CLASSIFIER_MODE}" \
  --reporting-split val \
  "${precision_args[@]}" \
  > "${LOG_DIR}/${METHOD}.log" 2>&1
