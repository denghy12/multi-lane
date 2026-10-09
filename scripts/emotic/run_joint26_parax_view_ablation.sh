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
RESULT_ROOT="${RESULT_ROOT:-./output/emotic_joint26_parax_view_ablation}"
LOG_ROOT="${LOG_ROOT:-./logs/emotic_joint26_parax_view_ablation}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-30}"
PARAX_MODE=disabled
VIEW_FUSION=disabled
AUX_WEIGHT=0
LOSS_SCALE=1.1
extra_args=()
case "${METHOD}" in
  FULL_ONLY) ;;
  FULL_ONLY_PARAX) PARAX_MODE=image ;;
  THREE_VIEW) VIEW_FUSION=fixed_three_view ;;
  THREE_VIEW_PARAX) VIEW_FUSION=fixed_three_view; PARAX_MODE=image ;;
  *) echo "Unknown METHOD=${METHOD}" >&2; exit 2 ;;
esac
if [[ "${VIEW_FUSION}" == fixed_three_view ]]; then
  AUX_WEIGHT=0.1
  LOSS_SCALE=1
  extra_args+=(--face-manifest-root "${FACE_MANIFEST_ROOT}" --view-evaluation-diagnostics --save-view-evaluation-scores)
fi
if [[ -n "${SMOKE_UPDATES:-}" ]]; then
  extra_args+=(--optimizer-updates-per-task "${SMOKE_UPDATES}" --skip-validation-eval)
fi
for path in "${DATA_ROOT}/CVPR17_Annotations.mat" "${CLIP_CHECKPOINT}"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
if [[ "${VIEW_FUSION}" == fixed_three_view ]]; then
  for split in train val; do
    [[ -f "${FACE_MANIFEST_ROOT}/manifests/${split}.jsonl" ]] || {
      echo "Missing Face manifest: ${split}" >&2; exit 2;
    }
  done
fi
RUN_ROOT="${RESULT_ROOT}/${RUN_ID}/${METHOD}"
LOG_PATH="${LOG_ROOT}/${RUN_ID}/${METHOD}.log"
[[ ! -e "${RUN_ROOT}" ]] || { echo "Output already exists: ${RUN_ROOT}" >&2; exit 2; }
mkdir -p "$(dirname "${LOG_PATH}")"
echo "JOINT26 method=${METHOD} GPU=${GPU} seed=${SEED} labels=all26 lane=1 epochs=${EPOCHS} batch64 Adam main_lr0.0125 ParaX_lr0.0004 cosine clip_frozen ParaX=${PARAX_MODE} image_adapter=off validation_only test_forbidden loss_scale=${LOSS_SCALE} view_auxiliary=${AUX_WEIGHT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --training-protocol joint26 --max-tasks 1 --seed "${SEED}" \
  --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output-root "${RUN_ROOT}" \
  --epochs "${EPOCHS}" --train-batch-size 64 --eval-batch-size 64 --workers 2 \
  --source-learning-rate 0.05 --source-reference-batch-size 256 \
  --weight-decay 0 --scheduler-mode cosine --scheduler-min-lr-ratio 0 \
  --scheduler-warmup-ratio 0 --temperature 1 --threshold 0.5 \
  --training-loss-mode legacy_full_zero --loss-routing joint_bce \
  --supervised-loss-scale "${LOSS_SCALE}" \
  --input-mode full --input-normalization clip --train-crop-scale 0.05 1.0 \
  --full-crop-mode legacy --person-transform-mode letterbox --person-crop-margin 0.15 \
  --person-color-jitter-strength 0 --person-color-jitter-probability 0 \
  --selector-mode shared --prompt-mode shared --num-selectors 10 \
  --view-fusion "${VIEW_FUSION}" --view-classifier-mode shared_post_fusion \
  --view-auxiliary-loss-weight "${AUX_WEIGHT}" --view-gradient-routing joint \
  --adapter-mode disabled --adapter-learning-rate 0.0004 --adapter-weight-decay 0 \
  --parax-mode "${PARAX_MODE}" --parax-layer-indices 10 --parax-rank 32 \
  --parax-num-experts 3 --parax-router-hidden 16 --parax-initialization official \
  --parax-residual-scale 0.1 --parax-output-scale-mode learnable \
  --parax-trainable-components all --parax-residual-ratio-cap 0 \
  --reporting-split val --save-evaluation-scores --save-compact-checkpoints \
  --no-save-checkpoints --evaluation-score-purpose validation_search \
  "${extra_args[@]}" 2>&1 | tee "${LOG_PATH}"
