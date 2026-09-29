#!/usr/bin/env bash
set -euo pipefail

# Paired test-only comparison: the stable shared b32 Image-token Adapter alone
# versus the same model with a small P-10 ParaX image-stream residual.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

METHOD="${METHOD:?Set METHOD to B32 or B32_P10}"
RUN_ID="${RUN_ID:?Set a unique paired batch ID}"
GPU="${GPU:?Set the CUDA GPU index}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-1}"
DATA_ROOT="${DATA_ROOT:-./datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-./models/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-./output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/emotic_track_a_b32_parax_p10_test}"
LOG_DIR="${LOG_DIR:-./logs/emotic_track_a_b32_parax_p10_test}"

case "${METHOD}" in
  B32) PARAX_MODE=disabled ;;
  B32_P10) PARAX_MODE=image ;;
  *) echo "Unknown METHOD=${METHOD}; use B32 or B32_P10" >&2; exit 2 ;;
esac

RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}/${METHOD}"
LOG_PATH="${LOG_DIR}/${RUN_ID}/${METHOD}.log"
[[ ! -e "${RUN_ROOT}" ]] || { echo "Output already exists: ${RUN_ROOT}" >&2; exit 2; }
for path in \
  "${CLIP_CHECKPOINT}" \
  "${DATA_ROOT}/CVPR17_Annotations.mat" \
  "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" \
  "${FACE_MANIFEST_ROOT}/manifests/test.jsonl"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
mkdir -p "$(dirname "${LOG_PATH}")"

parax_args=()
if [[ "${PARAX_MODE}" == image ]]; then
  parax_args+=(--parax-freeze-center-after-task0)
fi

echo "B32+ParaX paired test method=${METHOD} seed=${SEED} gpu=${GPU} tasks=8 epochs=30/task batch=64 main_lr=0.0125 adapter/parax_lr=0.0004 optimizer=Adam scheduler=cosine b32_layer=1 b32_scale=0.03 parax=P-10_rank32_3experts_zero_output_fixed0.001_task0_center_then_frozen fixed_three_view auxiliary=0.1 AMP=on TF32=on reporting=test validation_eval=skipped checkpoint=off"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed "${SEED}" --data-root "${DATA_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" --output-root "${RUN_ROOT}" \
  --epochs 30 --max-tasks 8 --scheduler-mode cosine \
  --scheduler-min-lr-ratio 0 --scheduler-warmup-ratio 0 \
  --train-batch-size 64 --eval-batch-size 64 --workers 2 \
  --threshold 0.5 --source-learning-rate 0.05 \
  --source-reference-batch-size 256 --weight-decay 0 --temperature 1 \
  --training-loss-mode legacy_full_zero --loss-routing adapter_asl \
  --asl-gamma-neg 9.8 --asl-gamma-pos 0 --asl-clip 0.05 --asl-eps 1e-8 \
  --no-save-checkpoints --input-mode full --input-normalization clip \
  --train-crop-scale 0.05 1.0 --full-crop-mode legacy \
  --person-crop-margin 0.15 --person-transform-mode letterbox \
  --person-color-jitter-strength 0 --person-color-jitter-probability 0 \
  --view-fusion fixed_three_view --view-classifier-mode shared_post_fusion \
  --view-fusion-hidden-dim 16 --view-fusion-learning-rate 0.0004 \
  --view-auxiliary-loss-weight 0.1 --view-gradient-routing joint \
  --view-evaluation-diagnostics --save-evaluation-scores \
  --evaluation-score-purpose fixed_test_fusion \
  --adapter-mode image_token --adapter-view-mode shared \
  --adapter-view-bottleneck-dim 0 --adapter-bottleneck-dim 32 \
  --adapter-layer-indices 1 --adapter-residual-scale 0.03 \
  --adapter-residual-gate-mode fixed --adapter-activation relu \
  --adapter-task-init independent --adapter-regularization none \
  --adapter-learning-rate 0.0004 --adapter-weight-decay 0 \
  --parax-mode "${PARAX_MODE}" --parax-layer-indices 10 \
  --parax-rank 32 --parax-num-experts 3 --parax-router-hidden 16 \
  --parax-initialization zero_output --parax-residual-scale 0.001 \
  --parax-output-scale-mode fixed --parax-trainable-components router \
  --parax-residual-ratio-cap 0 "${parax_args[@]}" \
  --selector-mode shared --prompt-mode shared --num-selectors 10 \
  --skip-validation-eval --reporting-split test 2>&1 | tee "${LOG_PATH}"
