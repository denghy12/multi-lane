#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

METHOD="${METHOD:?Set METHOD to B0, P10, PROJECTOR_ONLY, P10_PROJECTOR, or P10_PROJECTOR_ALIGN}"
RUN_ID="${RUN_ID:?Set a unique batch ID}"
GPU="${GPU:?Set a CUDA GPU index}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-0}"
MAX_TASKS="${MAX_TASKS:-3}"
EPOCHS="${EPOCHS:-30}"
DUAL_REPORT="${DUAL_REPORT:-0}"
DATA_ROOT="${DATA_ROOT:-./datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-./models/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-./output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/emotic_track_a_parax_projector_val}"
LOG_DIR="${LOG_DIR:-./logs/emotic_track_a_parax_projector_val}"

PARAX_MODE=image
PARAX_SCALE=0.1
SCALE_MODE=learnable
PROJECTOR_DIM=0
ALIGN_WEIGHT=0
case "${METHOD}" in
  B0) PARAX_MODE=disabled ;;
  P10) ;;
  PROJECTOR_ONLY) PARAX_SCALE=0; SCALE_MODE=fixed; PROJECTOR_DIM=64 ;;
  P10_PROJECTOR) PROJECTOR_DIM=64 ;;
  P10_PROJECTOR_ALIGN) PROJECTOR_DIM=64; ALIGN_WEIGHT=0.1 ;;
  *) echo "Unknown METHOD=${METHOD}" >&2; exit 2 ;;
esac

RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}/${METHOD}"
LOG_PATH="${LOG_DIR}/${RUN_ID}/${METHOD}.log"
[[ ! -e "${RUN_ROOT}" ]] || { echo "Output already exists: ${RUN_ROOT}" >&2; exit 2; }
for path in \
  "${CLIP_CHECKPOINT}" \
  "${DATA_ROOT}/CVPR17_Annotations.mat" \
  "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" \
  "${FACE_MANIFEST_ROOT}/manifests/val.jsonl"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
dual_args=()
if [[ "${DUAL_REPORT}" == 1 ]]; then
  [[ -f "${FACE_MANIFEST_ROOT}/manifests/test.jsonl" ]] || {
    echo "Missing test manifest" >&2; exit 2;
  }
  dual_args+=(--also-report-test)
fi
mkdir -p "$(dirname "${LOG_PATH}")"

echo "Projector paired run method=${METHOD} seed=${SEED} gpu=${GPU} tasks=0-$((MAX_TASKS-1)) epochs=${EPOCHS}/task batch=64 optimizer=Adam main_lr=0.0125 parax/projector_lr=0.0004 scheduler=cosine ParaX=FrozenForward_after_block11_patch_only_rank32_3experts_shared_live router_hidden16 init=official scale=${PARAX_SCALE} projector_bottleneck=${PROJECTOR_DIM} alignment_weight=${ALIGN_WEIGHT} fixed_three_view auxiliary=0.1 AMP=on TF32=on reporting=val dual_test=${DUAL_REPORT} checkpoint=off"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed "${SEED}" --data-root "${DATA_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" --output-root "${RUN_ROOT}" \
  --epochs "${EPOCHS}" --max-tasks "${MAX_TASKS}" --scheduler-mode cosine \
  --scheduler-min-lr-ratio 0 --scheduler-warmup-ratio 0 \
  --train-batch-size 64 --eval-batch-size 64 --workers 2 \
  --threshold 0.5 --source-learning-rate 0.05 \
  --source-reference-batch-size 256 --weight-decay 0 --temperature 1 \
  --training-loss-mode legacy_full_zero --loss-routing joint_bce \
  --no-save-checkpoints --input-mode full --input-normalization clip \
  --train-crop-scale 0.05 1.0 --full-crop-mode legacy \
  --person-crop-margin 0.15 --person-transform-mode letterbox \
  --person-color-jitter-strength 0 --person-color-jitter-probability 0 \
  --view-fusion fixed_three_view --view-classifier-mode shared_post_fusion \
  --view-fusion-hidden-dim 16 --view-fusion-learning-rate 0.0004 \
  --view-auxiliary-loss-weight 0.1 --view-gradient-routing joint \
  --view-evaluation-diagnostics --save-evaluation-scores \
  --evaluation-score-purpose validation_search \
  --adapter-mode disabled --adapter-learning-rate 0.0004 \
  --adapter-weight-decay 0 --parax-mode "${PARAX_MODE}" \
  --parax-layer-indices 10 --parax-rank 32 --parax-num-experts 3 \
  --parax-router-hidden 16 --parax-initialization official \
  --parax-residual-scale "${PARAX_SCALE}" \
  --parax-output-scale-mode "${SCALE_MODE}" --parax-trainable-components all \
  --parax-residual-ratio-cap 0 \
  --parax-projector-bottleneck-dim "${PROJECTOR_DIM}" \
  --parax-projector-alignment-weight "${ALIGN_WEIGHT}" \
  --selector-mode shared --prompt-mode shared --num-selectors 10 \
  --reporting-split val "${dual_args[@]}" 2>&1 | tee "${LOG_PATH}"
