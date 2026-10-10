#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
METHOD="${METHOD:?Set METHOD}"
RUN_ROOT="${RUN_ROOT:?Set fresh RUN_ROOT}"
LOG_PATH="${LOG_PATH:?Set LOG_PATH}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-../multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-30}"
AMP_INITIAL_SCALE="${AMP_INITIAL_SCALE:-1024}"
AMP_GROWTH_INTERVAL="${AMP_GROWTH_INTERVAL:-1000000000}"
PARAX_MODE=disabled
VIEW_FUSION=disabled
AUX_WEIGHT=0
LOSS_SCALE=1.1
extra_args=()
if [[ -z "${SMOKE_UPDATES:-}" ]]; then extra_args+=(--also-report-test); fi
case "${METHOD}" in
  THREE_VIEW) VIEW_FUSION=fixed_three_view ;;
  THREE_VIEW_PARAX) VIEW_FUSION=fixed_three_view; PARAX_MODE=image ;;
  *) echo "Unknown METHOD=${METHOD}" >&2; exit 2 ;;
esac
if [[ "${VIEW_FUSION}" != disabled ]]; then
  AUX_WEIGHT=0.1
  LOSS_SCALE=1
  extra_args+=(--face-manifest-root "${FACE_MANIFEST_ROOT}" --view-evaluation-diagnostics --save-view-evaluation-scores)
fi
if [[ -n "${SMOKE_UPDATES:-}" ]]; then
  extra_args+=(--optimizer-updates-per-task "${SMOKE_UPDATES}" )
fi
for path in "${DATA_ROOT}/CVPR17_Annotations.mat" "${CLIP_CHECKPOINT}"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
if [[ "${VIEW_FUSION}" != disabled ]]; then
  for split in train val; do
    [[ -f "${FACE_MANIFEST_ROOT}/manifests/${split}.jsonl" ]] || {
      echo "Missing Face manifest: ${split}" >&2; exit 2;
    }
  done
fi
[[ ! -e "${RUN_ROOT}" ]] || { echo "Output already exists: ${RUN_ROOT}" >&2; exit 2; }
mkdir -p "$(dirname "${LOG_PATH}")"
echo "INCREMENTAL_TRANSFER method=${METHOD} GPU=${GPU} seed=${SEED} tasks=${MAX_TASKS:-8} epochs=${EPOCHS} original_image_ParaX=${PARAX_MODE} validation+test_at_task_end AMP_scale1024"
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --training-protocol incremental --image-stream-transfer-audit --skip-validation-eval --max-tasks "${MAX_TASKS:-8}" --seed "${SEED}" \
  --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output-root "${RUN_ROOT}" \
  --epochs "${EPOCHS}" --train-batch-size 64 --eval-batch-size 64 --workers 2 \
  --amp-initial-scale "${AMP_INITIAL_SCALE}" --amp-growth-interval "${AMP_GROWTH_INTERVAL}" \
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
