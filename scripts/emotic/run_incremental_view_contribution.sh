#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD="${METHOD:?Set METHOD}"
RUN_ROOT="${RUN_ROOT:?Set fresh RUN_ROOT}"
LOG_PATH="${LOG_PATH:?Set LOG_PATH}"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
DATA_ROOT="${DATA_ROOT:-../multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}"
fusion=disabled
auxiliary=0
loss_scale=1.1
extra=()
case "${METHOD}" in
  FULL_ONLY) ;;
  FULL_PERSON) fusion=fixed_full_person ;;
  FULL_FACE) fusion=fixed_full_face ;;
  THREE_VIEW) fusion=fixed_three_view ;;
  *) echo "Unknown view combination" >&2; exit 2 ;;
esac
if [[ "${fusion}" != disabled ]]; then
  auxiliary=0.1
  loss_scale=1
  extra+=(--face-manifest-root "${FACE_MANIFEST_ROOT}" --view-evaluation-diagnostics --save-view-evaluation-scores)
fi
split=test
purpose=fixed_test_fusion
if [[ -n "${SMOKE_UPDATES:-}" ]]; then
  split=val
  purpose=validation_search
  extra+=(--optimizer-updates-per-task "${SMOKE_UPDATES}")
fi
[[ ! -e "${RUN_ROOT}" ]] || { echo "Output exists: ${RUN_ROOT}" >&2; exit 2; }
mkdir -p "$(dirname "${LOG_PATH}")"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --training-protocol incremental --fixed-view-paired-audit --max-tasks "${MAX_TASKS:-8}" --seed "${SEED}" \
  --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" --output-root "${RUN_ROOT}" \
  --epochs 30 --train-batch-size 64 --eval-batch-size 64 --workers 2 \
  --source-learning-rate 0.05 --source-reference-batch-size 256 --weight-decay 0 \
  --scheduler-mode cosine --scheduler-min-lr-ratio 0 --scheduler-warmup-ratio 0 \
  --amp-initial-scale 1024 --amp-growth-interval 1000000000 --temperature 1 --threshold 0.5 \
  --training-loss-mode legacy_full_zero --loss-routing joint_bce --supervised-loss-scale "${loss_scale}" \
  --input-mode full --input-normalization clip --train-crop-scale 0.05 1.0 --full-crop-mode legacy \
  --person-transform-mode letterbox --person-crop-margin 0.15 \
  --person-color-jitter-strength 0 --person-color-jitter-probability 0 \
  --selector-mode shared --prompt-mode shared --num-selectors 10 --view-classifier-mode shared_post_fusion \
  --view-fusion "${fusion}" --view-auxiliary-loss-weight "${auxiliary}" --view-gradient-routing joint \
  --adapter-mode disabled --parax-mode disabled --adapter-learning-rate 0.0004 --adapter-weight-decay 0 \
  --skip-validation-eval --reporting-split "${split}" --evaluation-score-purpose "${purpose}" \
  --save-evaluation-scores --save-compact-checkpoints --no-save-checkpoints "${extra[@]}" 2>&1 | tee "${LOG_PATH}"
