#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
METHOD="${METHOD:?Set METHOD}"
GPU="${GPU:?Set GPU}"
RUN_ID="${RUN_ID:?Set RUN_ID}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-0}"
MAX_TASKS="${MAX_TASKS:-3}"
EPOCHS="${EPOCHS:-30}"
DATA_ROOT="${DATA_ROOT:-./datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-./models/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-./output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/emotic_track_a_post_task_router_val}"
LOG_DIR="${LOG_DIR:-./logs/emotic_track_a_post_task_router_val}"

parax_args=()
PARAX_SCALE=0.001
PARAX_BOUND=0
PARAX_LAYER_KEY=10
POST_DESCRIPTION="shared_center_task0_then_frozen task_specific_routers_frozen_after_task"
case "${METHOD}" in
  B0) PARAX_MODE=disabled ;;
  POST_TASK_ROUTER)
    PARAX_MODE=post_task_router
    parax_args+=(--parax-freeze-center-after-task0)
    ;;
  POST_TASK_STAGED|POST_TASK_STAGED_STATIC)
    PARAX_MODE=post_task_staged
    [[ "${METHOD}" == POST_TASK_STAGED_STATIC ]] && PARAX_MODE=post_task_staged_static
    PARAX_SCALE=1.0
    PARAX_BOUND=0.02
    # Staged routing has one logical bank entry after the whole Task Forward;
    # this key does not denote a Transformer insertion layer.
    PARAX_LAYER_KEY=0
    POST_DESCRIPTION="base_first frozen_random_shared_center task_local_zero_projection smooth_ratio_bound=0.02 calibration_epochs=5 consistency=0.1"
    parax_args+=(--parax-calibration-epochs 5 --parax-calibration-consistency-weight 0.1)
    ;;
  *) echo "Unknown METHOD=${METHOD}" >&2; exit 2 ;;
esac
[[ "${EXPORT_VIEW_SCORES:-0}" == 1 ]] && parax_args+=(--save-view-evaluation-scores)
[[ "${SAVE_COMPACT:-0}" == 1 ]] && parax_args+=(--save-compact-checkpoints)
if [[ "${EXPORT_HELDOUT_VIEWS:-0}" == 1 ]]; then
  [[ "${METHOD}" == B0 ]] || { echo "Heldout fusion source must be B0" >&2; exit 2; }
  parax_args+=(--calibration-fraction 0.2 --save-calibration-scores --export-calibration-view-scores)
fi
for path in \
  "${CLIP_CHECKPOINT}" "${DATA_ROOT}/CVPR17_Annotations.mat" \
  "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" \
  "${FACE_MANIFEST_ROOT}/manifests/val.jsonl"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}/${METHOD}"
LOG_PATH="${LOG_DIR}/${RUN_ID}/${METHOD}.log"
[[ ! -e "${RUN_ROOT}" ]] || { echo "Output already exists: ${RUN_ROOT}" >&2; exit 2; }
mkdir -p "$(dirname "${LOG_PATH}")"

echo "Post-Task-Forward ParaX method=${METHOD} seed=${SEED} gpu=${GPU} tasks=0-$((MAX_TASKS-1)) epochs=${EPOCHS}/task batch=64 Adam main_lr=0.0125 ParaX_lr=0.0004 cosine frozen_CLIP ${POST_DESCRIPTION} fixed_scale=${PARAX_SCALE} zero_output rank32 experts3 router_hidden16 fixed_three_view validation_only test_forbidden"
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
  --parax-layer-indices "${PARAX_LAYER_KEY}" --parax-rank 32 --parax-num-experts 3 \
  --parax-router-hidden 16 --parax-initialization zero_output \
  --parax-residual-scale "${PARAX_SCALE}" --parax-output-scale-mode fixed \
  --parax-smooth-ratio-bound "${PARAX_BOUND}" \
  --parax-trainable-components router --parax-residual-ratio-cap 0 \
  "${parax_args[@]}" \
  --selector-mode shared --prompt-mode shared --num-selectors 10 \
  --reporting-split val 2>&1 | tee "${LOG_PATH}"
