#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
VIEW="${VIEW:?VIEW must be full, person, or face}"
FOLD="${FOLD:?FOLD must be 0, 1, or 2}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_three_view_oof}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"

[[ "${FOLD}" =~ ^[012]$ ]] || { echo "FOLD must be 0, 1, or 2" >&2; exit 2; }
face_args=()
case "${VIEW}" in
  full)
    input_mode=full; person_margin=0; person_transform=legacy_crop
    jitter_strength=0; jitter_probability=0; crop_min=0.05
    ;;
  person)
    input_mode=person_crop; person_margin=0.15; person_transform=letterbox
    jitter_strength=0.10; jitter_probability=0.20; crop_min=0.70
    ;;
  face)
    input_mode=face_crop; person_margin=0; person_transform=legacy_crop
    jitter_strength=0; jitter_probability=0; crop_min=0.05
    face_args=(--face-manifest-root "${FACE_MANIFEST_ROOT}")
    ;;
  *) echo "Unknown VIEW=${VIEW}" >&2; exit 2 ;;
esac

[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "OOF source requires a clean worktree" >&2; exit 2; }
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

echo "OOF source: view=${VIEW} fold=${FOLD}/3 seed=0 fit=other_two_image_group_folds held_out=one_fold epochs=30 batch=64 main_lr=0.0125 adapter=image_token_layer1_b32_lr4e-4_scale0.1_relu_independent main_loss=bce adapter_loss=asl9.8/0/0.05 amp=on tf32=on reporting=val test=forbidden"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed 0 --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output-root "${RUN_ROOT}" --epochs 30 --scheduler-mode cosine \
  --scheduler-min-lr-ratio 0 --scheduler-warmup-ratio 0 \
  --train-batch-size 64 --eval-batch-size 64 --workers 2 --threshold 0.5 \
  --source-learning-rate 0.05 --source-reference-batch-size 256 --weight-decay 0 \
  --temperature 1 --training-loss-mode legacy_full_zero --loss-routing adapter_asl \
  --asl-gamma-neg 9.8 --asl-gamma-pos 0 --asl-clip 0.05 --asl-eps 1e-8 \
  --no-save-checkpoints --input-mode "${input_mode}" "${face_args[@]}" \
  --person-crop-margin "${person_margin}" --person-transform-mode "${person_transform}" \
  --person-color-jitter-strength "${jitter_strength}" \
  --person-color-jitter-probability "${jitter_probability}" \
  --save-evaluation-scores --evaluation-score-purpose validation_search \
  --crossfit-folds 3 --crossfit-held-out-fold "${FOLD}" --save-calibration-scores \
  --input-normalization clip --train-crop-scale "${crop_min}" 1.0 \
  --adapter-mode image_token --adapter-bottleneck-dim 32 --adapter-layer-indices 1 \
  --adapter-residual-scale 0.1 --adapter-residual-gate-mode fixed \
  --adapter-activation relu --adapter-learning-rate 0.0004 \
  --adapter-weight-decay 0 --adapter-task-init independent \
  --adapter-regularization none --max-tasks 8 --reporting-split val \
  2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"

echo "OOF_SOURCE_COMPLETE view=${VIEW} fold=${FOLD} run=${RUN_ROOT}"
