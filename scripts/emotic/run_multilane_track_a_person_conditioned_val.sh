#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
MODE="${MODE:?MODE must be disabled, bbox, person, or bbox_person}"
SEED="${SEED:-0}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
REPORTING_SPLIT="${REPORTING_SPLIT:-val}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-./datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:?Set the existing OpenAI ViT-B-16.pt path}"
OUTPUT_BASE="${OUTPUT_BASE:-./output/emotic_track_a_person_conditioned_selector}"
LOG_DIR="${LOG_DIR:-./logs/emotic_track_a_person_conditioned_selector}"

case "${MODE}" in disabled|bbox|person|bbox_person) ;; *) echo "Invalid MODE" >&2; exit 2 ;; esac
case "${REPORTING_SPLIT}" in val) score_purpose=validation_search ;; test) score_purpose=fixed_test_fusion ;; *) echo "Invalid REPORTING_SPLIT" >&2; exit 2 ;; esac
[[ "${SEED}" =~ ^[012]$ ]] || { echo "Invalid SEED" >&2; exit 2; }
[[ "${RUN_ID}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || { echo "Invalid RUN_ID" >&2; exit 2; }
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"
[[ ! -e "${RUN_ROOT}" && ! -e "${LOG_PATH}" ]] || { echo "Run/log already exists" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" && -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || {
  echo "Missing CLIP checkpoint or EMOTIC annotations" >&2; exit 2;
}
[[ -z "$(git status --porcelain)" ]] || { echo "Requires a clean Git worktree" >&2; exit 2; }
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

echo "Person-conditioned Selector run: mode=${MODE} gpu=${GPU} seed=${SEED} dataset=EMOTIC input=${DATA_ROOT} checkpoint=${CLIP_CHECKPOINT} output=${RUN_ROOT} log=${LOG_PATH} tasks=8 epochs=30 batch=64 optimizer=Adam main_lr=0.0125 adapter_lr=0.0004 condition_lr=0.0004 scheduler=cosine min_lr=0 warmup=0 adapter=image_token/layer1/b32/scale0.1/ReLU/independent condition=layer1/hidden32/scale0.1/ReLU/independent_zero_output person_source=frozen_clip_cls full_crop=0.05-1.0 person_margin=0.15 person_letterbox=224 shared_flip=yes person_jitter=0.1@0.2 normalization=clip model_loss=BCE adapter_loss=ASL9.8/0/0.05 AMP=on TF32=on report=${REPORTING_SPLIT} scores=on checkpoint_save=off calibration_fraction=0"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed "${SEED}" --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output-root "${RUN_ROOT}" --epochs 30 --train-batch-size 64 --eval-batch-size 64 \
  --workers 2 --threshold 0.5 --source-learning-rate 0.05 --source-reference-batch-size 256 \
  --weight-decay 0 --temperature 1 --scheduler-mode cosine \
  --scheduler-min-lr-ratio 0 --scheduler-warmup-ratio 0 \
  --training-loss-mode legacy_full_zero --loss-routing adapter_asl \
  --asl-gamma-neg 9.8 --asl-gamma-pos 0 --asl-clip 0.05 --asl-eps 1e-8 \
  --input-mode full --input-normalization clip --train-crop-scale 0.05 1.0 \
  --paired-full-person --person-crop-margin 0.15 --person-transform-mode letterbox \
  --person-color-jitter-strength 0.1 --person-color-jitter-probability 0.2 \
  --selector-conditioning "${MODE}" --selector-condition-layers 1 \
  --selector-condition-hidden-dim 32 --selector-condition-scale 0.1 \
  --selector-condition-learning-rate 0.0004 \
  --adapter-mode image_token --adapter-layer-indices 1 --adapter-bottleneck-dim 32 \
  --adapter-residual-scale 0.1 --adapter-activation relu --adapter-task-init independent \
  --adapter-learning-rate 0.0004 --adapter-weight-decay 0 --adapter-regularization none \
  --max-tasks 8 --reporting-split "${REPORTING_SPLIT}" --save-evaluation-scores \
  --evaluation-score-purpose "${score_purpose}" --no-save-checkpoints \
  2>&1 | tee "${LOG_PATH}"

echo "PERSON_CONDITIONED_RUN_COMPLETE mode=${MODE} split=${REPORTING_SPLIT} run=${RUN_ROOT}"
