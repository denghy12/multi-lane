#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
SEED="${SEED:?SEED is required}"
VIEW="${VIEW:?VIEW must be full or person}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_learned_reliability_gate}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

[[ "${SEED}" =~ ^[012]$ ]] || { echo "SEED must be 0, 1, or 2" >&2; exit 2; }
case "${VIEW}" in
  full)
    input_mode=full
    person_margin=0
    person_transform=legacy_crop
    person_jitter_strength=0
    person_jitter_probability=0
    crop_min=0.05
    ;;
  person)
    input_mode=person_crop
    person_margin=0.15
    person_transform=letterbox
    person_jitter_strength=0.10
    person_jitter_probability=0.20
    crop_min=0.70
    ;;
  *)
    echo "VIEW must be full or person, got ${VIEW}" >&2
    exit 2
    ;;
esac

[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Source run requires a clean Git worktree" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}"

echo "Learned-gate source: run_id=${RUN_ID} view=${VIEW} seed=${SEED} gpu=${GPU} dataset=EMOTIC fit=stable90% calibration=stable10% tasks=8 epochs=30 batch=64 scheduler=cosine eta_min=0 warmup=0 main_lr=0.0125 image_token_layer=1 bottleneck=32 adapter_lr=0.0004 scale=0.1 activation=relu init=independent main_loss=bce adapter_loss=asl9.8/0/0.05 normalization=clip amp=on tf32=on reporting=val val_scores=on calibration_scores=on compact_states=on full_checkpoints=off"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed "${SEED}" \
  --data-root "${DATA_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output-root "${RUN_ROOT}" \
  --epochs 30 \
  --scheduler-mode cosine \
  --scheduler-min-lr-ratio 0 \
  --scheduler-warmup-ratio 0 \
  --train-batch-size 64 \
  --eval-batch-size 64 \
  --workers 2 \
  --threshold 0.5 \
  --source-learning-rate 0.05 \
  --source-reference-batch-size 256 \
  --weight-decay 0.0 \
  --temperature 1.0 \
  --training-loss-mode legacy_full_zero \
  --loss-routing adapter_asl \
  --asl-gamma-neg 9.8 \
  --asl-gamma-pos 0.0 \
  --asl-clip 0.05 \
  --asl-eps 1e-8 \
  --no-save-checkpoints \
  --input-mode "${input_mode}" \
  --person-crop-margin "${person_margin}" \
  --person-transform-mode "${person_transform}" \
  --person-color-jitter-strength "${person_jitter_strength}" \
  --person-color-jitter-probability "${person_jitter_probability}" \
  --save-evaluation-scores \
  --evaluation-score-purpose validation_search \
  --calibration-fraction 0.10 \
  --save-calibration-scores \
  --save-compact-checkpoints \
  --input-normalization clip \
  --train-crop-scale "${crop_min}" 1.0 \
  --adapter-mode image_token \
  --adapter-bottleneck-dim 32 \
  --adapter-layer-indices 1 \
  --adapter-residual-scale 0.1 \
  --adapter-residual-gate-mode fixed \
  --adapter-activation relu \
  --adapter-learning-rate 0.0004 \
  --adapter-weight-decay 0 \
  --adapter-task-init independent \
  --adapter-regularization none \
  --max-tasks 8 \
  --reporting-split val \
  2>&1 | tee "${LOG_PATH}"

echo "LEARNED_GATE_SOURCE_COMPLETE view=${VIEW} seed=${SEED} run_root=${RUN_ROOT} log=${LOG_PATH}"
