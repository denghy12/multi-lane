#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
METHOD="${METHOD:?METHOD must be R0 or D1}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_r0_dynamic_test}"
EPOCHS="${EPOCHS:-30}"
MAX_TASKS="${MAX_TASKS:-8}"

case "${METHOD}" in
  R0) fusion_mode=fixed_three_view; auxiliary=0.1 ;;
  D1) fusion_mode=soft_three_view; auxiliary=0.1 ;;
  *) echo "METHOD must be R0 or D1" >&2; exit 2 ;;
esac

RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "GPU must be numeric" >&2; exit 2; }
[[ "${EPOCHS}" =~ ^[0-9]+$ ]] || { echo "EPOCHS must be numeric" >&2; exit 2; }
[[ "${MAX_TASKS}" =~ ^[0-9]+$ ]] || { echo "MAX_TASKS must be numeric" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations" >&2; exit 2; }
for split in train val test; do
  [[ -f "${FACE_MANIFEST_ROOT}/manifests/${split}.jsonl" ]] || {
    echo "Missing ${split} Face manifest" >&2; exit 2;
  }
done
[[ -z "$(git status --porcelain)" ]] || { echo "Test requires a clean Git worktree" >&2; exit 2; }
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

echo "R0/dynamic fusion test: method=${METHOD} mode=${fusion_mode} seed=0 gpu=${GPU} dataset=EMOTIC tasks=${MAX_TASKS} epochs=${EPOCHS} batch=64 main_lr=0.0125 cosine_min0_nowarmup FP32/TF32 shared_selector_prompt_adapter fixed_prior=0.64/0.16/0.20 invalid_face=0.80/0.20/0 view_hidden=16 view_lr=4e-4 aux=${auxiliary} frozen_CLIP checkpoint=off reporting=test score_dump=on test_search=off"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed 0 \
  --data-root "${DATA_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --output-root "${RUN_ROOT}" \
  --epochs "${EPOCHS}" --max-tasks "${MAX_TASKS}" \
  --scheduler-mode cosine --scheduler-min-lr-ratio 0 --scheduler-warmup-ratio 0 \
  --train-batch-size 64 --eval-batch-size 64 --workers 2 --threshold 0.5 \
  --source-learning-rate 0.05 --source-reference-batch-size 256 \
  --weight-decay 0 --temperature 1 \
  --training-loss-mode legacy_full_zero --loss-routing adapter_asl \
  --asl-gamma-neg 9.8 --asl-gamma-pos 0 --asl-clip 0.05 --asl-eps 1e-8 \
  --no-save-checkpoints --no-amp \
  --input-mode full --input-normalization clip \
  --train-crop-scale 0.05 1.0 --full-crop-mode legacy \
  --person-crop-margin 0.15 --person-transform-mode letterbox \
  --person-color-jitter-strength 0 --person-color-jitter-probability 0 \
  --view-fusion "${fusion_mode}" --view-fusion-hidden-dim 16 \
  --view-fusion-learning-rate 0.0004 --view-auxiliary-loss-weight "${auxiliary}" \
  --view-gradient-routing joint --view-evaluation-diagnostics \
  --save-evaluation-scores --evaluation-score-purpose fixed_test_fusion \
  --adapter-mode image_token --adapter-bottleneck-dim 32 \
  --adapter-view-bottleneck-dim 0 --adapter-view-mode shared \
  --adapter-layer-indices 1 --adapter-residual-scale 0.1 \
  --adapter-residual-gate-mode fixed --adapter-activation relu \
  --adapter-learning-rate 0.0004 --adapter-weight-decay 0 \
  --adapter-task-init independent --adapter-regularization none \
  --selector-mode shared --prompt-mode shared --num-selectors 10 \
  --reporting-split test \
  2>&1 | tee "${LOG_PATH}"

echo "R0_DYNAMIC_FUSION_TEST_COMPLETE method=${METHOD} run=${RUN_ROOT}"
