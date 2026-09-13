#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
SEED="${SEED:?SEED is required}"
METHOD="${METHOD:?METHOD is required (J0 or A2)}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_locked_architecture_test}"
EPOCHS="${EPOCHS:-30}"
MAX_TASKS="${MAX_TASKS:-8}"

case "${METHOD}" in
  J0) fusion_mode=fixed_three_view; auxiliary=0.1; residual=0.1 ;;
  A2) fusion_mode=residual_three_view; auxiliary=0; residual=0.1 ;;
  *) echo "METHOD must be J0 or A2" >&2; exit 2 ;;
esac

RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"
[[ "${SEED}" =~ ^[012]$ ]] || { echo "SEED must be 0, 1, or 2" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations" >&2; exit 2; }
for split in train test; do
  [[ -f "${FACE_MANIFEST_ROOT}/manifests/${split}.jsonl" ]] || { echo "Missing ${split} Face manifest" >&2; exit 2; }
done
[[ -z "$(git status --porcelain)" ]] || { echo "Locked test requires a clean Git worktree" >&2; exit 2; }
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

echo "Locked architecture test: method=${METHOD} mode=${fusion_mode} seed=${SEED} gpu=${GPU} dataset=EMOTIC tasks=${MAX_TASKS} epochs=${EPOCHS} batch=64 main_lr=0.0125 cosine_min0_nowarmup full_legacy+person_margin15_letterbox+face_reliable_letterbox normalization=clip layer=1 b32 adapter_lr=4e-4 scale0.1 relu independent main=BCE adapter=ASL9.8/0/0.05 view_hidden=16 view_lr=4e-4 view_aux=${auxiliary} view_residual=${residual} AMP/TF32=on reporting=test score_dump=on checkpoint=off test_search=forbidden"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed "${SEED}" \
  --data-root "${DATA_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output-root "${RUN_ROOT}" \
  --epochs "${EPOCHS}" \
  --scheduler-mode cosine \
  --scheduler-min-lr-ratio 0 \
  --scheduler-warmup-ratio 0 \
  --train-batch-size 64 \
  --eval-batch-size 64 \
  --workers 2 \
  --threshold 0.5 \
  --source-learning-rate 0.05 \
  --source-reference-batch-size 256 \
  --weight-decay 0 \
  --temperature 1 \
  --training-loss-mode legacy_full_zero \
  --loss-routing adapter_asl \
  --asl-gamma-neg 9.8 \
  --asl-gamma-pos 0 \
  --asl-clip 0.05 \
  --asl-eps 1e-8 \
  --no-save-checkpoints \
  --input-mode full \
  --input-normalization clip \
  --train-crop-scale 0.05 1.0 \
  --full-crop-mode legacy \
  --person-crop-margin 0.15 \
  --person-transform-mode letterbox \
  --person-color-jitter-strength 0 \
  --person-color-jitter-probability 0 \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --view-fusion "${fusion_mode}" \
  --view-fusion-hidden-dim 16 \
  --view-fusion-learning-rate 0.0004 \
  --view-residual-scale "${residual}" \
  --view-auxiliary-loss-weight "${auxiliary}" \
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
  --save-evaluation-scores \
  --evaluation-score-purpose fixed_test_fusion \
  --max-tasks "${MAX_TASKS}" \
  --reporting-split test \
  2>&1 | tee "${LOG_PATH}"

echo "LOCKED_ARCHITECTURE_TEST_COMPLETE method=${METHOD} seed=${SEED} run=${RUN_ROOT}"
