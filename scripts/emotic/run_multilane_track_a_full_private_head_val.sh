#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
METHOD="${METHOD:?METHOD is required (H0 or HF)}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_full_private_head}"

case "${METHOD}" in
  H0) classifier_mode=shared_per_view ;;
  HF) classifier_mode=full_private_per_view ;;
  *) echo "METHOD must be H0 or HF" >&2; exit 2 ;;
esac

RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations" >&2; exit 2; }
[[ -f "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" ]] || { echo "Missing train Face manifest" >&2; exit 2; }
[[ -f "${FACE_MANIFEST_ROOT}/manifests/val.jsonl" ]] || { echo "Missing val Face manifest" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Validation requires a clean Git worktree" >&2; exit 2; }
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

echo "Full-head diagnostic: method=${METHOD} classifier=${classifier_mode} seed0 EMOTIC B5-C3 8tasks 30epochs/task train/eval_batch64 workers2 frozen_CLIP_ViT-B16 shared_Selector_Prompt_Adapter_b32 layer1 scale0.1 relu independent mainLR0.0125 adapterLR4e-4 Adam_reset_per_task cosine_min0_nowarmup_noWD mainBCE adapterASL9.8/0/0.05 fixed_per_view_logit_fusion reliable=0.64/0.16/0.20 fallback=0.8/0.2/0 joint_gradient aux0.1 audit=epochs0,14,29_first3batches AMP_TF32 val_only scores compact_states no_full_checkpoints test_forbidden"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed 0 \
  --data-root "${DATA_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" \
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
  --weight-decay 0 \
  --temperature 1 \
  --training-loss-mode legacy_full_zero \
  --loss-routing adapter_asl \
  --asl-gamma-neg 9.8 \
  --asl-gamma-pos 0 \
  --asl-clip 0.05 \
  --asl-eps 1e-8 \
  --no-save-checkpoints \
  --save-compact-checkpoints \
  --input-mode full \
  --input-normalization clip \
  --train-crop-scale 0.05 1.0 \
  --full-crop-mode legacy \
  --person-crop-margin 0.15 \
  --person-transform-mode letterbox \
  --person-color-jitter-strength 0 \
  --person-color-jitter-probability 0 \
  --view-fusion fixed_three_view \
  --view-classifier-mode "${classifier_mode}" \
  --view-fusion-hidden-dim 16 \
  --view-fusion-learning-rate 0.0004 \
  --view-auxiliary-loss-weight 0.1 \
  --view-gradient-routing joint \
  --view-gradient-audit \
  --view-path-gradient-audit-epochs 0 14 29 \
  --view-path-gradient-audit-batches 3 \
  --view-evaluation-diagnostics \
  --save-evaluation-scores \
  --save-view-evaluation-scores \
  --evaluation-score-purpose validation_search \
  --adapter-mode image_token \
  --adapter-bottleneck-dim 32 \
  --adapter-view-bottleneck-dim 0 \
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

echo "FULL_PRIVATE_HEAD_RUN_COMPLETE method=${METHOD} run=${RUN_ROOT}"
