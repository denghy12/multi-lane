#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
METHOD="${METHOD:?METHOD is required (G0, G1, or G2)}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_shared_dgl}"

case "${METHOD}" in
  G0) gradient_routing=joint; auxiliary_weight=0.1 ;;
  G1) gradient_routing=fusion_detach; auxiliary_weight=0.1 ;;
  G2) gradient_routing=dgl; auxiliary_weight=0 ;;
  *) echo "METHOD must be G0, G1, or G2" >&2; exit 2 ;;
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

echo "Shared multi-view DGL validation: method=${METHOD} routing=${gradient_routing} seed=0 GPU=${GPU} EMOTIC 8tasks 30epochs/task batch64 mainLR0.0125 cosine_min0_nowarmup Full_legacy+Person_margin15_letterbox+reliable_Face_letterbox shared_selectors_prompts_head_image_token_adapter fixed_R1_feature_weights mainBCE adapterASL9.8/0/0.05 layer1_b32_adapterLR4e-4_scale0.1_relu_independent dgl_alpha1 aux=${auxiliary_weight} gradient_audit=first_batch_each_task view_diagnostics=all_tasks AMP_TF32 val_only no_checkpoint test_forbidden"

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
  --input-mode full \
  --input-normalization clip \
  --train-crop-scale 0.05 1.0 \
  --full-crop-mode legacy \
  --person-crop-margin 0.15 \
  --person-transform-mode letterbox \
  --person-color-jitter-strength 0 \
  --person-color-jitter-probability 0 \
  --view-fusion fixed_three_view \
  --view-fusion-hidden-dim 16 \
  --view-fusion-learning-rate 0.0004 \
  --view-auxiliary-loss-weight "${auxiliary_weight}" \
  --view-gradient-routing "${gradient_routing}" \
  --view-dgl-unimodal-weight 1 \
  --view-gradient-audit \
  --view-evaluation-diagnostics \
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

echo "SHARED_DGL_VALIDATION_COMPLETE method=${METHOD} run=${RUN_ROOT}"
