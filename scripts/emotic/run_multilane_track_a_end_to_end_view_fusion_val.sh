#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
METHOD="${METHOD:?METHOD is required (J0, J1, or J3)}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_end_to_end_view_fusion}"

case "${METHOD}" in
  J0) fusion_mode=fixed_three_view; needs_face=1 ;;
  J1) fusion_mode=soft_three_view; needs_face=1 ;;
  J3) fusion_mode=soft_full_person; needs_face=0 ;;
  *) echo "METHOD must be J0, J1, or J3" >&2; exit 2 ;;
esac

RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations" >&2; exit 2; }
if [[ "${needs_face}" == 1 ]]; then
  [[ -f "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" ]] || { echo "Missing train Face manifest" >&2; exit 2; }
  [[ -f "${FACE_MANIFEST_ROOT}/manifests/val.jsonl" ]] || { echo "Missing val Face manifest" >&2; exit 2; }
fi
[[ -z "$(git status --porcelain)" ]] || { echo "Validation requires a clean Git worktree" >&2; exit 2; }
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

face_args=()
if [[ "${needs_face}" == 1 ]]; then
  face_args=(--face-manifest-root "${FACE_MANIFEST_ROOT}")
fi

echo "End-to-end fusion validation: method=${METHOD} mode=${fusion_mode} seed=0 gpu=${GPU} dataset=EMOTIC tasks=8 epochs=30 batch=64 main_lr=0.0125 cosine_min0_nowarmup input=full_legacy+person_margin15_letterbox$([[ ${needs_face} == 1 ]] && echo '+face_reliable_letterbox') normalization=clip layer=1 b32 adapter_lr=4e-4 scale0.1 relu independent main=BCE adapter=ASL9.8/0/0.05 fusion_hidden=16 fusion_lr=4e-4 aux=0.1 AMP/TF32=on reporting=val checkpoint=off test=forbidden"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed 0 \
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
  --view-fusion "${fusion_mode}" \
  --view-fusion-hidden-dim 16 \
  --view-fusion-learning-rate 0.0004 \
  --view-auxiliary-loss-weight 0.1 \
  "${face_args[@]}" \
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

echo "END_TO_END_VIEW_FUSION_VALIDATION_COMPLETE method=${METHOD} run=${RUN_ROOT}"
