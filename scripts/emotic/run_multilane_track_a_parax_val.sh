#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
GPU="${GPU:?GPU is required}"
METHOD="${METHOD:?METHOD is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
LOG_PATH="${LOG_PATH:?LOG_PATH is required}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
mkdir -p "$(dirname "${LOG_PATH}")"
case "${METHOD}" in
  B0) PARAX_MODE=disabled; PARAX_LAYERS=(10) ;;
  P-post) PARAX_MODE=post; PARAX_LAYERS=(10) ;;
  P-10) PARAX_MODE=image; PARAX_LAYERS=(10) ;;
  P-8:10) PARAX_MODE=image; PARAX_LAYERS=(8 9 10) ;;
  P-8:10-level) PARAX_MODE=image_level; PARAX_LAYERS=(8 9 10) ;;
  Static-control) PARAX_MODE=static; PARAX_LAYERS=(8 9 10) ;;
  *) echo "Unknown METHOD=${METHOD}" >&2; exit 2 ;;
esac
RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
[[ ! -e "${RUN_ROOT}" ]] || { echo "Output already exists: ${RUN_ROOT}" >&2; exit 2; }
for path in "${CLIP_CHECKPOINT}" "${DATA_ROOT}/CVPR17_Annotations.mat" "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" "${FACE_MANIFEST_ROOT}/manifests/val.jsonl"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
precision_args=()
[[ "${NO_AMP:-0}" == 1 ]] && precision_args+=(--no-amp)
echo "ParaX validation method=${METHOD} seed=0 gpu=${GPU} tasks=8 epochs=30 batch=64 main_lr=0.0125 ParaX_lr=0.0004 fixed_three_view aux=0.1 AMP=${NO_AMP:-0} TF32=on reporting=val checkpoint=off test=forbidden parax=${PARAX_MODE}:${PARAX_LAYERS[*]} rank=${PARAX_RANK:-32} init=${PARAX_INITIALIZATION:-official}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
  --seed 0 --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" --output-root "${RUN_ROOT}" \
  --epochs 30 --max-tasks 8 --scheduler-mode cosine --scheduler-min-lr-ratio 0 \
  --scheduler-warmup-ratio 0 --train-batch-size 64 --eval-batch-size 64 --workers 2 \
  --threshold 0.5 --source-learning-rate 0.05 --source-reference-batch-size 256 \
  --weight-decay 0 --temperature 1 --training-loss-mode legacy_full_zero \
  --loss-routing joint_bce --no-save-checkpoints --input-mode full \
  --input-normalization clip --train-crop-scale 0.05 1.0 --full-crop-mode legacy \
  --person-crop-margin 0.15 --person-transform-mode letterbox \
  --person-color-jitter-strength 0 --person-color-jitter-probability 0 \
  --view-fusion fixed_three_view --view-fusion-hidden-dim 16 \
  --view-fusion-learning-rate 0.0004 --view-auxiliary-loss-weight 0.1 \
  --view-gradient-routing joint --view-evaluation-diagnostics \
  --save-evaluation-scores --evaluation-score-purpose validation_search \
  --adapter-mode disabled --parax-mode "${PARAX_MODE}" \
  --parax-rank "${PARAX_RANK:-32}" --parax-num-experts "${PARAX_EXPERTS:-3}" \
  --parax-layer-indices "${PARAX_LAYERS[@]}" \
  --parax-router-hidden "${PARAX_ROUTER_HIDDEN:-16}" \
  --parax-residual-scale "${PARAX_SCALE:-0.1}" \
  --parax-initialization "${PARAX_INITIALIZATION:-official}" \
  --adapter-learning-rate 0.0004 --adapter-weight-decay 0 \
  --selector-mode shared --prompt-mode shared --num-selectors 10 \
  --reporting-split val "${precision_args[@]}" 2>&1 | tee "${LOG_PATH}"
