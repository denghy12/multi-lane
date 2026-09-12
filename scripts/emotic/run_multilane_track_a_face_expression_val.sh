#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
VARIANT="${VARIANT:?VARIANT is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
EXPECTED_SHA256="${EXPECTED_SHA256:?EXPECTED_SHA256 is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
EXPRESSION_CHECKPOINT="${EXPRESSION_CHECKPOINT:-/mnt/haoyuan/workspace/pretrained/emotiefflib/enet_b0_8_best_afew.pt}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_face_expression}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"

[[ "${VARIANT}" == "projection" || "${VARIANT}" == "bottleneck_adapter" ]] || { echo "Invalid variant" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${EXPRESSION_CHECKPOINT}" ]] || { echo "Missing expression checkpoint" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean worktree required" >&2; exit 2; }
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

echo "Face expression validation: variant=${VARIANT} seed=0 gpu=${GPU} dataset=EMOTIC train=valid_nonambiguous alignment=five_point_similarity_margin0.15 input=224_imagenet encoder=EmotiEffLib_AffectNet_EfficientNetB0_frozen tasks=8 epochs=30 batch=64 headLR=0.0125 Adam cosine_min0_nowarmup adapter=b32_LR4e-4_scale0.1_relu_independent mainBCE_adapterASL9.8_0_0.05 AMP_TF32 val_scores no_checkpoint test_forbidden"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.face_expression_runner \
  --seed 0 --data-root "${DATA_ROOT}" --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --expression-checkpoint "${EXPRESSION_CHECKPOINT}" \
  --expected-checkpoint-sha256 "${EXPECTED_SHA256}" \
  --output-root "${RUN_ROOT}" --variant "${VARIANT}" \
  --epochs 30 --train-batch-size 64 --eval-batch-size 64 --workers 2 \
  --learning-rate 0.0125 --adapter-learning-rate 0.0004 \
  --adapter-bottleneck-dim 32 --adapter-residual-scale 0.1 \
  --adapter-activation relu --face-crop-margin 0.15 --threshold 0.5 \
  --max-tasks 8 2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"

echo "FACE_EXPRESSION_RUN_COMPLETE run=${RUN_ROOT}"
