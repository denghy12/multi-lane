#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
GPU="${GPU:?GPU is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
EXPECTED_SHA256="${EXPECTED_SHA256:?EXPECTED_SHA256 is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
EXPRESSION_CHECKPOINT="${EXPRESSION_CHECKPOINT:-/mnt/haoyuan/workspace/pretrained/emotiefflib/enet_b0_8_best_afew.pt}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_face_expression_residual}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"

[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root exists" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" && -f "${EXPRESSION_CHECKPOINT}" ]] || { echo "Missing pretrained checkpoint" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean worktree required" >&2; exit 2; }
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

echo "Face expression residual validation: seed0 GPU${GPU} EMOTIC 8tasks 30epochs/task batch64 CLIP_face_legacy_margin15 + frozen_AffectNet_fivepoint_margin15 descriptor=LN_embedding+LN_8logits residual=task_specific_zero_init_rank32_scale0.1_LR4e-4 mainLR0.0125 image_token_adapter_layer1_b32_LR4e-4_scale0.1_relu_independent mainBCE_expressionBCE_adapterASL9.8_0_0.05 Adam cosine_min0_nowarmup AMP_TF32 val_scores no_checkpoint test_forbidden"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.face_expression_residual_runner \
  --seed 0 --data-root "${DATA_ROOT}" --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" --expression-checkpoint "${EXPRESSION_CHECKPOINT}" \
  --expected-expression-sha256 "${EXPECTED_SHA256}" --output-root "${RUN_ROOT}" \
  --epochs 30 --train-batch-size 64 --eval-batch-size 64 --workers 2 \
  --learning-rate 0.0125 --adapter-learning-rate 0.0004 \
  --expression-residual-learning-rate 0.0004 --expression-residual-rank 32 \
  --expression-residual-scale 0.1 --threshold 0.5 --max-tasks 8 \
  2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
