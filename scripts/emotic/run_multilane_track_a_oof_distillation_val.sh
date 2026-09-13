#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
VARIANT="${VARIANT:?VARIANT must be D0, D1, D2, or D3}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
OOF_ROOT="${OOF_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_three_view_oof_v0.1/three_view_oof_seed0_20260910_160250/oof_sources}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_oof_distillation}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"

[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "OOF distillation requires a clean worktree" >&2; exit 2; }
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

common=(
  --seed 0 --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}"
  --output-root "${RUN_ROOT}" --epochs 30 --train-batch-size 64
  --eval-batch-size 64 --workers 2 --threshold 0.5 --max-tasks 8
)

echo "OOF distillation ${VARIANT}: seed0 EMOTIC 8tasks 30epochs/task batch64 Full legacy_crop CLIP_norm mainLR0.0125 Image-token Adapter layer1/b32/LR4e-4/scale0.1/ReLU/independent mainBCE AdapterASL9.8/0/0.05 cosine_min0_nowarmup AMP_TF32 val_scores no_checkpoint test_forbidden"
case "${VARIANT}" in
  D0)
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.runner \
      "${common[@]}" --device cuda --source-learning-rate 0.05 \
      --source-reference-batch-size 256 --weight-decay 0 --temperature 1 \
      --training-loss-mode legacy_full_zero --loss-routing adapter_asl \
      --asl-gamma-neg 9.8 --asl-gamma-pos 0 --asl-clip 0.05 --asl-eps 1e-8 \
      --scheduler-mode cosine --scheduler-min-lr-ratio 0 --scheduler-warmup-ratio 0 \
      --no-save-checkpoints --input-mode full --save-evaluation-scores \
      --evaluation-score-purpose validation_search --input-normalization clip \
      --train-crop-scale 0.05 1.0 --adapter-mode image_token \
      --adapter-bottleneck-dim 32 --adapter-layer-indices 1 \
      --adapter-residual-scale 0.1 --adapter-residual-gate-mode fixed \
      --adapter-activation relu --adapter-learning-rate 0.0004 \
      --adapter-weight-decay 0 --adapter-task-init independent \
      --adapter-regularization none --reporting-split val \
      2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
    ;;
  D1|D2|D3)
    mode=person
    [[ "${VARIANT}" == D2 ]] && mode=person_face
    [[ "${VARIANT}" == D3 ]] && mode=r1
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
      -m multi_lane.track_a.oof_distillation_runner \
      "${common[@]}" --mode "${mode}" --oof-root "${OOF_ROOT}" \
      --face-manifest-root "${FACE_MANIFEST_ROOT}" \
      --learning-rate 0.0125 --adapter-learning-rate 0.0004 \
      --distillation-mix 0.20 \
      2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
    ;;
  *) echo "Unknown VARIANT=${VARIANT}" >&2; exit 2 ;;
esac

echo "OOF_DISTILLATION_RUN_COMPLETE variant=${VARIANT} run=${RUN_ROOT}"
