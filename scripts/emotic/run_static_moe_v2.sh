#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
METHOD="${METHOD:?Set METHOD}"
RUN_ROOT="${RUN_ROOT:?Set fresh RUN_ROOT}"
LOG_PATH="${LOG_PATH:?Set LOG_PATH}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
GPU="${GPU:-0}"
RESIDUAL_BOUND_RATIO="${RESIDUAL_BOUND_RATIO:-0.1}"
DATA_ROOT="${DATA_ROOT:-../multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}"
adapter=image_token
routing=disabled
case "${METHOD}" in
  BASELINE) adapter=disabled ;;
  SHARED) ;;
  MOE_OVERLAP) routing=overlap ;;
  *) echo "Unknown static MoE v2 method" >&2; exit 2 ;;
esac
extra=()
if [[ -n "${SMOKE_UPDATES:-}" ]]; then
  extra+=(--optimizer-updates-per-task "${SMOKE_UPDATES}" --static-moe-smoke-samples 64)
fi
[[ ! -e "${RUN_ROOT}" ]] || { echo "Output exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean worktree required" >&2; exit 2; }
for path in "${DATA_ROOT}/CVPR17_Annotations.mat" "${CLIP_CHECKPOINT}"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
mkdir -p "$(dirname "${LOG_PATH}")"
export OMP_NUM_THREADS=1
echo "STATIC_MOE_V2 method=${METHOD} seed0 tasks8 epochs30 batch64 layer1 scale0.03 residual_bound_ratio=${RESIDUAL_BOUND_RATIO} rho=0.1 fixed_fusion no_ParaX"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m multi_lane.track_a.runner \
  --training-protocol incremental --static-moe-paired-audit --static-moe-mode "${routing}" \
  --max-tasks 8 --seed 0 --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" --output-root "${RUN_ROOT}" \
  --epochs 30 --train-batch-size 64 --eval-batch-size 64 --workers 2 \
  --amp-initial-scale 1024 --amp-growth-interval 1000000000 \
  --source-learning-rate 0.05 --source-reference-batch-size 256 --weight-decay 0 \
  --scheduler-mode cosine --scheduler-min-lr-ratio 0 --scheduler-warmup-ratio 0 \
  --temperature 1 --threshold 0.5 --training-loss-mode legacy_full_zero --loss-routing joint_bce \
  --supervised-loss-scale 1 --input-mode full --input-normalization clip --train-crop-scale 0.05 1.0 \
  --full-crop-mode legacy --person-transform-mode letterbox --person-crop-margin 0.15 \
  --person-color-jitter-strength 0 --person-color-jitter-probability 0 \
  --selector-mode shared --prompt-mode shared --num-selectors 10 \
  --view-fusion fixed_three_view --view-classifier-mode shared_post_fusion \
  --view-auxiliary-loss-weight 0.1 --view-gradient-routing joint \
  --adapter-mode "${adapter}" --adapter-bottleneck-dim 48 --adapter-layer-indices 1 \
  --adapter-residual-scale 0.03 --adapter-residual-bound-ratio "${RESIDUAL_BOUND_RATIO}" \
  --adapter-activation relu --adapter-task-init independent --adapter-residual-gate-mode fixed \
  --adapter-learning-rate 0.0004 --adapter-weight-decay 0 --adapter-regularization none \
  --parax-mode disabled --skip-validation-eval --reporting-split val \
  --save-evaluation-scores --save-view-evaluation-scores --view-evaluation-diagnostics \
  --save-compact-checkpoints --no-save-checkpoints --evaluation-score-purpose validation_search \
  "${extra[@]}" 2>&1 | tee "${LOG_PATH}"
