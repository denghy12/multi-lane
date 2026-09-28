#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
GPU="${GPU:?GPU is required}"
SUITE="${SUITE:?SUITE is required}"
METHOD="${METHOD:?METHOD is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
LOG_PATH="${LOG_PATH:?LOG_PATH is required}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}"

PARAX_MODE=disabled
PARAX_LAYERS=(10)
PARAX_INIT=official
PARAX_COMPONENTS=all
PARAX_SCALE=0.1
PARAX_SCALE_MODE=fixed
PARAX_LEVEL_CONDITIONED=0
PARAX_FREEZE_CENTER=0
PARAX_TASK_LOCAL=0
PARAX_PENALTY=0
PARAX_CAP=0

case "${SUITE}:${METHOD}" in
  level:B0) PARAX_MODE=disabled ;;
  level:P-post) PARAX_MODE=post ;;
  level:P-10) PARAX_MODE=image ;;
  level:P-8:10) PARAX_MODE=image; PARAX_LAYERS=(8 9 10) ;;
  level:P-8:10-level) PARAX_MODE=image_level; PARAX_LAYERS=(8 9 10); PARAX_LEVEL_CONDITIONED=1 ;;
  level:Static-control) PARAX_MODE=static; PARAX_LAYERS=(8 9 10) ;;

  paired:B0-paired) PARAX_MODE=disabled ;;
  paired:P10-identity) PARAX_MODE=image; PARAX_INIT=identity; PARAX_SCALE_MODE=learnable ;;
  paired:P10-small) PARAX_MODE=image; PARAX_INIT=small ;;
  paired:P10-zeroB) PARAX_MODE=image; PARAX_INIT=zero_b ;;

  center:B0) PARAX_MODE=disabled ;;
  center:Shared-live) PARAX_MODE=image; PARAX_INIT=zero_output; PARAX_COMPONENTS=all ;;
  center:Frozen-center) PARAX_MODE=image; PARAX_INIT=zero_output; PARAX_COMPONENTS=router; PARAX_FREEZE_CENTER=1 ;;
  center:Task-local-delta) PARAX_MODE=image; PARAX_INIT=zero_output; PARAX_COMPONENTS=router; PARAX_FREEZE_CENTER=1; PARAX_TASK_LOCAL=1 ;;

  frozen:B0) PARAX_MODE=disabled ;;
  frozen:Frozen-center-small) PARAX_MODE=image; PARAX_INIT=zero_output; PARAX_COMPONENTS=router; PARAX_SCALE=0.001; PARAX_FREEZE_CENTER=1 ;;
  frozen:Frozen-center-penalty) PARAX_MODE=image; PARAX_INIT=zero_output; PARAX_COMPONENTS=router; PARAX_FREEZE_CENTER=1; PARAX_PENALTY=1.0 ;;

  post:B0) PARAX_MODE=disabled ;;
  post:P-post-zero) PARAX_MODE=post; PARAX_INIT=zero_output; PARAX_SCALE=0.0; PARAX_FREEZE_CENTER=1 ;;
  post:P-post-small) PARAX_MODE=post; PARAX_INIT=zero_output; PARAX_SCALE=0.001; PARAX_FREEZE_CENTER=1 ;;
  post:P-post-tiny) PARAX_MODE=post; PARAX_INIT=zero_output; PARAX_SCALE=0.0001; PARAX_FREEZE_CENTER=1 ;;
  post:P-post-static) PARAX_MODE=post_static; PARAX_INIT=zero_output; PARAX_SCALE=0.001; PARAX_FREEZE_CENTER=1 ;;
  *) echo "Unknown SUITE/METHOD=${SUITE}/${METHOD}" >&2; exit 2 ;;
esac

RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
[[ ! -e "${RUN_ROOT}" ]] || { echo "Output already exists: ${RUN_ROOT}" >&2; exit 2; }
for path in \
  "${CLIP_CHECKPOINT}" \
  "${DATA_ROOT}/CVPR17_Annotations.mat" \
  "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" \
  "${FACE_MANIFEST_ROOT}/manifests/val.jsonl" \
  "${FACE_MANIFEST_ROOT}/manifests/test.jsonl"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
mkdir -p "$(dirname "${LOG_PATH}")"

extra_args=()
(( PARAX_LEVEL_CONDITIONED == 1 )) && extra_args+=(--parax-level-conditioned)
(( PARAX_FREEZE_CENTER == 1 )) && extra_args+=(--parax-freeze-center-after-task0)
(( PARAX_TASK_LOCAL == 1 )) && extra_args+=(--parax-task-local-gate)

echo "ParaX test rerun suite=${SUITE} method=${METHOD} seed=0 gpu=${GPU} tasks=8 epochs=30/task batch=64 main_lr=0.0125 parax_lr=0.0004 rank=32 experts=3 router_hidden=16 layers=${PARAX_LAYERS[*]} init=${PARAX_INIT} components=${PARAX_COMPONENTS} scale=${PARAX_SCALE} scale_mode=${PARAX_SCALE_MODE} cap=${PARAX_CAP} penalty=${PARAX_PENALTY} fixed_three_view auxiliary=0.1 reporting=test validation_eval=skipped test_search=off checkpoint=off AMP=on TF32=on"

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
  --save-evaluation-scores --evaluation-score-purpose fixed_test_fusion \
  --adapter-mode disabled --parax-mode "${PARAX_MODE}" \
  --parax-rank 32 --parax-num-experts 3 --parax-layer-indices "${PARAX_LAYERS[@]}" \
  --parax-router-hidden 16 --parax-residual-scale "${PARAX_SCALE}" \
  --parax-initialization "${PARAX_INIT}" --parax-trainable-components "${PARAX_COMPONENTS}" \
  --parax-output-scale-mode "${PARAX_SCALE_MODE}" \
  --parax-residual-ratio-cap "${PARAX_CAP}" \
  --parax-residual-penalty-weight "${PARAX_PENALTY}" \
  --adapter-learning-rate 0.0004 --adapter-weight-decay 0 \
  --selector-mode shared --prompt-mode shared --num-selectors 10 \
  --skip-validation-eval --reporting-split test "${extra_args[@]}" \
  2>&1 | tee "${LOG_PATH}"
