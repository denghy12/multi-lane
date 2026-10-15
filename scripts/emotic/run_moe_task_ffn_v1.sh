#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-moe_task_ffn_v1_seed0_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-../multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}"
RUN_ROOT="${ROOT}/output/moe_task_ffn_v1/${RUN_ID}"
LOG_ROOT="${ROOT}/logs/moe_task_ffn_v1/${RUN_ID}"
[[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "Run ID already exists" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean Git worktree required" >&2; exit 2; }
for input in "${DATA_ROOT}/CVPR17_Annotations.mat" "${CLIP_CHECKPOINT}" \
             "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" \
             "${FACE_MANIFEST_ROOT}/manifests/val.jsonl" \
             "${FACE_MANIFEST_ROOT}/manifests/test.jsonl"; do
  [[ -f "${input}" ]] || { echo "Missing input: ${input}" >&2; exit 2; }
done
mkdir -p "${LOG_ROOT}"
export OMP_NUM_THREADS=1
task_args=(--max-tasks 8)
if [[ -n "${SMOKE_UPDATES:-}" ]]; then
  task_args=(--max-tasks 1 --optimizer-updates-per-task "${SMOKE_UPDATES}")
fi
echo "Task-Forward MoE: GPU=${GPU} seed=0 tasks=${task_args[*]} epochs=30 batch=64 experts=22 topk=2 bottleneck=64 layer=1 scale=0.1"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m multi_lane.track_a.runner \
  --training-protocol incremental --moe-paired-audit \
  --seed 0 --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" --output-root "${RUN_ROOT}/train" \
  --epochs 30 --train-batch-size 64 --eval-batch-size 64 --workers 2 \
  --amp-initial-scale 1024 --amp-growth-interval 1000000000 \
  --source-learning-rate 0.05 --source-reference-batch-size 256 --weight-decay 0 \
  --scheduler-mode cosine --scheduler-min-lr-ratio 0 --scheduler-warmup-ratio 0 \
  --temperature 1 --threshold 0.5 --training-loss-mode legacy_full_zero \
  --loss-routing joint_bce --supervised-loss-scale 1 \
  --input-mode full --input-normalization clip --train-crop-scale 0.05 1.0 \
  --full-crop-mode legacy --person-transform-mode letterbox \
  --person-crop-margin 0.15 --person-color-jitter-strength 0 \
  --person-color-jitter-probability 0 \
  --selector-mode shared --prompt-mode shared --num-selectors 10 \
  --view-fusion fixed_three_view --view-classifier-mode shared_post_fusion \
  --view-auxiliary-loss-weight 0.1 --view-gradient-routing joint \
  --adapter-mode moe_task_ffn --adapter-bottleneck-dim 64 \
  --adapter-layer-indices 1 --adapter-residual-scale 0.1 \
  --adapter-learning-rate 0.0004 --adapter-weight-decay 0 \
  --moe-num-experts 22 --moe-top-k 2 --moe-dropout 0.1 \
  --parax-mode disabled --skip-validation-eval --reporting-split val \
  --save-compact-checkpoints --no-save-checkpoints \
  "${task_args[@]}" 2>&1 | tee "${LOG_ROOT}/train.log"

if [[ -n "${SMOKE_UPDATES:-}" ]]; then
  echo "MOE_TASK_FFN_SMOKE_COMPLETE run_root=${RUN_ROOT}"
  exit 0
fi

# Both readers use the same eight immutable checkpoints. Validation and test
# run concurrently in separate GPU0 processes after all training has stopped.
for split in val test; do
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m multi_lane.track_a.evaluate_moe_task_ffn \
    --source-run "${RUN_ROOT}/train" --output "${RUN_ROOT}/${split}" \
    --split "${split}" --data-root "${DATA_ROOT}" \
    --clip-checkpoint "${CLIP_CHECKPOINT}" \
    --face-manifest-root "${FACE_MANIFEST_ROOT}" --batch-size 32 \
    > "${LOG_ROOT}/${split}.log" 2>&1 &
  if [[ "${split}" == val ]]; then val_pid=$!; else test_pid=$!; fi
done
set +e
wait "${val_pid}"; val_status=$?
wait "${test_pid}"; test_status=$?
set -e
if [[ "${val_status}" -ne 0 || "${test_status}" -ne 0 ]]; then
  echo "Parallel evaluation failed: val=${val_status}, test=${test_status}" >&2
  exit 1
fi
echo "MOE_TASK_FFN_V1_COMPLETE run_root=${RUN_ROOT} log_root=${LOG_ROOT}"
