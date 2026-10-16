#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
GPU="${GPU:-0}"
ROUTE_MODE="${ROUTE_MODE:-dynamic}"
RUN_ID="${RUN_ID:-moe_task_ffn_v3_${ROUTE_MODE}_seed0_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-../multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-../CODE_DDP/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}"
RUN_ROOT="${ROOT}/output/moe_task_ffn_v3/${RUN_ID}"
LOG_ROOT="${ROOT}/logs/moe_task_ffn_v3/${RUN_ID}"
case "${ROUTE_MODE}" in dynamic|shared_fixed|view_fixed) ;; *) exit 2 ;; esac
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
echo "GPU=${GPU} method=moe_task_ffn_v3 route=${ROUTE_MODE} seed=0 tasks=${task_args[*]} epochs=30 batch=64 experts=4 shared=0.5 specialist_topk=2 bottleneck=64 layer=1 residual=0.1 router_scale=2 temperature=1 noise=0 balance_weight=0.001"
echo "data=${DATA_ROOT} clip=${CLIP_CHECKPOINT} face=${FACE_MANIFEST_ROOT} output=${RUN_ROOT} logs=${LOG_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m multi_lane.track_a.runner \
  --training-protocol incremental --moe-paired-audit --seed 0 \
  --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" --output-root "${RUN_ROOT}/train" \
  --epochs 30 --train-batch-size 64 --eval-batch-size 64 --workers 2 \
  --amp-initial-scale 1024 --amp-growth-interval 1000000000 \
  --source-learning-rate 0.05 --source-reference-batch-size 256 --weight-decay 0 \
  --scheduler-mode cosine --scheduler-min-lr-ratio 0 --scheduler-warmup-ratio 0 \
  --temperature 1 --threshold 0.5 --training-loss-mode legacy_full_zero \
  --loss-routing joint_bce --supervised-loss-scale 1 \
  --input-mode full --input-normalization clip --train-crop-scale 0.05 1.0 \
  --full-crop-mode legacy --person-transform-mode letterbox --person-crop-margin 0.15 \
  --person-color-jitter-strength 0 --person-color-jitter-probability 0 \
  --selector-mode shared --prompt-mode shared --num-selectors 10 \
  --view-fusion fixed_three_view --view-classifier-mode shared_post_fusion \
  --view-auxiliary-loss-weight 0.1 --view-gradient-routing joint \
  --adapter-mode moe_task_ffn_v3 --adapter-bottleneck-dim 64 \
  --adapter-layer-indices 1 --adapter-residual-scale 0.1 \
  --adapter-learning-rate 0.0004 --adapter-weight-decay 0 \
  --moe-num-experts 4 --moe-top-k 2 --moe-dropout 0.1 \
  --moe-route-mode "${ROUTE_MODE}" --moe-shared-fraction 0.5 \
  --moe-router-logit-scale 2 --moe-router-temperature 1 \
  --moe-noise-std 0 --moe-balance-weight 0.001 \
  --parax-mode disabled --skip-validation-eval --reporting-split val \
  --save-compact-checkpoints --no-save-checkpoints \
  "${task_args[@]}" 2>&1 | tee "${LOG_ROOT}/train.log"

evaluate_split() {
  local split="$1" directory="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m multi_lane.track_a.evaluate_moe_task_ffn \
    --source-run "${RUN_ROOT}/train" --output "${RUN_ROOT}/${directory}" \
    --split "${split}" --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" \
    --face-manifest-root "${FACE_MANIFEST_ROOT}" --batch-size 64 "$@" \
    > "${LOG_ROOT}/${directory}.log" 2>&1
}
if [[ -n "${SMOKE_UPDATES:-}" ]]; then
  evaluate_split val smoke_val --full-cohort --allow-smoke
  evaluate_split val replay_val --allow-smoke
  "${PYTHON}" - "${RUN_ROOT}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
a = json.loads((p / 'train/seed_summary.json').read_text())['task_metrics'][0]['mAP']
b = json.loads((p / 'replay_val/result.json').read_text())['task_metrics'][0]['mAP']
assert abs(a - b) < 1e-7, (a, b)
print(f'CHECKPOINT_REPLAY_PASSED mAP={b:.8f}')
PY
else
  # Two independent processes replay the same eight checkpoints simultaneously.
  evaluate_split val full_val --full-cohort &
  val_pid=$!
  evaluate_split test full_test --full-cohort &
  test_pid=$!
  val_status=0; test_status=0
  wait "${val_pid}" || val_status=$?
  wait "${test_pid}" || test_status=$?
  echo "val_exit=${val_status} test_exit=${test_status}" | tee "${LOG_ROOT}/evaluation_status.txt"
  [[ "${val_status}" == 0 && "${test_status}" == 0 ]] || exit 1
  # Mechanism check uses validation only, after the locked val/test pair.
  if [[ "${ROUTE_MODE}" != shared_fixed ]]; then
    evaluate_split val full_val_specialist_off --full-cohort --disable-specialist-runtime
  fi
fi
echo "MOE_TASK_FFN_V3_COMPLETE run_root=${RUN_ROOT} log_root=${LOG_ROOT}"
