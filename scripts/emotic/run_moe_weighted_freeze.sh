#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
GPU="${GPU:-1}"
[[ "${GPU}" == 1 && "${SEED:-0}" == 0 ]] || { echo "Locked to GPU1/seed0" >&2; exit 2; }
RUN_ID="${RUN_ID:-formal_seed0_20261018_01}"
DATA_ROOT="${DATA_ROOT:-../multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-../CODE_DDP/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}"
RUN_ROOT="${ROOT}/output/moe_weighted_freeze/${RUN_ID}"
LOG_ROOT="${ROOT}/logs/moe_weighted_freeze/${RUN_ID}"
[[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "Run ID already exists" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean Git worktree required" >&2; exit 2; }
for input in "${DATA_ROOT}/CVPR17_Annotations.mat" "${CLIP_CHECKPOINT}" \
             "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" \
             "${FACE_MANIFEST_ROOT}/manifests/val.jsonl" \
             "${FACE_MANIFEST_ROOT}/manifests/test.jsonl"; do
  [[ -f "${input}" ]] || { echo "Missing input: ${input}" >&2; exit 2; }
done
"${PYTHON}" - <<'GPU_CHECK'
import subprocess
uuid,free,util=(v.strip() for v in subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
if uuid in busy or int(free)<12000 or int(util)>5:
 raise SystemExit('GPU1 must be idle; no other processes will be stopped')
GPU_CHECK
mkdir -p "${LOG_ROOT}"
trap 'run_exit=$?; echo "${run_exit}" > "${LOG_ROOT}/exit_status.txt"' EXIT
export OMP_NUM_THREADS=1
task_args=(--max-tasks 8)
expected_tasks=8
if [[ -n "${SMOKE_UPDATES:-}" ]]; then
  task_args=(--max-tasks 2 --optimizer-updates-per-task "${SMOKE_UPDATES}")
  expected_tasks=2
fi
echo "Gate-mass weighted freeze shared expert pool: GPU=${GPU} seed=0 tasks=${task_args[*]} epochs=30 batch=64 experts=22 topk=2 bottleneck=64 layer=1 scale=0.1"
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
  --adapter-mode moe_task_ffn_activate_freeze --adapter-bottleneck-dim 64 \
  --adapter-layer-indices 1 --adapter-residual-scale 0.1 \
  --adapter-learning-rate 0.0004 --adapter-weight-decay 0 \
  --moe-num-experts 22 --moe-top-k 2 --moe-dropout 0.1 --moe-freeze-metric gate_mass \
  --parax-mode disabled --skip-validation-eval --reporting-split val \
  --save-compact-checkpoints --no-save-checkpoints \
  "${task_args[@]}" 2>&1 | tee "${LOG_ROOT}/train.log"

BASELINE_RUN="${BASELINE_RUN:-../multi-lane-main-moe-task-ffn-v4-20261017/output/moe_task_ffn_v4/formal_baseline_seed0_20261017_01/train}"
"${PYTHON}" -u -m multi_lane.track_a.verify_moe_activate_freeze "${RUN_ROOT}" \
  --expected-tasks "${expected_tasks}" --baseline "${BASELINE_RUN}" \
  > "${LOG_ROOT}/verification.log" 2>&1

evaluate_split() {
  local split="$1" directory="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m multi_lane.track_a.evaluate_moe_task_ffn \
    --source-run "${RUN_ROOT}/train" --output "${RUN_ROOT}/${directory}" \
    --split "${split}" --data-root "${DATA_ROOT}" \
    --clip-checkpoint "${CLIP_CHECKPOINT}" --expected-tasks "${expected_tasks}" \
    --face-manifest-root "${FACE_MANIFEST_ROOT}" --batch-size 64 "$@" \
    > "${LOG_ROOT}/${directory}.log" 2>&1
}
if [[ -n "${SMOKE_UPDATES:-}" ]]; then
  evaluate_split val smoke_val --full-cohort
  evaluate_split val replay_val
  "${PYTHON}" - "${RUN_ROOT}" <<'REPLAY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
a=json.loads((p/'train/seed_summary.json').read_text())['task_metrics']
b=json.loads((p/'replay_val/result.json').read_text())['task_metrics']
assert len(a)==len(b)==2
assert all(abs(x['mAP']-y['mAP'])<1e-7 for x,y in zip(a,b)),(a,b)
print('TWO_TASK_CHECKPOINT_REPLAY_PASSED')
REPLAY
else
  evaluate_split val full_val --full-cohort &
  val_pid=$!
  evaluate_split test full_test --full-cohort &
  test_pid=$!
  val_status=0; test_status=0
  wait "${val_pid}" || val_status=$?
  wait "${test_pid}" || test_status=$?
  echo "val_exit=${val_status} test_exit=${test_status}" > "${LOG_ROOT}/evaluation_status.txt"
  [[ "${val_status}" == 0 && "${test_status}" == 0 ]] || exit 1
fi
echo "WEIGHTED_FREEZE_COMPLETE run_root=${RUN_ROOT} log_root=${LOG_ROOT}"
