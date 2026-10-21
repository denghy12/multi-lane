#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
GPU="${GPU:-0}"
[[ "${GPU}" == 0 && "${SEED:-0}" == 0 ]] || { echo "Cross-view experiment requires GPU0 seed0" >&2; exit 2; }
ARM="${ARM:-person_full}"
case "${ARM}" in
  disabled|person_self|task_full|person_full) MODE="${ARM}" ;;
  *) echo "Unknown arm ${ARM}" >&2; exit 2 ;;
esac
RUN_ID="${RUN_ID:-formal_seed0_20261021_01}"
DATA_ROOT="${DATA_ROOT:-../multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-../CODE_DDP/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}"
RUN_ROOT="${ROOT}/output/person_conditioned_cross_view/${RUN_ID}/${ARM}"
LOG_ROOT="${ROOT}/logs/person_conditioned_cross_view/${RUN_ID}/${ARM}"
[[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "Run ID already exists" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean Git worktree required" >&2; exit 2; }
for input in "${DATA_ROOT}/CVPR17_Annotations.mat" "${CLIP_CHECKPOINT}" \
             "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" \
             "${FACE_MANIFEST_ROOT}/manifests/val.jsonl" \
             "${FACE_MANIFEST_ROOT}/manifests/test.jsonl"; do
  [[ -f "${input}" ]] || { echo "Missing input: ${input}" >&2; exit 2; }
done
mkdir -p "${LOG_ROOT}"
trap 'run_exit=$?; echo "${run_exit}" > "${LOG_ROOT}/exit_status.txt"' EXIT
export OMP_NUM_THREADS=1
task_args=(--max-tasks 8)
expected_tasks=8
if [[ -n "${SMOKE_UPDATES:-}" ]]; then
  task_args=(--max-tasks 2 --optimizer-updates-per-task "${SMOKE_UPDATES}")
  expected_tasks=2
fi
echo "Cross-view: GPU=${GPU} mode=${MODE} seed=0 tasks=${task_args[*]} epochs=30 batch=64 aux=0 module_lr=0.0004 layer=5 width=64 heads=4 scale=0.1"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m multi_lane.track_a.runner \
  --training-protocol incremental --moe-paired-audit \
  --seed 0 --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" --output-root "${RUN_ROOT}/train" \
  --epochs 30 --train-batch-size 64 --eval-batch-size 64 --workers 2 --eval-workers 0 --save-pre-evaluation-recovery \
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
  --view-auxiliary-loss-weight 0 --view-gradient-routing joint \
  --adapter-mode disabled --joint-summary-mode disabled \
  --cross-view-mode "${MODE}" --cross-view-layer 5 --cross-view-dim 64 \
  --cross-view-heads 4 --cross-view-scale 0.1 \
  --view-fusion-learning-rate 0.0004 \
  --parax-mode disabled --skip-validation-eval --reporting-split val \
  --save-compact-checkpoints --no-save-checkpoints \
  "${task_args[@]}" 2>&1 | tee "${LOG_ROOT}/train.log"

BASELINE_RUN="${BASELINE_RUN:-../multi-lane-main-joint-view-summary-20261020/output/joint_view_summary/formal_seed0_20261020_01/baseline_noaux/train}"
"${PYTHON}" -u -m multi_lane.track_a.verify_joint_summary "${RUN_ROOT}" \
  --expected-tasks "${expected_tasks}" --baseline "${BASELINE_RUN}" \
  > "${LOG_ROOT}/verification.log" 2>&1


evaluate_split() {
  local split="$1" directory="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u -m multi_lane.track_a.evaluate_moe_task_ffn \
    --source-run "${RUN_ROOT}/train" --output "${RUN_ROOT}/${directory}" \
    --split "${split}" --data-root "${DATA_ROOT}" \
    --clip-checkpoint "${CLIP_CHECKPOINT}" --expected-tasks "${expected_tasks}" \
    --face-manifest-root "${FACE_MANIFEST_ROOT}" --batch-size 64 --workers 0 "$@" \
    > "${LOG_ROOT}/${directory}.log" 2>&1
}
if [[ -n "${SMOKE_UPDATES:-}" ]]; then
  # Two concurrent validation readers exercise the formal dual-process budget
  # without accessing test during implementation smoke.
  evaluate_split val smoke_val --full-cohort &
  smoke_pid=$!
  evaluate_split val replay_val &
  replay_pid=$!
  smoke_status=0; replay_status=0
  wait "${smoke_pid}" || smoke_status=$?
  wait "${replay_pid}" || replay_status=$?
  [[ "${smoke_status}" == 0 && "${replay_status}" == 0 ]] || exit 1
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
echo "CROSS_VIEW_COMPLETE run_root=${RUN_ROOT} log_root=${LOG_ROOT}"
