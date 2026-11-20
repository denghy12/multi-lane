#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
phase=${1:?specify preflight, smoke, or formal}
export CUDA_VISIBLE_DEVICES=${2:-0} PYTHONHASHSEED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
out=output/d9_split90/seed1_20261120
logs=logs/d9_split90/seed1_20261120
hist=../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118
ref=../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119/formal
mkdir -p "$out" "$logs"
exec 9>"$out/.pipeline.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/$phase.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/$phase.exit.status.tmp"; mv "$out/$phase.exit.status.tmp" "$out/$phase.exit.status"' EXIT
echo "$phase gpu=$CUDA_VISIBLE_DEVICES commit=$(git rev-parse HEAD) $(date -Is)"
args=()
batch=64
last=7
case "$phase" in
  preflight)
    "$py" -m unittest discover -s tests -p 'test_d9_split90*.py' -v
    "$py" -m unittest discover -s tests -p 'test_d9_equal_update*.py' -v
    "$py" -u -m scripts.emotic.d9_split90_roles --out "$out/audit" --seed 1
    cmp "$out/audit/baseline/train_image_roles.json" "$ref/train_image_roles.json"
    exit 0 ;;
  smoke)
    test "$(cat "$out/preflight.exit.status")" = 0
    batch=4; last=1; args=(--smoke) ;;
  formal)
    test "$(cat "$out/smoke.exit.status")" = 0 ;;
  *) exit 2 ;;
esac
train_val() {
  local arm=$1 model="$out/$phase/$1" task
  for ((task=0; task<=last; task++)); do
    if [[ -f "$model/compact_checkpoints/task$task.pth" && $task -lt $last ]]; then continue; fi
    "$py" -u -m scripts.emotic.run_d9_relation_training --mode sample_holdout --protect-branches \
    --output "$model" --workers 2 --micro-batch-size "$batch" --precision bf16 \
    --equal-updates-reference "$hist/sample_full" --equal-updates-holdout "$hist/sample_holdout" \
    --split90 "$arm" --stop-after-task "$task" "${args[@]}" || return 1
  done
  cmp "$out/audit/$arm/train_image_roles.json" "$model/train_image_roles.json" || return 1
  "$py" -u -m scripts.emotic.evaluate_d9_sample_fusion --root "$model" --split val \
    --batch-size "$batch" --workers 2 "${args[@]}" || return 1
  if [[ "$phase" == formal ]]; then
    "$py" -u -m scripts.emotic.d9_split90_compare --root "$model" --reference "$ref" --out "$model/comparison" --split val || return 1
  fi
}
evaluate_test() {
  local arm=$1 model="$out/$phase/$1"
  "$py" -u -m scripts.emotic.evaluate_d9_sample_fusion --root "$model" --split test \
    --batch-size "$batch" --workers 2 "${args[@]}" || return 1
  if [[ "$phase" == formal ]]; then
    "$py" -u -m scripts.emotic.d9_split90_compare --root "$model" --reference "$ref" --out "$model/comparison" --split test || return 1
  fi
}
run_all() {
  local operation=$1 arm i logfile failed=0
  local arms=(hash2 meta online) pids=() retry=()
  for arm in "${arms[@]}"; do
    logfile="$logs/${phase}_${arm}_${operation}.log"
    "$operation" "$arm" > "$logfile" 2>&1 &
    pids+=("$!")
  done
  for i in "${!arms[@]}"; do
    arm=${arms[i]}
    if ! wait "${pids[i]}"; then
      logfile="$logs/${phase}_${arm}_${operation}.log"
      tail -12 "$logfile" >&2
      if grep -Eiq 'CUDA out of memory|OutOfMemoryError' "$logfile"; then retry+=("$arm"); else failed=1; fi
    fi
  done
  for arm in "${retry[@]}"; do
    "$operation" "$arm" >> "$logs/${phase}_${arm}_${operation}.log" 2>&1 || failed=1
  done
  return "$failed"
}
run_all train_val
if [[ "$phase" == formal ]]; then
  "$py" -u -m scripts.emotic.d9_split90_summary --root "$out/formal" --split val
fi
run_all evaluate_test
if [[ "$phase" == formal ]]; then
  "$py" -u -m scripts.emotic.d9_split90_summary --root "$out/formal" --split test
fi
echo "Completed $phase $(date -Is)"
