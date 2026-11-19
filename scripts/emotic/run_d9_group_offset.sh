#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
phase=${1:?expected smoke, fit, val or test}
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
src=../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118/sample_holdout
previous=../multi-lane-main-d9-paired-a-eight-20261119/output/d9_paired_a_eight/seed1_20261119
out=output/d9_group_offset/seed1_20261119
logs=logs/d9_group_offset/seed1_20261119
mkdir -p "$out" "$logs"
exec 9>"$out/.phase.lock"
flock -n 9
exec > >(tee -a "$logs/$phase.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/$phase.exit.status"' EXIT
export CUDA_VISIBLE_DEVICES="" MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=1
echo "phase=$phase git=$(git rev-parse HEAD) start=$(date -Is)"
shared=(--source "$src" --previous "$previous" --out "$out")
case "$phase" in
  smoke) "$py" -m unittest discover -s tests -p 'test_d9_group_offset*.py' -v ;;
  fit)
    test "$(cat "$out/smoke.exit.status")" = 0
    "$py" -u -m scripts.emotic.d9_group_offset_fit "${shared[@]}" ;;
  val)
    test "$(cat "$out/fit.exit.status")" = 0
    "$py" -u -m scripts.emotic.d9_group_offset_eval "${shared[@]}" --split val ;;
  test)
    test "$(cat "$out/val.exit.status")" = 0
    "$py" -u -m scripts.emotic.d9_group_offset_eval "${shared[@]}" --split test ;;
  *) exit 2 ;;
esac
