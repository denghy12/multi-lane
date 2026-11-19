#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
phase=${1:?expected smoke, extract, fit-val or test}
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
source_root=../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118/sample_holdout
prior=../multi-lane-main-d9-task10-probe-20261118/output/d9_task10_probe/seed1_20261118
out=output/d9_matched_source/seed1_20261119
logs=logs/d9_matched_source/seed1_20261119
mkdir -p "$out" "$logs"
exec 9>"$out/.phase.lock"
flock -n 9
exec > >(tee -a "$logs/$phase.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/$phase.exit.status"' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
extract=("$py" -u -m scripts.emotic.d9_matched_extract --source "$source_root" --prior "$prior" --workers 2)
run=("$py" -u -m scripts.emotic.d9_matched_run --source "$source_root" --prior "$prior" --pairs "$out/pairs" --out "$out/readout")
echo "phase=$phase git=$(git rev-parse HEAD) start=$(date -Is)"
case "$phase" in
 smoke)
   export CUDA_VISIBLE_DEVICES=0
   "$py" -m unittest discover -s tests -p 'test_d9_matched_cohort.py' -v
   "$py" -m unittest discover -s tests -p 'test_d9_matched_extract.py' -v
   "${extract[@]}" --tasks 6 --smoke --out "$out/smoke"
   ;;
 extract)
   test "$(cat "$out/smoke.exit.status")" = 0
   export CUDA_VISIBLE_DEVICES=0
   "${extract[@]}" --tasks 0 3 6 --out "$out/pairs"
   ;;
 fit-val)
   test "$(cat "$out/extract.exit.status")" = 0
   export CUDA_VISIBLE_DEVICES=""
   "$py" -m unittest discover -s tests -p 'test_d9_matched*.py' -v
   "${run[@]}" --phase fit-val
   ;;
 test)
   test "$(cat "$out/fit-val.exit.status")" = 0
   export CUDA_VISIBLE_DEVICES=""
   "${run[@]}" --phase test
   ;;
 *) exit 2 ;;
esac
