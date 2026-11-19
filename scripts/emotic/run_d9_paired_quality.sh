#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
phase=${1:?expected smoke, extract, fit-val or test}
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
source_root=../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118/sample_full
cache=../multi-lane-main-d9-disagreement-20261119/output/d9_disagreement/seed1_20261119/cache
cache6=../multi-lane-main-d9-sadness-20261119/output/d9_sadness/seed1_20261119/full_cache
out=output/d9_paired_quality/seed1_20261119
logs=logs/d9_paired_quality/seed1_20261119
mkdir -p "$out" "$logs"
exec 9>"$out/.run.lock"
flock -n 9
exec > >(tee -a "$logs/$phase.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/$phase.exit.status"' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--source "$source_root" --cache "$cache" --cache6 "$cache6")
extract=("$py" -u -m scripts.emotic.d9_quality_extract "${args[@]}" --workers 2)
probe=("$py" -u -m scripts.emotic.d9_quality_run "${args[@]}" --pairs "$out/pairs" --out "$out/readout")
echo "phase=$phase git=$(git rev-parse HEAD) start=$(date -Is)"
case "$phase" in
 smoke)
   export CUDA_VISIBLE_DEVICES=0
   "$py" -m unittest discover -s tests -p 'test_d9_quality_extract.py' -v
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
   "$py" -m unittest discover -s tests -p 'test_d9_quality*.py' -v
   "${probe[@]}" --phase fit-val
   ;;
 test)
   test "$(cat "$out/fit-val.exit.status")" = 0
   export CUDA_VISIBLE_DEVICES=""
   "${probe[@]}" --phase test
   ;;
 *) exit 2 ;;
esac
