#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
phase=${1:?use smoke, extract, fit-val or test}
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
src=../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118/sample_holdout
probe=../multi-lane-main-d9-task10-probe-20261118/output/d9_task10_probe/seed1_20261118
prev=../multi-lane-main-d9-a-eight-20261119/output/d9_a_eight/seed1_20261119
matched=../multi-lane-main-d9-matched-source-20261119/output/d9_matched_source/seed1_20261119
out=output/d9_paired_a_eight/seed1_20261119
logs=logs/d9_paired_a_eight/seed1_20261119
mkdir -p "$out" "$logs"
exec 9>"$out/.phase.lock"
flock -n 9
exec > >(tee -a "$logs/$phase.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/$phase.exit.status"' EXIT
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "phase=$phase git=$(git rev-parse HEAD) start=$(date -Is)"
extract=("$py" -u -m scripts.emotic.d9_paired_a_extract --source "$src" --matched "$matched/pairs" --natural "$prev" --prior "$probe" --workers 2)
evaluate=("$py" -u -m scripts.emotic.d9_paired_a_evaluate --source "$src" --prior "$prev" --matched "$matched" --pairs "$out/pairs" --out "$out/readout")
case "$phase" in
    smoke)
        export CUDA_VISIBLE_DEVICES=0
        "$py" -m unittest discover -s tests -p 'test_d9_paired_a*.py' -v
        "${extract[@]}" --smoke --tasks 2 --out "$out/smoke" ;;
    extract)
        test "$(cat "$out/smoke.exit.status")" = 0
        export CUDA_VISIBLE_DEVICES=0
        "${extract[@]}" --out "$out/pairs" ;;
    fit-val)
        test "$(cat "$out/extract.exit.status")" = 0
        export CUDA_VISIBLE_DEVICES=""
        "${evaluate[@]}" --phase fit-val ;;
    test)
        test "$(cat "$out/fit-val.exit.status")" = 0
        export CUDA_VISIBLE_DEVICES=""
        "${evaluate[@]}" --phase test ;;
    *) exit 2 ;;
esac
