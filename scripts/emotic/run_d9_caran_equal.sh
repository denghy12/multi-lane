#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
phase=${1:?use smoke or formal}
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
source_run=../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119/formal
root=output/d9_caran_equal_updates/seed1_20261123
logs=logs/d9_caran_equal_updates/seed1_20261123
out="$root/$phase"
mkdir -p "$out" "$logs"
exec 9>"$out/.run.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/$phase.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/exit.status.tmp"; mv "$out/exit.status.tmp" "$out/exit.status"' EXIT
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
date -Is
git rev-parse HEAD
git diff --exit-code
extract=("$py" -u -m scripts.emotic.extract_d9_caran_equal --source "$source_run" --out "$out" --workers 2)
extract_two() {
    "${extract[@]}" --tasks 0 2 4 6 --splits "$@" > "$logs/$phase-$1-even.log" 2>&1 &
    local even=$!
    "${extract[@]}" --tasks 1 3 5 7 --splits "$@" > "$logs/$phase-$1-odd.log" 2>&1 &
    local odd=$!
    local rc=0
    wait "$even" || rc=1
    wait "$odd" || rc=1
    tail -n 4 "$logs/$phase-$1-even.log" "$logs/$phase-$1-odd.log"
    return "$rc"
}
case "$phase" in
    smoke)
        "$py" -m unittest discover -s tests -p 'test_d9_caran_eight*.py' -v
        "${extract[@]}" --tasks 0 7 --splits train val --smoke
        ;;
    formal)
        test "$(cat "$root/smoke/exit.status")" = 0
        extract_two train val
        CUDA_VISIBLE_DEVICES="" "$py" -u -m scripts.emotic.d9_caran_equal_fit --source "$source_run" --out "$out"
        CUDA_VISIBLE_DEVICES="" "$py" -u -m scripts.emotic.d9_caran_equal_eval --source "$source_run" --out "$out" --split val
        extract_two test
        CUDA_VISIBLE_DEVICES="" "$py" -u -m scripts.emotic.d9_caran_equal_eval --source "$source_run" --out "$out" --split test
        ;;
    *) exit 2 ;;
esac
date -Is
