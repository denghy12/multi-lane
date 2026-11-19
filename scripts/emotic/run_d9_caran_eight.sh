#!/usr/bin/env bash
set -euo pipefail
phase=${1:?use smoke, train-val-features, fit-val, test-features or test}
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
source_run=../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118/sample_holdout
prior=../multi-lane-main-d9-task10-probe-20261118/output/d9_task10_probe/seed1_20261118
prior_a=../multi-lane-main-d9-a-eight-20261119/output/d9_a_eight/seed1_20261119
out=output/d9_caran_eight/seed1_20261119
logs=logs/d9_caran_eight/seed1_20261119
mkdir -p "$out" "$logs"
exec 9>"$out/.$phase.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/$phase.log") 2>&1
trap 'status=$?; echo "$status" > "$out/$phase.exit.status"' EXIT
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
echo "phase=$phase git=$(git rev-parse HEAD) start=$(date -Is)"
extract=("$py" -u -m scripts.emotic.extract_d9_caran_eight --source "$source_run" --prior "$prior" --eight "$prior_a" --batch-size 64 --workers 2)
readout=(--source "$source_run" --prior-a "$prior_a" --out "$out")
extract_two() {
    "${extract[@]}" --out "$out" --tasks 0 2 4 6 --splits "$@" > "$logs/$phase-even.log" 2>&1 &
    local even_pid=$!
    "${extract[@]}" --out "$out" --tasks 1 3 5 7 --splits "$@" > "$logs/$phase-odd.log" 2>&1 &
    local odd_pid=$!
    local result=0
    wait "$even_pid" || result=1
    wait "$odd_pid" || result=1
    tail -n 6 "$logs/$phase-even.log" "$logs/$phase-odd.log"
    return "$result"
}
case "$phase" in
    smoke) "${extract[@]}" --out "$out/smoke" --tasks 0 7 --splits train val --smoke ;;
    train-val-features) extract_two train val ;;
    fit-val)
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
        "$py" -u -m scripts.emotic.d9_caran_eight_fit "${readout[@]}"
        "$py" -u -m scripts.emotic.d9_caran_eight_eval "${readout[@]}" --split val ;;
    test-features)
        test -s "$out/validation_seal.json"
        extract_two test ;;
    test)
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
        "$py" -u -m scripts.emotic.d9_caran_eight_eval "${readout[@]}" --split test ;;
    *) exit 2 ;;
esac
