#!/usr/bin/env bash
set -euo pipefail
phase=${1:?use smoke, extract or evaluate}
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
sample=../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118/sample_holdout
prior=../multi-lane-main-d9-task10-probe-20261118/output/d9_task10_probe/seed1_20261118
out=output/d9_a_eight/seed1_20261119
logs=logs/d9_a_eight/seed1_20261119
mkdir -p "$out" "$logs"
exec 9>"$out/.$phase.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/$phase.log") 2>&1
trap 'status=$?; echo "$status" > "$out/$phase.exit.status"' EXIT
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
echo "phase=$phase git=$(git rev-parse HEAD) time=$(date -Is)"
extract=("$py" -u -m scripts.emotic.extract_d9_task10_probe --source "$sample" --splits train --batch-size 64 --workers 2)
case "$phase" in
    smoke) "${extract[@]}" --tasks 1 --output "$out/smoke" --smoke ;;
    extract) "${extract[@]}" --tasks 1 2 4 5 7 --output "$out" ;;
    evaluate)
        export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
        "$py" -u -m scripts.emotic.d9_a_eight_fit --source "$sample" --prior "$prior" --features "$out" --out "$out"
        for split in val test; do
            "$py" -u -m scripts.emotic.d9_a_eight_eval --source "$sample" --prior "$prior" --out "$out" --split "$split"
        done ;;
    *) exit 2 ;;
esac
