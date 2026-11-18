#!/usr/bin/env bash
set -euo pipefail
phase=${1:?use smoke or formal}
export CUDA_VISIBLE_DEVICES=${2:-0} CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
source_root=${D9_PROBE_SOURCE:-../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118/sample_holdout}
out=output/d9_task10_probe/seed1_20261118
logs=logs/d9_task10_probe/seed1_20261118
tasks=(0 3 6)
steps=300
perms=100
extract_args=()
case "$phase" in
    smoke) out="$out/smoke"; logs="$logs/smoke"; tasks=(0); steps=5; perms=3; extract_args+=(--smoke) ;;
    formal) ;;
    *) exit 2 ;;
esac
mkdir -p "$out" "$logs"
exec 9>"$out/.pipeline.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/run.log") 2>&1
trap 'status=$?; echo "$status" > "$out/exit.status"' EXIT
echo "phase=$phase git=$(git rev-parse HEAD) device=$CUDA_VISIBLE_DEVICES $(date -Is)"
extract=("$py" -u -m scripts.emotic.extract_d9_task10_probe --source "$source_root" --output "$out" --tasks "${tasks[@]}" "${extract_args[@]}")
fit=("$py" -u -m scripts.emotic.fit_d9_task10_probe --root "$out" --tasks "${tasks[@]}" --steps "$steps" --permutations "$perms")
"${extract[@]}" --splits train val
"${fit[@]}" --phase fit-val
"${extract[@]}" --splits test
"${fit[@]}" --phase test
