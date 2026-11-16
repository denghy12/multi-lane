#!/usr/bin/env bash
# Run from the repository root, inside the Linux GPU training environment.
set -euo pipefail
phase=${1:?supply smoke or formal}
device=${2:?supply physical GPU number}
shift 2
if [[ "$phase" != smoke && "$phase" != formal ]] || (( $# == 0 )); then exit 2; fi
export CUDA_VISIBLE_DEVICES="$device"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=${PYTHON:-python}
root=output/d9_relation_logit/seed1_20261116
logs=logs/d9_relation_logit/seed1_20261116
extra=()
micro=${D9_MICRO_BATCH:-64}
workers=${D9_WORKERS:-2}
if [[ "$phase" == smoke ]]; then
    root="$root/smoke"
    logs="$logs/smoke"
    extra+=(--smoke)
    micro=4
fi
mkdir -p "$root" "$logs"
for mode in "$@"; do
    group="$root/$mode"
    (
        echo "Starting $mode ($phase), GPU=$device, git=$(git rev-parse HEAD), microbatch=$micro"
        "$py" -u -m scripts.emotic.run_d9_relation_training --mode "$mode" --output "$group" \
            --micro-batch-size "$micro" --workers "$workers" --precision bf16 "${extra[@]}"
        for split in val test; do
            "$py" -u -m scripts.emotic.evaluate_d9_relation --root "$group" --split "$split" \
                --batch-size "$micro" --workers "$workers" "${extra[@]}"
        done
    ) 2>&1 | tee -a "$logs/$mode.log"
done
