#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
source_root=${D9_PROBE_SOURCE:-../multi-lane-main-d9-task10-probe-20261118/output/d9_task10_probe/seed1_20261118}
out=output/d9_shrink/seed1_20261119
logs=logs/d9_shrink/seed1_20261119
mkdir -p "$out" "$logs"
exec 9>"$out/.pipeline.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/run.log") 2>&1
trap 'status=$?; echo "$status" > "$out/exit.status"' EXIT
echo "git=$(git rev-parse HEAD) start=$(date -Is)"
"$py" -u -m scripts.emotic.d9_shrink_fit --source "$source_root" --out "$out"
for split in val test; do
    "$py" -u -m scripts.emotic.d9_shrink_eval --source "$source_root" --out "$out" --split "$split"
done
