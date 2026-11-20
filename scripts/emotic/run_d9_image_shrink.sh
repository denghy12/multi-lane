#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
source_root=../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119/formal
gate_root=../multi-lane-main-d9-image-gate/output/d9_image_gate/seed1_20261120
root=output/d9_image_shrink/seed1_20261120
logs=logs/d9_image_shrink/seed1_20261120
mkdir -p "$root" "$logs"
exec 9>"$root/.pipeline.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$root/exit.status.tmp"; mv "$root/exit.status.tmp" "$root/exit.status"' EXIT
echo "Started TRAIN image shrinkage $(date -Is) $(git rev-parse HEAD)"
"$py" -m unittest discover -s tests -p 'test_d9_image_shrink*.py' -v
for phase in smoke formal; do
    args=(--source "$source_root" --image-run "$gate_root/$phase" --out "$root/$phase")
    if [[ $phase == smoke ]]; then args+=(--smoke); fi
    "$py" -u -m scripts.emotic.d9_image_shrink_fit "${args[@]}"
    "$py" -u -m scripts.emotic.d9_image_shrink_eval "${args[@]}" --split val
    "$py" -u -m scripts.emotic.d9_image_shrink_eval "${args[@]}" --split test
    "$py" -u -m scripts.emotic.d9_image_shrink_report --out "$root/$phase"
    echo "Completed $phase $(date -Is)"
done
