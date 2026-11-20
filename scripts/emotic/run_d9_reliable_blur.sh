#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
source_root=../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119/formal
historical=../multi-lane-main-d9-matched-natural-a/output/d9_matched_natural_a/seed1_20261120
out=output/d9_reliable_blur/seed1_20261120
logs=logs/d9_reliable_blur/seed1_20261120
mkdir -p "$logs" "$out"
exec 9>"$out/.pipeline.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/exit.status.tmp"; mv "$out/exit.status.tmp" "$out/exit.status"' EXIT
echo "Start CPU seed1 $(git rev-parse HEAD) $(date -Is)"
"$py" -m unittest discover -s tests -p 'test_d9_reliable_blur*.py' -v
for phase in smoke formal; do
    args=(--root "$source_root" --historical "$historical" --out "$out/$phase")
    if [[ "$phase" == smoke ]]; then args+=(--smoke); fi
    "$py" -u -m scripts.emotic.d9_reliable_blur_fit "${args[@]}"
    "$py" -u -m scripts.emotic.d9_reliable_blur_eval "${args[@]}" --split val
    "$py" -u -m scripts.emotic.d9_reliable_blur_eval "${args[@]}" --split test
    echo "Completed $phase $(date -Is)"
done
"$py" -m scripts.emotic.d9_reliable_blur_report --out "$out/formal"
echo "Completed $(date -Is)"
