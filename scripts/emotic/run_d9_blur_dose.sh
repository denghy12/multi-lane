#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${1:-0} CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
root=../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119/formal
historical=../multi-lane-main-d9-reliable-blur/output/d9_reliable_blur/seed1_20261120/formal
out=output/d9_blur_dose/seed1_20261121
logs=logs/d9_blur_dose/seed1_20261121
mkdir -p "$out" "$logs"
exec 9>"$out/.pipeline.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/exit.status.tmp"; mv "$out/exit.status.tmp" "$out/exit.status"' EXIT
echo "Start $(date -Is) git=$(git rev-parse HEAD) GPU=$CUDA_VISIBLE_DEVICES"
"$py" -m unittest discover -s tests -p 'test_d9_blur_dose*.py' -v
for phase in smoke formal; do
    args=(--root "$root" --historical "$historical" --out "$out/$phase")
    if [[ "$phase" == smoke ]]; then
        tasks=(0 1)
        args+=(--smoke)
    else
        tasks=(2 3 4 5 6 7)
    fi
    "$py" -u -m scripts.emotic.d9_blur_dose_extract --root "$root" --out "$out/extra" --tasks "${tasks[@]}" --workers 2
    CUDA_VISIBLE_DEVICES="" "$py" -u -m scripts.emotic.d9_blur_dose_fit "${args[@]}" --extra "$out/extra"
    CUDA_VISIBLE_DEVICES="" "$py" -u -m scripts.emotic.d9_blur_dose_eval "${args[@]}" --split val
    if [[ "$phase" == formal ]]; then
        CUDA_VISIBLE_DEVICES="" "$py" -u -m scripts.emotic.d9_blur_dose_eval "${args[@]}" --split test
    fi
    echo "Completed $phase $(date -Is)"
done
"$py" -m scripts.emotic.d9_blur_dose_report --out "$out/formal"
echo "Completed $(date -Is)"
