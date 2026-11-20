#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
source_root=../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119/formal
historical=../multi-lane-main-d9-reliable-blur/output/d9_reliable_blur/seed1_20261120/formal
out=output/d9_natural_rank/seed1_20261120
logs=logs/d9_natural_rank/seed1_20261120
mkdir -p "$out" "$logs"
exec 9>"$out/.pipeline.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/exit.status.tmp"; mv "$out/exit.status.tmp" "$out/exit.status"' EXIT
echo "Start CPU seed1 $(git rev-parse HEAD) $(date -Is)"
"$py" -m unittest discover -s tests -p 'test_d9_natural_rank*.py' -v
for phase in smoke formal; do
    args=(--root "$source_root" --historical "$historical" --out "$out/$phase")
    if [[ "$phase" == smoke ]]; then args+=(--smoke); fi
    "$py" -u -m scripts.emotic.d9_natural_rank_fit "${args[@]}"
    "$py" -u -m scripts.emotic.d9_natural_rank_eval "${args[@]}" --split val
    if [[ "$phase" == formal ]]; then
        "$py" -u -m scripts.emotic.d9_natural_rank_eval "${args[@]}" --split test
    fi
    echo "Completed $phase $(date -Is)"
done
"$py" -m scripts.emotic.d9_natural_rank_report --out "$out/formal"
echo "Completed $(date -Is)"
