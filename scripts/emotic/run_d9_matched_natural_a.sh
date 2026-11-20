#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=1
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
root=../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119/formal
out=output/d9_matched_natural_a/seed1_20261120
logs=logs/d9_matched_natural_a/seed1_20261120
mkdir -p "$out" "$logs"
exec 9>"$out/.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/exit.status.tmp"; mv "$out/exit.status.tmp" "$out/exit.status"' EXIT
echo "seed1 CPU matched natural/paired A $(git rev-parse HEAD) $(date -Is)"
"$py" -m unittest discover -s tests -p 'test_d9_matched_natural*.py' -v
"$py" -u -m scripts.emotic.d9_matched_natural_fit --root "$root" --out "$out"
for split in val test; do
  "$py" -u -m scripts.emotic.d9_matched_natural_eval --root "$root" --out "$out" --split "$split"
done
