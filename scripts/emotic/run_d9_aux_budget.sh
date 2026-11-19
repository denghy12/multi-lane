#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
source_root=../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118/sample_full
cache=../multi-lane-main-d9-disagreement-20261119/output/d9_disagreement/seed1_20261119/cache
cache6=../multi-lane-main-d9-sadness-20261119/output/d9_sadness/seed1_20261119/full_cache
out=output/d9_aux_budget/seed1_20261119
logs=logs/d9_aux_budget/seed1_20261119
mkdir -p "$out" "$logs"
exec 9>"$out/.run.lock"
flock -n 9
exec > >(tee -a "$logs/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/exit.status"' EXIT
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=1
echo "git=$(git rev-parse HEAD) start=$(date -Is)"
"$py" -m unittest discover -s tests -p 'test_d9_aux_budget*.py' -v
for phase in smoke fit-val test; do
    "$py" -u -m scripts.emotic.d9_aux_budget_run --cache "$cache" --cache6 "$cache6" --source "$source_root" --out "$out/readout" --phase "$phase"
done
