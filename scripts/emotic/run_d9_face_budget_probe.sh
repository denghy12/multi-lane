#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=1
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
matched=../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119/formal
old=../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118/sample_holdout
pairs=../multi-lane-main-d9-paired-a-eight-20261119/output/d9_paired_a_eight/seed1_20261119
out=output/d9_face_budget_probe/seed1_20261120
logs=logs/d9_face_budget_probe/seed1_20261120
mkdir -p "$out" "$logs"
exec 9>"$out/.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/exit.status.tmp"; mv "$out/exit.status.tmp" "$out/exit.status"' EXIT
echo "seed1 CPU face-budget probe $(git rev-parse HEAD) $(date -Is)"
"$py" -m unittest discover -s tests -p 'test_d9_face_budget*.py' -v
"$py" -u -m scripts.emotic.d9_face_budget_fit --matched "$matched" --old "$old" --old-pairs "$pairs" --out "$out"
for split in val test; do
  "$py" -u -m scripts.emotic.d9_face_budget_eval --matched "$matched" --old "$old" --split "$split" --out "$out"
done
