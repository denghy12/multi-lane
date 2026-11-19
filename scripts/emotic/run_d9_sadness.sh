#!/usr/bin/env bash
set -euo pipefail
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
phase=${1:?specify smoke, full-val, full-test, holdout-val or holdout-test}
out=output/d9_sadness/seed1_20261119
logs=logs/d9_sadness/seed1_20261119
source_root=../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118/sample_full
mkdir -p "$out" "$logs"
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9>"$out/.$phase.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/$phase.log") 2>&1
trap 'status=$?; echo "$status" > "$out/$phase.exit.status"' EXIT
echo "phase=$phase git=$(git rev-parse HEAD) start=$(date -Is)"
prior="$out/full_cache"
tcp="$out/full_fit"
extract=("$py" -u -m scripts.emotic.extract_d9_task10_probe --source "$source_root" --source-mode sample_full --tasks 6 --batch-size 64 --workers 2)
fit=("$py" -u -m scripts.emotic.d9_tcp_probe --source-mode sample_full --prior "$prior" --out "$tcp/probe" --tasks 6)
case "$phase" in
smoke)
    "$py" -m unittest discover -s tests -p 'test_d9_sadness*.py' -v
    "${extract[@]}" --output "$out/smoke" --smoke --splits train val ;;
full-val|full-test)
    test "$(cat "$out/smoke.exit.status")" = 0
    split=${phase#full-}
    if [[ "$split" == val ]]; then
        "${extract[@]}" --output "$prior" --splits train val
        "${fit[@]}" --phase fit-val
    else
        test "$(cat "$out/full-val.exit.status")" = 0
        "${extract[@]}" --output "$prior" --splits test
        "${fit[@]}" --phase test
    fi
    "$py" -u -m scripts.emotic.d9_sadness_rank --mode sample_full --tcp "$tcp" --prior "$prior" --out "$out/full" --split "$split" ;;
holdout-val|holdout-test)
    prior=../multi-lane-main-d9-task10-probe-20261118/output/d9_task10_probe/seed1_20261118
    tcp=../multi-lane-main-d9-tcp-20261119/output/d9_tcp/seed1_20261119
    "$py" -u -m scripts.emotic.d9_sadness_rank --mode sample_holdout --tcp "$tcp" --prior "$prior" --out "$out/holdout" --split "${phase#holdout-}" ;;
*) exit 2 ;;
esac
