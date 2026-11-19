#!/usr/bin/env bash
set -euo pipefail
phase=${1:?expected smoke-extract, train-val-features, fit-val, test-features or test}
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
source_root=../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118/sample_full
cache6=../multi-lane-main-d9-sadness-20261119/output/d9_sadness/seed1_20261119/full_cache
out=output/d9_disagreement/seed1_20261119
logs=logs/d9_disagreement/seed1_20261119
mkdir -p "$out" "$logs"
exec 9>"$out/.$phase.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/$phase.log") 2>&1
trap 'status=$?; echo "$status" > "$out/$phase.exit.status"' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "phase=$phase git=$(git rev-parse HEAD) start=$(date -Is)"
extract=("$py" -u -m scripts.emotic.extract_d9_task10_probe --source "$source_root" --source-mode sample_full --batch-size 64 --workers 2)
analyze=("$py" -u -m scripts.emotic.d9_disagreement_run --cache "$out/cache" --cache6 "$cache6" --source "$source_root" --out "$out/readout")
case "$phase" in
    smoke-extract) "${extract[@]}" --tasks 0 --splits train val --smoke --output "$out/smoke" ;;
    train-val-features)
        test "$(cat "$out/smoke-extract.exit.status")" = 0
        "${extract[@]}" --tasks 0 3 --splits train val --output "$out/cache" ;;
    fit-val)
        "$py" -m unittest discover -s tests -p 'test_d9_disagreement*.py' -v
        "${analyze[@]}" --phase fit-val ;;
    test-features)
        test "$(cat "$out/fit-val.exit.status")" = 0
        "${extract[@]}" --tasks 0 3 --splits test --output "$out/cache" ;;
    test)
        test "$(cat "$out/test-features.exit.status")" = 0
        "${analyze[@]}" --phase test ;;
    *) exit 2 ;;
esac
