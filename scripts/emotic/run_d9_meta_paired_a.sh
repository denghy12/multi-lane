#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${1:-0} PYTHONHASHSEED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
split=../multi-lane-main-d9-split90/output/d9_split90/seed1_20261120
random=../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119/formal
out=output/d9_meta_paired_a/seed1_20261120
logs=logs/d9_meta_paired_a/seed1_20261120
mkdir -p "$out" "$logs"
exec 9>"$out/.pipeline.lock"
flock -n 9 || { echo "Another run owns $out" >&2; exit 1; }
exec > >(tee -a "$logs/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/exit.status.tmp"; mv "$out/exit.status.tmp" "$out/exit.status"' EXIT
echo "Start $(date -Is) git=$(git rev-parse HEAD) gpu=$CUDA_VISIBLE_DEVICES"
"$py" -m unittest discover -s tests -p 'test_d9_meta_paired*.py' -v
for phase in smoke formal; do
    source_model="$split/$phase/meta"
    test "$(cat "$split/$phase.exit.status")" = 0
    cmp "$split/audit/meta/train_image_roles.json" "$source_model/train_image_roles.json"
    last=7
    if [[ "$phase" == smoke ]]; then last=1; fi
    paired=("$py" -u -m scripts.emotic.d9_equal_updates_paired --replay-meta --root "$source_model" --out "$out/$phase" --workers 2)
    for ((t=0;t<=last;t++)); do "${paired[@]}" --phase fit --task "$t"; done
    "${paired[@]}" --phase val
    "${paired[@]}" --phase test
    echo "Completed $phase $(date -Is)"
done
"$py" -u -m scripts.emotic.d9_meta_paired_summary --out "$out/formal" --reference "$random/paired_A" --full-comparison "$random/comparison"
echo "Complete $(date -Is)"
