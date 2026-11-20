#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${1:-0} PYTHONHASHSEED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
source_root=../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119/formal
root=output/d9_image_gate/seed1_20261120
logs=logs/d9_image_gate/seed1_20261120
mkdir -p "$root" "$logs"
exec 9>"$root/.pipeline.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$root/exit.status.tmp"; mv "$root/exit.status.tmp" "$root/exit.status"' EXIT
echo "Started image-level fusion at $(date -Is), GPU $CUDA_VISIBLE_DEVICES, commit $(git rev-parse HEAD)"
"$py" -m unittest discover -s tests -p 'test_d9_image_gate*.py' -v

run_round() {
    local phase=$1 last=7 split task
    local out="$root/$phase"
    local flags=()
    if [[ $phase == smoke ]]; then last=0; flags+=(--smoke); fi
    for split in train val test; do
        echo "$phase $split feature extraction $(date -Is)"
        for ((task=0; task<=last; task++)); do
            "$py" -u -m scripts.emotic.d9_image_gate_extract --source "$source_root" --out "$out/features" \
                --split "$split" --task "$task" --workers 2 --batch-size 12 "${flags[@]}"
        done
        local operation=$split
        if [[ $split == train ]]; then operation=fit; fi
        echo "$phase $operation readout $(date -Is)"
        "$py" -u -m scripts.emotic.d9_image_gate_readout --source "$source_root" --features "$out/features" \
            --out "$out/readout" --phase "$operation" "${flags[@]}"
    done
    "$py" -u -m scripts.emotic.d9_image_gate_report --root "$out" --source "$source_root" "${flags[@]}"
    echo "Completed $phase $(date -Is)"
}
run_round smoke
run_round formal
