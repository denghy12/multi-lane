#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
phase=${1:?smoke or formal}
gpu=${2:-0}
case "$phase" in smoke|formal) ;; *) exit 2 ;; esac
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
source_root=${SOURCE_ROOT:-../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118/sample_full}
root=${FILL_ROOT:-output/d9_fill_fusion_tables/seed1_20261122}
out="$root/$phase/image_full"
logs="logs/d9_fill_fusion_tables/seed1_20261122"
mkdir -p "$out" "$logs" "$root/shared/image_full_native"
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONHASHSEED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
exec 9>"$out/.pipeline.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/image_full_$phase.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/exit.status.tmp"; mv "$out/exit.status.tmp" "$out/exit.status"' EXIT
echo "Image fill $phase $(date -Is) commit=$(git rev-parse HEAD) GPU=$gpu"
if [[ ! -e "$out/native_train" ]]; then
    ln -s ../../shared/image_full_native "$out/native_train"
fi
last=7
flags=()
if [[ $phase == smoke ]]; then last=0; flags+=(--smoke); fi
for ((task=0; task<=last; task++)); do
    "$py" -u -m scripts.emotic.d9_fill_image_source --source "$source_root" --out "$out/native_train" \
        --task "$task" --workers 2 --batch-size 64
    "$py" -u -m scripts.emotic.d9_image_gate_extract --source "$source_root" --out "$out/features" \
        --split train --task "$task" --workers 2 --batch-size 12 "${flags[@]}"
done
# These small readouts run on CPU while the GPU extracts Validation image descriptors.
(
    export CUDA_VISIBLE_DEVICES=""
    "$py" -u -m scripts.emotic.d9_image_gate_readout --source "$source_root" --features "$out/features" \
        --out "$out/readout" --phase fit "${flags[@]}"
    "$py" -u -m scripts.emotic.d9_image_shrink_fit --source "$source_root" --image-run "$out" \
        --out "$out/shrink" "${flags[@]}"
) &
fit_pid=$!
for split in val test; do
    for ((task=0; task<=last; task++)); do
        "$py" -u -m scripts.emotic.d9_image_gate_extract --source "$source_root" --out "$out/features" \
            --split "$split" --task "$task" --workers 2 --batch-size 12 "${flags[@]}"
    done
    if [[ $split == val ]]; then wait "$fit_pid"; fi
    CUDA_VISIBLE_DEVICES="" "$py" -u -m scripts.emotic.d9_image_gate_readout --source "$source_root" \
        --features "$out/features" --out "$out/readout" --phase "$split" "${flags[@]}"
    CUDA_VISIBLE_DEVICES="" "$py" -u -m scripts.emotic.d9_image_shrink_eval --source "$source_root" \
        --image-run "$out" --out "$out/shrink" --split "$split" "${flags[@]}"
done
CUDA_VISIBLE_DEVICES="" "$py" -u -m scripts.emotic.d9_image_gate_report --root "$out" --source "$source_root" "${flags[@]}"
CUDA_VISIBLE_DEVICES="" "$py" -u -m scripts.emotic.d9_image_shrink_report --out "$out/shrink"
echo "Completed image fill $phase $(date -Is)"
