#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
phase=${1:?smoke or formal}
case "$phase" in smoke|formal) ;; *) exit 2 ;; esac
export CUDA_VISIBLE_DEVICES=${2:-0} PYTHONHASHSEED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
source_root=../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119/formal
root=output/d9_fill_fusion_tables/seed1_20261122
out="$root/$phase/probes_equal90"
logs=logs/d9_fill_fusion_tables/seed1_20261122
mkdir -p "$out" "$logs"
exec 9>"$out/.pipeline.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/probes_$phase.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/exit.status.tmp"; mv "$out/exit.status.tmp" "$out/exit.status"' EXIT
args=(--source "$source_root" --out "$out")
if [[ "$phase" == smoke ]]; then
    "$py" -u -m scripts.emotic.d9_fill_probes "${args[@]}" --phase smoke
    exit 0
fi
[[ "$(cat "$root/smoke/probes_equal90/exit.status" 2>/dev/null || true)" == 0 ]]
"$py" -u -m scripts.emotic.d9_fill_probes "${args[@]}" --phase extract --splits train val
pids=()
for family in tcp quality budget disagreement; do
    CUDA_VISIBLE_DEVICES="" "$py" -u -m scripts.emotic.d9_fill_probes "${args[@]}" \
        --phase fit-val --family "$family" > "$logs/probes_${family}_fit.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]]
"$py" -u -m scripts.emotic.d9_fill_probes "${args[@]}" --phase extract --splits test
CUDA_VISIBLE_DEVICES="" "$py" -u -m scripts.emotic.d9_fill_probes "${args[@]}" --phase test
