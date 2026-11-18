#!/usr/bin/env bash
set -euo pipefail
phase=${1:?supply smoke or formal}
gpu=${2:-0}
root=output/d9_sample_fusion/seed1_20261118
logs=logs/d9_sample_fusion/seed1_20261118
args=()
case "$phase" in
    smoke) root="$root/smoke"; logs="$logs/smoke"; args=(--smoke) ;;
    formal) ;;
    *) exit 2 ;;
esac
mkdir -p "$root" "$logs"
exec 8>"$root/.pair.lock"
flock -n 8 || exit 2
trap 'status=$?; echo "$status" > "$root/pair.exit.status"' EXIT
modes=(sample_full sample_holdout)
for mode in "${modes[@]}"; do
    bash scripts/emotic/run_d9_sample_fusion.sh "$phase" "$mode" "$gpu" > "$logs/$mode.console" 2>&1 &
    pids+=("$!")
done
retry=()
failed=0
for i in "${!modes[@]}"; do
    mode=${modes[i]}
    if ! wait "${pids[i]}"; then
        if grep -Eiq 'CUDA out of memory|CUDA error: out of memory|OutOfMemoryError' "$logs/$mode.console"; then
            retry+=("$mode")
        else
            failed=1
        fi
    fi
done
for mode in "${retry[@]}"; do
    echo "Retrying $mode alone from last complete epoch after CUDA OOM"
    bash scripts/emotic/run_d9_sample_fusion.sh "$phase" "$mode" "$gpu" || failed=1
done
if ((failed)); then exit 1; fi
"${PYTHON:-/opt/conda/envs/ddp/bin/python}" -m scripts.emotic.summarize_d9_sample_fusion \
    --root "$root" "${args[@]}"
