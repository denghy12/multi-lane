#!/usr/bin/env bash
# Linux only. Select a free GPU explicitly before preflight/pilot.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
phase="${1:?Usage: bash scripts/emotic/run_d9_summary_fusion.sh check|preflight|pilot|execute}"
case "$phase" in check|preflight|pilot|execute) ;; *) echo "Unknown phase: $phase" >&2; exit 2;; esac
if [[ "$phase" != check ]]; then
    : "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to a checked, available GPU}"
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
log_dir=logs/d9_summary_fusion/seed1_20261126
mkdir -p "$log_dir"
phases=("$phase")
if [[ "$phase" == execute ]]; then
    phases=(check preflight pilot)
fi
for current in "${phases[@]}"; do
    python -u -m scripts.emotic.d9_summary_run --phase "$current" \
        2>&1 | tee "$log_dir/${current}_$(date +%Y%m%d_%H%M%S).log"
done
