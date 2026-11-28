#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
phase="${1:?Usage: bash scripts/emotic/run_d9_task_patch.sh check|tests|preflight|formal|execute}"
case "$phase" in check|tests|preflight|formal|execute) ;; *) exit 2;; esac
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
log_dir=logs/d9_task_patch/seed1_20261128
mkdir -p "$log_dir"
phases=("$phase")
if [[ "$phase" == execute ]]; then phases=(check tests preflight formal); fi
for current in "${phases[@]}"; do
    if [[ "$current" == tests ]]; then
        python -m unittest discover -s tests -p 'test_d9_task_patch*.py' -v 2>&1 | tee "$log_dir/tests_$(date +%Y%m%d_%H%M%S).log"
        python -m unittest discover -s tests -p 'test_d9_final_patch*.py' -v 2>&1 | tee "$log_dir/regression_$(date +%Y%m%d_%H%M%S).log"
    else
        python -u -m scripts.emotic.d9_task_patch_run --phase "$current" 2>&1 | tee "$log_dir/${current}_$(date +%Y%m%d_%H%M%S).log"
    fi
done
