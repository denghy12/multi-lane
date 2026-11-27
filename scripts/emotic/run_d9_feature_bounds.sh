#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
phase="${1:?Usage: bash scripts/emotic/run_d9_feature_bounds.sh check|preflight|formal|execute}"
case "$phase" in check|preflight|formal|execute) ;; *) exit 2;; esac
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
log_dir=logs/d9_feature_bounds/seed1_20261127
mkdir -p "$log_dir"
phases=("$phase")
if [[ "$phase" == execute ]]; then phases=(check preflight formal); fi
for current in "${phases[@]}"; do
    if [[ "$current" == preflight ]]; then
        python -m unittest discover -s tests -p 'test_d9_feature_bounds.py' -v 2>&1 | tee "$log_dir/tests_$(date +%Y%m%d_%H%M%S).log"
    fi
    python -u -m scripts.emotic.d9_feature_bounds --phase "$current" 2>&1 | tee "$log_dir/${current}_$(date +%Y%m%d_%H%M%S).log"
done
