#!/usr/bin/env bash
# Run the full-TRAIN ASL addition without changing either original training job.
set -euo pipefail
gpu=${1:-0}
original=${2:-../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118}
out=output/d9_sample_fusion/seed1_20261118
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
for phase in smoke formal; do
    extra="$out/sample_full_asl"
    pair="$original"
    args=()
    if [[ "$phase" == smoke ]]; then
        extra="$out/smoke/sample_full_asl"
        pair="$original/smoke"
        args+=(--smoke)
    fi
    bash scripts/emotic/run_d9_sample_fusion.sh "$phase" sample_full_asl "$gpu"
    # The independent training and both splits finish before the cross-run comparison.
    while [[ ! -s "$pair/comparison.json" ]]; do
        if [[ -e "$pair/pair.exit.status" ]]; then
            [[ -s "$pair/comparison.json" ]] && break
            echo "Original pair exited without comparison.json ($pair)" >&2
            exit 1
        fi
        sleep 30
    done
    "$py" -m scripts.emotic.summarize_d9_sample_fusion --root "$pair" --full-asl-root "$extra" "${args[@]}"
done
