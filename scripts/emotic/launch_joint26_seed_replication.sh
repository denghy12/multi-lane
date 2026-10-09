#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BATCH_PREFIX="${BATCH_PREFIX:?Set BATCH_PREFIX}"
PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
export AMP_INITIAL_SCALE=1024 AMP_GROWTH_INTERVAL=1000000000
for seed in ${SEEDS}; do
  [[ "${seed}" =~ ^[012]$ ]] || { echo "Invalid seed: ${seed}" >&2; exit 2; }
  BATCH_ID="${BATCH_PREFIX}_seed${seed}" SEED="${seed}" \
    bash scripts/emotic/launch_joint26_parax_view_ablation.sh
done
"${PYTHON}" -m multi_lane.track_a.summarize_joint26_seeds \
  --result-root ./output/emotic_joint26_parax_view_ablation --batch-prefix "${BATCH_PREFIX}"
