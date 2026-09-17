#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU must be set}"
RUN_ID="${RUN_ID:?RUN_ID must be set}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE must be set}"
LOG_DIR="${LOG_DIR:?LOG_DIR must be set}"

# Keep the P3 architecture and all registered hyperparameters unchanged. The
# only stabilization is a fixed post-unscale global gradient cap, recorded in
# the runner config and applied to every optimizer parameter group.
GPU="${GPU}" METHOD=P3 RUN_ID="${RUN_ID}" OUTPUT_BASE="${OUTPUT_BASE}" \
  LOG_DIR="${LOG_DIR}" GRADIENT_CLIP_NORM=1.0 \
  bash scripts/emotic/run_multilane_track_a_view_private_components_val.sh
