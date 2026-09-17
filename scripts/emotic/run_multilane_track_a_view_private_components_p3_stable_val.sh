#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU must be set}"
RUN_ID="${RUN_ID:?RUN_ID must be set}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE must be set}"
LOG_DIR="${LOG_DIR:?LOG_DIR must be set}"

# Keep the P3 architecture and all registered hyperparameters unchanged. The
# AMP run overflowed before clipping could be applied, so this stability
# control uses FP32; the model, optimizer, learning rates, ASL and data remain
# unchanged. Gradient clipping is disabled to isolate the precision change.
GPU="${GPU}" METHOD=P3 RUN_ID="${RUN_ID}" OUTPUT_BASE="${OUTPUT_BASE}" \
  LOG_DIR="${LOG_DIR}" GRADIENT_CLIP_NORM=0 NO_AMP=1 \
  bash scripts/emotic/run_multilane_track_a_view_private_components_val.sh
