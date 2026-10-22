#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
RUN_ID="${RUN_ID:-formal_seed0_20261022_01}"
args=(--run-id "${RUN_ID}")
if [[ "${SMOKE:-0}" == 1 ]]; then
    args+=(--smoke)
fi
exec "${PYTHON}" -u -m multi_lane.track_a.run_spatial_summary "${args[@]}" "$@"
