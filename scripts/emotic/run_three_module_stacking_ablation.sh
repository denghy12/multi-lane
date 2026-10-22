#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export OMP_NUM_THREADS=1
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
exec "${PYTHON}" -u -m multi_lane.track_a.run_module_stacking "$@"
