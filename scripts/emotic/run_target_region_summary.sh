#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
exec "${PYTHON:-/opt/conda/envs/ddp/bin/python}" -u -m multi_lane.track_a.run_regional_summary "$@"
