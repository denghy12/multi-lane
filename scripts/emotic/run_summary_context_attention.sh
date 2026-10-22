#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec "${PYTHON:-/opt/conda/envs/ddp/bin/python}" -u -m multi_lane.track_a.run_summary_context "$@"
