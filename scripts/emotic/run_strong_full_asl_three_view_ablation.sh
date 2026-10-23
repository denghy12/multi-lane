#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec "${PYTHON:-python}" -u -m multi_lane.track_a.run_strong_full_asl "$@"
