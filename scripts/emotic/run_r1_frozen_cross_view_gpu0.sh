#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=72 MKL_NUM_THREADS=72 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/ddp/bin/python}"
RUN_ID="${RUN_ID:-seed1_20261105}"
mkdir -p "logs/r1_frozen_cross_view/$RUN_ID"
"$PYTHON_BIN" -m unittest discover -s tests -p test_r1_frozen_cross_view.py -v > "logs/r1_frozen_cross_view/$RUN_ID/frozen_tests.log" 2>&1
"$PYTHON_BIN" -m unittest discover -s tests -p test_r1_crossvit.py -v > "logs/r1_frozen_cross_view/$RUN_ID/r1_regression.log" 2>&1
"$PYTHON_BIN" -m unittest discover -s tests -p test_r1_selector_query.py -v > "logs/r1_frozen_cross_view/$RUN_ID/query_regression.log" 2>&1
"$PYTHON_BIN" -u scripts/emotic/run_r1_frozen_cross_view.py --smoke --root "output/r1_frozen_cross_view/${RUN_ID}_smoke" > "logs/r1_frozen_cross_view/$RUN_ID/smoke_controller.log" 2>&1
"$PYTHON_BIN" -u scripts/emotic/run_r1_frozen_cross_view.py --root "output/r1_frozen_cross_view/$RUN_ID" > "logs/r1_frozen_cross_view/$RUN_ID/controller.log" 2>&1
