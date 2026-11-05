#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=72 MKL_NUM_THREADS=72 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/ddp/bin/python}"
RUN_ID="${RUN_ID:-seed1_20261105}"
mkdir -p "logs/shared_b0_face_cross_view/$RUN_ID"
for suite in shared_b0_face_cross_view historical_b32_modules historical_b32_search track_a_cross_view_attention; do
  "$PYTHON_BIN" -m unittest discover -s tests -p "test_${suite}.py" -v > "logs/shared_b0_face_cross_view/$RUN_ID/test_${suite}.log" 2>&1
done
"$PYTHON_BIN" -u scripts/emotic/run_shared_b0_face_cross_view.py --smoke --root "output/shared_b0_face_cross_view/${RUN_ID}_smoke" > "logs/shared_b0_face_cross_view/$RUN_ID/smoke_controller.log" 2>&1
"$PYTHON_BIN" -u scripts/emotic/run_shared_b0_face_cross_view.py --root "output/shared_b0_face_cross_view/$RUN_ID" --smoke-root "output/shared_b0_face_cross_view/${RUN_ID}_smoke" > "logs/shared_b0_face_cross_view/$RUN_ID/controller.log" 2>&1
