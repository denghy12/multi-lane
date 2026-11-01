#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=72 MKL_NUM_THREADS=72 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/ddp/bin/python}"
RUN_ID="${RUN_ID:-seed1_20261101_01}"
COMMON=(--source-run ../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train
  --data-root ../multi-lane-main/datasets/EMOTIC
  --clip-checkpoint ../CODE_DDP/pretrained/clip/ViT-B-16.pt
  --face-manifest-root ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909)
mkdir -p "logs/crossvit_independent_three_view/$RUN_ID"
"$PYTHON_BIN" -m unittest discover -s tests -p test_crossvit_three_view.py -v \
  > "logs/crossvit_independent_three_view/$RUN_ID/tests.log" 2>&1
"$PYTHON_BIN" -u -m multi_lane.track_a.run_crossvit_three_view "${COMMON[@]}" \
  --smoke --smoke-updates 4 --output "output/crossvit_independent_three_view/${RUN_ID}_smoke" \
  --logs "logs/crossvit_independent_three_view/${RUN_ID}_smoke"
"$PYTHON_BIN" -u -m multi_lane.track_a.run_crossvit_three_view "${COMMON[@]}" \
  --smoke-root "output/crossvit_independent_three_view/${RUN_ID}_smoke" \
  --output "output/crossvit_independent_three_view/$RUN_ID" \
  --logs "logs/crossvit_independent_three_view/$RUN_ID"
