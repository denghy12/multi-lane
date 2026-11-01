#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=72 MKL_NUM_THREADS=72 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/ddp/bin/python}"
RUN_ID="${RUN_ID:-seed1_20261101_01}"
COMMON=(--reference-root ../emotic_benchmark_runs --data-root ../multi-lane-main/datasets/EMOTIC
  --clip-checkpoint ../CODE_DDP/pretrained/clip/ViT-B-16.pt
  --face-manifest-root ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909)
mkdir -p "logs/r1_crossvit/$RUN_ID"
"$PYTHON_BIN" -m unittest discover -s tests -p test_r1_crossvit.py -v > "logs/r1_crossvit/$RUN_ID/tests.log" 2>&1
"$PYTHON_BIN" -u -m multi_lane.track_a.run_r1_crossvit "${COMMON[@]}" --smoke \
  --output "output/r1_crossvit/${RUN_ID}_smoke" --logs "logs/r1_crossvit/${RUN_ID}_smoke"
"$PYTHON_BIN" -u -m multi_lane.track_a.run_r1_crossvit "${COMMON[@]}" \
  --smoke-root "output/r1_crossvit/${RUN_ID}_smoke" \
  --output "output/r1_crossvit/$RUN_ID" --logs "logs/r1_crossvit/$RUN_ID"
