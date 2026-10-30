#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_id="${1:-seed1_query_repair_20261030_01}"
if (( $# )); then shift; fi
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=72
export MKL_NUM_THREADS=72
export OPENBLAS_NUM_THREADS=1
exec "${ML_PYTHON:-/opt/conda/envs/ddp/bin/python}" -u -m multi_lane.track_a.run_person_query_conditioning \
  --source-run ../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train \
  --data-root ../multi-lane-main/datasets/EMOTIC \
  --clip-checkpoint ../CODE_DDP/pretrained/clip/ViT-B-16.pt \
  --face-manifest-root ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909 \
  --census ../multi-lane-main-person-grounded-summary-20261029/output/person_grounded_summary/train_geometry_census_20261029/result.json \
  --output "output/person_query_conditioning/${run_id}" \
  --logs "logs/person_query_conditioning/${run_id}" "$@"
