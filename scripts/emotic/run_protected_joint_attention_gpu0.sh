#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_id="${1:-seed1_20261028_01}"
if (( $# )); then shift; fi
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=72
export MKL_NUM_THREADS=72
exec "${ML_PYTHON:-/opt/conda/envs/ddp/bin/python}" -u -m multi_lane.track_a.run_protected_joint_attention \
  --source-run ../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules/formal_seed1_20261025_01/BASE/train \
  --data-root ../multi-lane-main/datasets/EMOTIC \
  --clip-checkpoint ../CODE_DDP/pretrained/clip/ViT-B-16.pt \
  --face-manifest-root ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909 \
  --output "output/protected_joint_attention/${run_id}" \
  --logs "logs/protected_joint_attention/${run_id}" \
  --maximum-jobs 4 "$@"
