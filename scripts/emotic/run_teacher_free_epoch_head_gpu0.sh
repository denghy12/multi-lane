#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_id="${1:-seed1_20261028_01}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=72 MKL_NUM_THREADS=72
exec "${ML_PYTHON:-/opt/conda/envs/ddp/bin/python}" -u -m multi_lane.track_a.run_teacher_free_epoch_head \
 --source-root ../multi-lane-main-teacher-free-shared-learning-20261027/output/teacher_free_shared_learning/seed1_20261027_01 \
 --data-root ../multi-lane-main/datasets/EMOTIC \
 --clip-checkpoint ../CODE_DDP/pretrained/clip/ViT-B-16.pt \
 --face-manifest-root ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909 \
 --output "output/teacher_free_epoch_head/${run_id}" \
 --logs "logs/teacher_free_epoch_head/${run_id}"
