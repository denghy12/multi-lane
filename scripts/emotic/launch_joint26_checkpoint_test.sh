#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
RUN_ID="${RUN_ID:?Set RUN_ID}"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-1}"
exec "${PYTHON}" -m multi_lane.track_a.run_joint26_locked_test \
  --original-batch-root ../multi-lane-main-joint26-parax-view-ablation/output/emotic_joint26_parax_view_ablation/joint26_parax_views_seed0_val_gpu0_20261009_01 \
  --stable-result-root ../multi-lane-main-joint26-parax-seed-replication/output/emotic_joint26_parax_view_ablation \
  --stable-prefix joint26_stableamp_multiseed_val_gpu0_20261009_01 \
  --output-root "./output/emotic_joint26_locked_test/${RUN_ID}" \
  --log-root "./logs/emotic_joint26_locked_test/${RUN_ID}" \
  --data-root ../multi-lane-main/datasets/EMOTIC \
  --clip-checkpoint ../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt \
  --face-manifest-root ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909 \
  --gpu "${GPU}" --wait-hours 12
