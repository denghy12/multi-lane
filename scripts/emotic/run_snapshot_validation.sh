#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OMP_NUM_THREADS=1
PYTHON="${PYTHON:-python}"
SOURCE_ROOT="${SOURCE_ROOT:-../multi-lane-main-three-view-joint-to-incremental/output/emotic_three_view_joint_transfer/three_view_joint_to_incremental_val_test_20261010_01}"
RESULT_ROOT="${RESULT_ROOT:?Set fresh RESULT_ROOT}"
LOG_ROOT="${LOG_ROOT:?Set LOG_ROOT}"
SEED="${SEED:?Set SEED}"
METHOD="${METHOD:?Set METHOD}"
GPU="${GPU:?Set GPU}"
case "${METHOD}" in THREE_VIEW|THREE_VIEW_PARAX) ;; *) exit 2 ;; esac
extra=()
[[ "${PREFLIGHT:-0}" == 1 ]] && extra+=(--preflight)
mkdir -p "${LOG_ROOT}/seed${SEED}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.snapshot_validation \
  --source "${SOURCE_ROOT}/seed${SEED}/${METHOD}" \
  --output "${RESULT_ROOT}/seed${SEED}/${METHOD}" \
  --data-root ../multi-lane-main/datasets/EMOTIC \
  --clip-checkpoint ../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt \
  --face-manifest-root ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909 \
  "${extra[@]}" 2>&1 | tee "${LOG_ROOT}/seed${SEED}/${METHOD}.log"
