#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
[[ "${GPU:-0}" == 0 ]] || { echo 'This control is locked to GPU0' >&2; exit 2; }
RUN_ID="${RUN_ID:-formal_seed0_20261020_01}"
OUTPUT="${ROOT}/output/moe_frozen_route_control/${RUN_ID}"
LOGS="${ROOT}/logs/moe_frozen_route_control/${RUN_ID}"
[[ ! -e "${OUTPUT}" && ! -e "${LOGS}" && -z "$(git status --porcelain)" ]] || { echo 'Fresh output and clean worktree required' >&2; exit 2; }
"${PYTHON}" - <<'GPU_CHECK'
import subprocess
uuid,free,util=(v.strip() for v in subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
assert uuid not in busy and int(free)>=8000 and int(util)<=5,'GPU0 must be idle; no processes will be stopped'
GPU_CHECK
mkdir -p "${LOGS}"
trap 'status=$?; echo "${status}" > "${LOGS}/exit_status.txt"' EXIT
export OMP_NUM_THREADS=1
options=(--calibration-batches 4 --bootstrap 500)
if [[ "${SMOKE:-0}" == 1 ]]; then
  options=(--calibration-batches 2 --validation-limit 64 --bootstrap 20)
fi
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u -m multi_lane.track_a.late_route_control \
  --source ../multi-lane-main-moe-frozen-forward-20261019/output/moe_frozen_forward/formal_seed0_20261019_01 \
  --output "${OUTPUT}" --data-root ../multi-lane-main/datasets/EMOTIC \
  --clip-checkpoint ../CODE_DDP/pretrained/clip/ViT-B-16.pt \
  --face-manifest-root ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909 \
  --tasks 0 2 7 --calibration-seed 20261020 --bootstrap-seed 20261020 \
  "${options[@]}" 2>&1 | tee "${LOGS}/run.log"
