#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
RUN_ID="${RUN_ID:-formal_seed0_20261018_01}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
[[ "${GPU:-1}" == 1 ]] || { echo "Locked to GPU1" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean checkout required" >&2; exit 2; }
"${PYTHON}" - <<'CHECK'
import subprocess
uuid,free,util=[x.strip() for x in subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(',')]
busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
assert uuid not in busy and int(free)>12000 and int(util)<=5,'GPU1 must be idle; no job will be stopped'
CHECK
LOG_ROOT="logs/moe_expert_rollback/${RUN_ID}"
[[ ! -e "${LOG_ROOT}" ]] || { echo "Existing run ID" >&2; exit 2; }
mkdir -p "${LOG_ROOT}"
trap 'status=$?; echo "$status" > "${LOG_ROOT}/exit_status.txt"' EXIT
args=()
if [[ "${SMOKE:-0}" == 1 ]]; then args=(--train-batches 1 --val-limit 64); fi
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 "${PYTHON}" -u -m multi_lane.track_a.expert_rollback \
  --af ../multi-lane-main-moe-activate-freeze-20261018/output/moe_activate_freeze/formal_seed0_20261018_01/train \
  --waf ../multi-lane-main-moe-weighted-freeze-20261018/output/moe_weighted_freeze/formal_seed0_20261018_01/train \
  --data-root ../multi-lane-main/datasets/EMOTIC \
  --clip-checkpoint ../CODE_DDP/pretrained/clip/ViT-B-16.pt \
  --face-manifest-root ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909 \
  --output "output/moe_expert_rollback/${RUN_ID}" "${args[@]}" \
  2>&1 | tee "${LOG_ROOT}/run.log"
