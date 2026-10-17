#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
RUN_ID="${RUN_ID:-formal_20261017_01}"
SOURCE_ROOT="${SOURCE_ROOT:-../multi-lane-main-moe-task-ffn-v4-20261017/output/moe_task_ffn_v4}"
DATA_ROOT="${DATA_ROOT:-../multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-../CODE_DDP/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}"
OUTPUT="output/moe_v4_gradient_audit/${RUN_ID}"
LOG_ROOT="logs/moe_v4_gradient_audit/${RUN_ID}"
[[ ! -e "${OUTPUT}" && ! -e "${LOG_ROOT}" ]] || { echo "Run ID exists" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean diagnostic Git checkout required" >&2; exit 2; }
"${PYTHON}" - <<'PY'
import subprocess
uuid,free,util=(x.strip() for x in subprocess.check_output([
 'nvidia-smi','-i','0','--query-gpu=uuid,memory.free,utilization.gpu',
 '--format=csv,noheader,nounits'],text=True).strip().split(','))
busy=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines())
if uuid in busy or int(free)<14000 or int(util)>5:
 raise SystemExit('GPU0 must be idle; no other process will be stopped')
PY
mkdir -p "${LOG_ROOT}"
args=(--tasks 0 2 7 --batches 4 --batch-size 64)
if [[ "${SMOKE:-0}" == 1 ]]; then
  args=(--tasks 0 --batches 1 --batch-size 16)
fi
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u -m multi_lane.track_a.gradient_attribution \
  --source-root "${SOURCE_ROOT}" --data-root "${DATA_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --output "${OUTPUT}" --seed 0 "${args[@]}" 2>&1 | tee "${LOG_ROOT}/run.log"
