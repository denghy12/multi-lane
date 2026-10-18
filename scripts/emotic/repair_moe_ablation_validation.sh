#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
SOURCE="${SOURCE:-../multi-lane-main-moe-layer-routing-ablation-20261018/output/moe_layer_routing_ablation/formal_seed0_20261018_01}"
RUN_ID="${RUN_ID:-val_workers0_20261018_01}"
OUTPUT="${ROOT}/output/moe_ablation_repair/${RUN_ID}"
LOGS="${ROOT}/logs/moe_ablation_repair/${RUN_ID}"
[[ ! -e "${OUTPUT}" && ! -e "${LOGS}" ]] || { echo 'Repair ID already exists' >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Clean repair checkout required' >&2; exit 2; }
"${PYTHON}" - "${SOURCE}" <<'CHECK'
import hashlib,json,subprocess,sys
from pathlib import Path
source=Path(sys.argv[1])
for arm in ('L1_K2_F2','L1_K3_F3'):
    verification=json.loads((source/arm/'verification.json').read_text())
    assert verification['status']=='verified' and len(verification['checks'])==8
    for row in verification['checks']:
        checkpoint=source/arm/f"train/compact_checkpoints/task{row['task']}.pth"
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['checkpoint_sha256']
for gpu in (0,1):
    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    assert free>=8000,(gpu,free)
CHECK
mkdir -p "${OUTPUT}" "${LOGS}"
trap 'status=$?; echo "${status}" > "${LOGS}/exit_status.txt"' EXIT
export OMP_NUM_THREADS=1
pids=()
arms=(L1_K2_F2 L1_K3_F3)
for i in "${!arms[@]}"; do
  arm="${arms[$i]}"
  CUDA_VISIBLE_DEVICES="${i}" "${PYTHON}" -u -m multi_lane.track_a.evaluate_moe_task_ffn \
    --source-run "${SOURCE}/${arm}/train" --output "${OUTPUT}/${arm}/full_val" \
    --split val --data-root ../multi-lane-main/datasets/EMOTIC \
    --clip-checkpoint ../CODE_DDP/pretrained/clip/ViT-B-16.pt \
    --face-manifest-root ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909 \
    --expected-tasks 8 --full-cohort --batch-size 64 --workers 0 \
    > "${LOGS}/${arm}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for i in "${!arms[@]}"; do
  status=0
  wait "${pids[$i]}" || status=$?
  printf '%s\t%s\n' "${arms[$i]}" "${status}" >> "${LOGS}/arm_status.tsv"
  [[ "${status}" == 0 ]] || failed=1
done
exit "${failed}"
