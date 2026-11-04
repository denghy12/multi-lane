#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=72 MKL_NUM_THREADS=72 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/ddp/bin/python}"
RUN_ID="${RUN_ID:-seed1_20261104}"
BASE=../multi-lane-main-r1-final-cls/output/r1_final_cls/seed1_20261104
REFERENCE=../multi-lane-main-r1-crossvit-20261101/output/r1_crossvit/seed1_20261101_02
mkdir -p "logs/r1_cls_qk_control/$RUN_ID"
"$PYTHON_BIN" -m unittest discover -s tests -p test_r1_cls_qk_control.py -v > "logs/r1_cls_qk_control/$RUN_ID/tests.log" 2>&1
while [ "$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 6000 ]; do sleep 30; done
"$PYTHON_BIN" -u scripts/emotic/run_r1_cls_qk_control.py --smoke --base-root "$BASE" --reference "$REFERENCE" \
  --output "output/r1_cls_qk_control/${RUN_ID}_smoke" > "logs/r1_cls_qk_control/$RUN_ID/smoke.log" 2>&1
"$PYTHON_BIN" -u scripts/emotic/run_r1_cls_qk_control.py --base-root "$BASE" --reference "$REFERENCE" \
  --output "output/r1_cls_qk_control/$RUN_ID" > "logs/r1_cls_qk_control/$RUN_ID/formal.log" 2>&1
