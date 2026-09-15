#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BATCH_ID="${BATCH_ID:-full_private_head_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1}"
IFS=',' read -r -a gpus <<< "${GPU_LIST}"
[[ ${#gpus[@]} -eq 2 ]] || { echo "GPU_LIST must contain exactly two GPUs" >&2; exit 2; }
[[ "${gpus[0]}" != "${gpus[1]}" ]] || { echo "GPUs must be distinct" >&2; exit 2; }
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
READY_CHECKS="${READY_CHECKS:-2}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_full_private_head/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_full_private_head/${BATCH_ID}"
RESULT_BASE="${CONTROL_DIR}/runs"
mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}" "${RESULT_BASE}"

cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF
batch=${BATCH_ID}
selection=seed0 complete 8-task validation only; test forbidden
H0=shared classifier applied per view, then fixed logit fusion
HF=Full private classifier copied exactly from shared initialization; Body/Face retain shared classifier
constant=shared frozen CLIP, Selector, Prompt, b32 Adapter, losses, data order, optimizer schedule, fixed weights and masks
evaluation=training logit fusion plus offline probability fusion; per-view scores and compact states saved
advance=H0 within 0.10 of historical P0; HF final >= H0+0.10; average and Full standalone not lower; gap to R1 shrinks
uncertainty=2000 paired original-image-group bootstrap replicates; clear effect then seed1/2
EOF

consecutive=0
while (( consecutive < READY_CHECKS )); do
  ready=1
  for gpu in "${gpus[@]}"; do
    snapshot="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)"
    IFS=',' read -r free_mib utilization <<< "${snapshot}"
    free_mib="${free_mib// /}"
    utilization="${utilization// /}"
    echo "GPU ${gpu} free_mib=${free_mib} utilization=${utilization}"
    if (( free_mib < MIN_FREE_MIB || utilization > MAX_UTILIZATION )); then ready=0; fi
  done
  if (( ready )); then consecutive=$((consecutive + 1)); else consecutive=0; fi
  if (( consecutive < READY_CHECKS )); then sleep "${WAIT_SECONDS}"; fi
done

methods=(H0 HF)
pids=()
for index in 0 1; do
  method="${methods[$index]}"
  run_id="${BATCH_ID}_${method}_seed0_val"
  (
    GPU="${gpus[$index]}" METHOD="${method}" RUN_ID="${run_id}" \
      OUTPUT_BASE="${RESULT_BASE}" LOG_DIR="${LOG_DIR}" \
      bash scripts/emotic/run_multilane_track_a_full_private_head_val.sh
    printf '0\n' > "${CONTROL_DIR}/status/${method}.exit_code"
  ) > "${LOG_DIR}/${method}.launcher.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in 0 1; do
  if ! wait "${pids[$index]}"; then
    printf '1\n' > "${CONTROL_DIR}/status/${methods[$index]}.exit_code"
    status=1
  fi
done
[[ ${status} -eq 0 ]] || { echo "FULL_PRIVATE_HEAD_BATCH_FAILED" >&2; exit 1; }

"${PYTHON:-/opt/conda/envs/ddp/bin/python}" \
  -m multi_lane.track_a.summarize_full_private_head \
  --h0-run "${RESULT_BASE}/${BATCH_ID}_H0_seed0_val" \
  --hf-run "${RESULT_BASE}/${BATCH_ID}_HF_seed0_val" \
  --bootstrap-replicates 2000 \
  --output "${CONTROL_DIR}/validation_summary.json" \
  2>&1 | tee "${LOG_DIR}/summary.log"
printf '0\n' > "${CONTROL_DIR}/status/summary.exit_code"
echo "FULL_PRIVATE_HEAD_BATCH_COMPLETE batch=${BATCH_ID} summary=${CONTROL_DIR}/validation_summary.json"
