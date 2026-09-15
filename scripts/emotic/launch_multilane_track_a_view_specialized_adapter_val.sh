#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BATCH_ID="${BATCH_ID:-view_specialized_adapter_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1,2}"
IFS=',' read -r -a gpus <<< "${GPU_LIST}"
[[ ${#gpus[@]} -eq 3 ]] || { echo "GPU_LIST must contain exactly three GPUs" >&2; exit 2; }
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_view_specialized_adapter_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_view_specialized_adapter/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_view_specialized_adapter/${BATCH_ID}"
mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}" "${RESULT_BASE}"

cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF
batch=${BATCH_ID}
selection=seed0 complete 8-task validation only; test forbidden
P0=shared Image-token Adapter b32, 49952 parameters/task
P1=parameter-matched shared Image-token Adapter b45, 69933 parameters/task
P2=shared b32 plus Full/Person/Face b4 deltas, 70700 parameters/task
fusion=fixed reliable-Face weights, joint gradients, auxiliary0.1
audit=epochs 0/14/29, first 3 batches; exact per-view fused-loss VJP versus unimodal gradient for representation BCE and Adapter ASL
advance=P2 final >= P0+0.10 and P1+0.10; average and standalone Full not below P0; gap to independent R1 43.5812 shrinks
EOF

methods=(P0 P1 P2)
pids=()
for index in 0 1 2; do
  method="${methods[$index]}"
  run_id="${BATCH_ID}_${method}_seed0_val"
  (
    GPU="${gpus[$index]}" METHOD="${method}" RUN_ID="${run_id}" \
      OUTPUT_BASE="${RESULT_BASE}" LOG_DIR="${LOG_DIR}" \
      bash scripts/emotic/run_multilane_track_a_view_specialized_adapter_val.sh
    printf '0\n' > "${CONTROL_DIR}/status/${method}.exit_code"
  ) > "${LOG_DIR}/${method}.launcher.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in 0 1 2; do
  if ! wait "${pids[$index]}"; then
    printf '1\n' > "${CONTROL_DIR}/status/${methods[$index]}.exit_code"
    status=1
  fi
done
[[ ${status} -eq 0 ]] || { echo "VIEW_SPECIALIZED_ADAPTER_BATCH_FAILED" >&2; exit 1; }

"${PYTHON:-/opt/conda/envs/ddp/bin/python}" \
  -m multi_lane.track_a.summarize_view_specialized_adapter \
  --p0-run "${RESULT_BASE}/${BATCH_ID}_P0_seed0_val" \
  --p1-run "${RESULT_BASE}/${BATCH_ID}_P1_seed0_val" \
  --p2-run "${RESULT_BASE}/${BATCH_ID}_P2_seed0_val" \
  --output "${CONTROL_DIR}/validation_summary.json" \
  2>&1 | tee "${LOG_DIR}/summary.log"
printf '0\n' > "${CONTROL_DIR}/status/summary.exit_code"
echo "VIEW_SPECIALIZED_ADAPTER_BATCH_COMPLETE batch=${BATCH_ID} result=${RESULT_BASE}"
