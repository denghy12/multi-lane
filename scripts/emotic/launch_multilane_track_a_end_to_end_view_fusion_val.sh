#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BATCH_ID="${BATCH_ID:-end_to_end_view_fusion_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1,2}"
IFS=',' read -r -a gpus <<< "${GPU_LIST}"
[[ ${#gpus[@]} -eq 3 ]] || { echo "GPU_LIST must contain exactly three GPUs" >&2; exit 2; }
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_end_to_end_view_fusion_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_end_to_end_view_fusion/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_end_to_end_view_fusion/${BATCH_ID}"
mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}" "${RESULT_BASE}"

cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF
batch=${BATCH_ID}
selection=seed0 complete 8-task validation only; test forbidden
J0=fixed reliable-Face feature fusion + auxiliary supervision
J1=taskwise sample-soft Full/Person/Face feature fusion + auxiliary supervision
J3=taskwise sample-soft Full/Person feature fusion + auxiliary supervision
advance=J1 final validation mAP must exceed both J0 and J3
EOF

methods=(J0 J1 J3)
pids=()
for index in 0 1 2; do
  method="${methods[$index]}"
  run_id="${BATCH_ID}_${method}_seed0_val"
  (
    GPU="${gpus[$index]}" METHOD="${method}" RUN_ID="${run_id}" \
      OUTPUT_BASE="${RESULT_BASE}" LOG_DIR="${LOG_DIR}" \
      bash scripts/emotic/run_multilane_track_a_end_to_end_view_fusion_val.sh
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
[[ ${status} -eq 0 ]] || { echo "END_TO_END_VIEW_FUSION_BATCH_FAILED" >&2; exit 1; }

"${PYTHON:-/opt/conda/envs/ddp/bin/python}" \
  -m multi_lane.track_a.summarize_end_to_end_view_fusion \
  --j0-run "${RESULT_BASE}/${BATCH_ID}_J0_seed0_val" \
  --j1-run "${RESULT_BASE}/${BATCH_ID}_J1_seed0_val" \
  --j3-run "${RESULT_BASE}/${BATCH_ID}_J3_seed0_val" \
  --output "${CONTROL_DIR}/validation_summary.json" \
  2>&1 | tee "${LOG_DIR}/summary.log"
printf '0\n' > "${CONTROL_DIR}/status/summary.exit_code"
echo "END_TO_END_VIEW_FUSION_BATCH_COMPLETE batch=${BATCH_ID} result=${RESULT_BASE}"
