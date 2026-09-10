#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BATCH_ID="${BATCH_ID:-full_anchored_residual_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1,2}"
IFS=',' read -r -a gpus <<< "${GPU_LIST}"
[[ ${#gpus[@]} -eq 3 ]] || { echo "GPU_LIST must contain exactly three GPUs" >&2; exit 2; }
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_full_anchored_residual_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_full_anchored_residual/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_full_anchored_residual/${BATCH_ID}"
mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}" "${RESULT_BASE}"

cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF
batch=${BATCH_ID}
selection=seed0 complete 8-task validation only; test forbidden
A0=fresh Full champion anchor
A1=Full coefficient1 + no-Full-Adapter detached Person taskwise zero-init bounded residual
A2=Full coefficient1 + no-Full-Adapter detached Person/valid-Face taskwise zero-init bounded residuals
residual=bottleneck16, GELU, task scalar sigmoid gate, coefficient range [0,0.1], no auxiliary branch loss
advance=A2 final validation mAP must exceed both A0 and A1
EOF

methods=(A0 A1 A2)
pids=()
for index in 0 1 2; do
  method="${methods[$index]}"
  run_id="${BATCH_ID}_${method}_seed0_val"
  (
    GPU="${gpus[$index]}" METHOD="${method}" RUN_ID="${run_id}" \
      OUTPUT_BASE="${RESULT_BASE}" LOG_DIR="${LOG_DIR}" \
      bash scripts/emotic/run_multilane_track_a_full_anchored_residual_val.sh
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
[[ ${status} -eq 0 ]] || { echo "FULL_ANCHORED_RESIDUAL_BATCH_FAILED" >&2; exit 1; }

"${PYTHON:-/opt/conda/envs/ddp/bin/python}" \
  -m multi_lane.track_a.summarize_full_anchored_residual \
  --a0-run "${RESULT_BASE}/${BATCH_ID}_A0_seed0_val" \
  --a1-run "${RESULT_BASE}/${BATCH_ID}_A1_seed0_val" \
  --a2-run "${RESULT_BASE}/${BATCH_ID}_A2_seed0_val" \
  --output "${CONTROL_DIR}/validation_summary.json" \
  2>&1 | tee "${LOG_DIR}/summary.log"
printf '0\n' > "${CONTROL_DIR}/status/summary.exit_code"
echo "FULL_ANCHORED_RESIDUAL_BATCH_COMPLETE batch=${BATCH_ID} result=${RESULT_BASE}"
