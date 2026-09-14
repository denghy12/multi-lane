#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BATCH_ID="${BATCH_ID:-shared_dgl_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1,2}"
IFS=',' read -r -a gpus <<< "${GPU_LIST}"
[[ ${#gpus[@]} -eq 3 ]] || { echo "GPU_LIST must contain exactly three GPUs" >&2; exit 2; }
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_shared_dgl_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_shared_dgl/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_shared_dgl/${BATCH_ID}"
mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}" "${RESULT_BASE}"

cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF
batch=${BATCH_ID}
selection=seed0 complete 8-task validation only; test forbidden
G0=current J0 fixed-three-view joint gradients and auxiliary0.1
G1=fixed-three-view fused-feature detach and auxiliary0.1
G2=fixed-three-view full DGL: unimodal BCE/Adapter-ASL to representation only; fused BCE to head only; alpha1
audit=first train batch of every task; fused/view norms and cosines
diagnostics=standalone view mAP, lane weight quantiles, Full-relative corrected/damaged ranking pairs
advance=G2 final mAP >= G0+0.10, average mAP not lower, Full standalone not lower, gap to independent R1 43.5812 shrinks
EOF

methods=(G0 G1 G2)
pids=()
for index in 0 1 2; do
  method="${methods[$index]}"
  run_id="${BATCH_ID}_${method}_seed0_val"
  (
    GPU="${gpus[$index]}" METHOD="${method}" RUN_ID="${run_id}" \
      OUTPUT_BASE="${RESULT_BASE}" LOG_DIR="${LOG_DIR}" \
      bash scripts/emotic/run_multilane_track_a_shared_dgl_val.sh
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
[[ ${status} -eq 0 ]] || { echo "SHARED_DGL_BATCH_FAILED" >&2; exit 1; }

"${PYTHON:-/opt/conda/envs/ddp/bin/python}" \
  -m multi_lane.track_a.summarize_shared_dgl \
  --g0-run "${RESULT_BASE}/${BATCH_ID}_G0_seed0_val" \
  --g1-run "${RESULT_BASE}/${BATCH_ID}_G1_seed0_val" \
  --g2-run "${RESULT_BASE}/${BATCH_ID}_G2_seed0_val" \
  --output "${CONTROL_DIR}/validation_summary.json" \
  2>&1 | tee "${LOG_DIR}/summary.log"
printf '0\n' > "${CONTROL_DIR}/status/summary.exit_code"
echo "SHARED_DGL_BATCH_COMPLETE batch=${BATCH_ID} result=${RESULT_BASE}"
