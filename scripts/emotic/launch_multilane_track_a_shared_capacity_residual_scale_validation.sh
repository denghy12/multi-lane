#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BATCH_ID="${BATCH_ID:-shared_capacity_residual_scale_validation_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1}"
IFS=',' read -r -a gpus <<< "${GPU_LIST}"
[[ ${#gpus[@]} -eq 2 ]] || { echo "GPU_LIST must contain exactly two GPUs" >&2; exit 2; }
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_shared_capacity_residual_scale_v0.1}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_shared_capacity_residual_scale/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_shared_capacity_residual_scale/${BATCH_ID}"
mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}" "${RESULT_BASE}"
cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF
batch=${BATCH_ID}
selection=paired seed0 tasks0-7 EMOTIC validation only; test forbidden
dataset=EMOTIC Track-A aligned Full/Person/Face
backbone=frozen OpenAI CLIP ViT-B/16
A0=shared Image-token Adapter b32, 49952 parameters/task, residual_scale=0.03
A-cap=shared Image-token Adapter b97, 149857 parameters/task, residual_scale=0.03
loss=shared BCE; Adapter ASL gamma_neg9.8 gamma_pos0 clip0.05; auxiliary0.1; joint view gradients
schedule=30 epochs/task; batch64; Adam reset/task; main lr0.0125; Adapter lr0.0004; cosine min0; AMP+TF32
decision=compare capacity at a controlled residual scale; no routing/level/private components; no test
EOF
for gpu in "${gpus[@]}"; do
  free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free_mib >= MIN_FREE_MIB )) || { echo "GPU ${gpu} has only ${free_mib} MiB free" >&2; exit 3; }
done
methods=(A0-shared-b32-scale003 A-cap-shared-b97-scale003)
pids=()
for index in 0 1; do
  method="${methods[${index}]}"
  run_id="${BATCH_ID}_${method}_seed0_val"
  ( GPU="${gpus[${index}]}" METHOD="${method}" RUN_ID="${run_id}" OUTPUT_BASE="${RESULT_BASE}" LOG_DIR="${LOG_DIR}" \
    ADAPTER_RESIDUAL_SCALE=0.03 bash scripts/emotic/run_multilane_track_a_shared_capacity_scale_validation.sh; \
    printf '0\n' > "${CONTROL_DIR}/status/${method}.exit_code" ) > "${LOG_DIR}/${method}.launcher.log" 2>&1 &
  pids+=("$!")
done
status=0
for index in 0 1; do
  if ! wait "${pids[${index}]}"; then
    printf '1\n' > "${CONTROL_DIR}/status/${methods[${index}]}.exit_code"
    status=1
  fi
done
[[ ${status} -eq 0 ]] || { echo "SHARED_CAPACITY_RESIDUAL_SCALE_BATCH_FAILED" >&2; exit 1; }
printf '%s\n' "${methods[*]} complete" > "${CONTROL_DIR}/status/complete.txt"
echo "SHARED_CAPACITY_RESIDUAL_SCALE_BATCH_COMPLETE batch=${BATCH_ID} result=${RESULT_BASE}"
