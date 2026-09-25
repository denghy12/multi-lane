#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BATCH_ID="${BATCH_ID:-logit_residual_control_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
[[ ${#GPUS[@]} -eq 2 ]] || { echo "GPU_LIST must contain two GPUs" >&2; exit 2; }
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_logit_residual_control_v0.1}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_logit_residual_control/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_logit_residual_control/${BATCH_ID}"
mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}" "${RESULT_BASE}"
cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF
batch=${BATCH_ID}
selection=paired seed0 task0-2 EMOTIC validation only; test forbidden
dataset=EMOTIC Track-A; aligned Full/Person/Face views
model=frozen OpenAI CLIP ViT-B/16; shared Selector/Prompt/head; Adapter and ParaX disabled
baseline=fixed reliable-Face three-view fusion; joint BCE; auxiliary view loss 0.1
candidate=baseline-preserving task-local classwise Person/Face logit residual; tanh bound 0.1; zero initialization; 52 parameters/task; detached calibration inputs
gradient_routing=ordinary B0 loss updates Selector/Prompt/head; calibrated BCE updates only logit residual coefficients
schedule=30 epochs/task; batch64; Adam reset/task; main lr0.0125; calibration lr0.0004; cosine min0; AMP+TF32
arms=B0-paired L-post-logit
logs=${LOG_DIR}
results=${RESULT_BASE}
EOF
for gpu in "${GPUS[@]}"; do
  free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free_mib >= 5000 )) || { echo "GPU ${gpu} has only ${free_mib} MiB free; need 5000" >&2; exit 3; }
done
methods=(B0-paired L-post-logit)
pids=()
for index in "${!methods[@]}"; do
  method="${methods[$index]}"
  run_id="${BATCH_ID}_${method}_seed0_val"
  ( GPU="${GPUS[$index]}" METHOD="${method}" RUN_ID="${run_id}" OUTPUT_ROOT="${RESULT_BASE}" LOG_PATH="${LOG_DIR}/${method}.log" bash scripts/emotic/run_multilane_track_a_logit_residual_control.sh; printf '0\n' > "${CONTROL_DIR}/status/${method}.exit_code" ) > "${LOG_DIR}/${method}.launcher.log" 2>&1 &
  pids+=("$!")
done
status=0
for index in "${!methods[@]}"; do
  wait "${pids[$index]}" || { printf '1\n' > "${CONTROL_DIR}/status/${methods[$index]}.exit_code"; status=1; }
done
(( status == 0 )) || { echo "LOGIT_RESIDUAL_CONTROL_FAILED" >&2; exit 1; }
printf '%s\n' "${methods[*]} complete" > "${CONTROL_DIR}/status/complete.txt"
echo "LOGIT_RESIDUAL_CONTROL_COMPLETE batch=${BATCH_ID} results=${RESULT_BASE}"
