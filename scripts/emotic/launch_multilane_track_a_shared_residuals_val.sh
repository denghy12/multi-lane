#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BATCH_ID="${BATCH_ID:-shared_residuals_late_prompt_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
[[ ${#GPUS[@]} -eq 4 ]] || { echo "GPU_LIST must contain exactly 4 GPUs" >&2; exit 2; }
MIN_FREE_MIB="${MIN_FREE_MIB:-5000}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_shared_residuals/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_shared_residuals/${BATCH_ID}"
RESULT_BASE="${CONTROL_DIR}/runs"
mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}" "${RESULT_BASE}"

cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF
batch=${BATCH_ID}
selection=seed0 complete 8-task validation only; test forbidden
precision=FP32 autocast disabled; TF32 enabled; gradient clipping disabled
constant=EMOTIC Track A, view fusion fixed reliable-Face 0.64/0.16/0.20, joint gradient, auxiliary view loss 0.1, frozen CLIP
schedule=30 epochs/task; train/eval batch 64; workers 2; Adam reset per task; main LR 0.0125; adapter LR 0.0004; cosine min ratio 0; no warmup; weight decay 0
selector=R0 shared; R1 shared plus zero-initialized Full/Person/Face residual scale 0.1 with L2 coefficient 0.1
adapter=R0/R1/R3 shared Image-token Adapter; R2 shared base plus zero-initialized Full/Person/Face residual bottleneck32
prompt=R0/R1/R2 shared all layers; R3 first 3 layers shared and last 2 layers Full/Face private residual banks; Person shared
arms=R0 R1 R2 R3
outputs=${RESULT_BASE}
logs=${LOG_DIR}
EOF

for gpu in "${GPUS[@]}"; do
  snapshot="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits)"
  snapshot="${snapshot// /}"
  (( snapshot >= MIN_FREE_MIB )) || { echo "GPU ${gpu} has only ${snapshot} MiB free; need ${MIN_FREE_MIB}" >&2; exit 3; }
done

methods=(R0 R1 R2 R3)
pids=()
for index in "${!methods[@]}"; do
  method="${methods[$index]}"
  run_id="${BATCH_ID}_${method}_seed0_val"
  residual_reg=0
  [[ "${method}" == R1 ]] && residual_reg=0.1
  (
    GPU="${GPUS[$index]}" METHOD="${method}" RUN_ID="${run_id}" \
      OUTPUT_BASE="${RESULT_BASE}" LOG_DIR="${LOG_DIR}" NO_AMP=1 \
      SELECTOR_RESIDUAL_REG="${residual_reg}" \
      bash scripts/emotic/run_multilane_track_a_view_private_components_val.sh
    printf '0\n' > "${CONTROL_DIR}/status/${method}.exit_code"
  ) > "${LOG_DIR}/${method}.launcher.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!methods[@]}"; do
  if ! wait "${pids[$index]}"; then
    printf '1\n' > "${CONTROL_DIR}/status/${methods[$index]}.exit_code"
    status=1
  fi
done
if (( status != 0 )); then
  echo "SHARED_RESIDUALS_BATCH_FAILED" >&2
  exit 1
fi
printf 'R0 R1 R2 R3 complete\n' > "${CONTROL_DIR}/status/complete.txt"
echo "SHARED_RESIDUALS_BATCH_COMPLETE batch=${BATCH_ID} output=${CONTROL_DIR}"
