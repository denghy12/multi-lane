#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BATCH_ID="${BATCH_ID:-fp32_interaction_selective_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
[[ ${#GPUS[@]} -eq 5 ]] || { echo "GPU_LIST must contain exactly 5 GPUs" >&2; exit 2; }
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_fp32_interaction_selective/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_fp32_interaction_selective/${BATCH_ID}"
RESULT_BASE="${CONTROL_DIR}/runs"
mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}" "${RESULT_BASE}"

cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF
batch=${BATCH_ID}
selection=seed0 complete 8-task validation only; test forbidden
precision=FP32 autocast disabled; TF32 enabled; gradient clipping disabled
constant=shared frozen CLIP, view-specific Selector x10, EMOTIC protocol, fixed reliable-Face fusion, joint gradient, auxiliary view loss 0.1
schedule=30 epochs/task; train/eval batch 64; workers 2; Adam reset per task; main LR 0.0125; adapter LR 0.0004; cosine scheduler min ratio 0; no warmup; weight decay 0
adapter=image_token layer1 bottleneck32 residual_scale0.1 ReLU independent Full/Person/Face banks; ASL gamma_neg9.8 gamma_pos0 clip0.05
F0=shared Prompt + shared Adapter
F1=view-specific Prompt + shared Adapter
F2=shared Prompt + independent Full/Person/Face Adapter
F3=view-specific Prompt + independent Full/Person/Face Adapter
S1=selective Prompt (Person shared, Full/Face private) + independent Full/Person/Face Adapter
outputs=${RESULT_BASE}
logs=${LOG_DIR}
EOF

for gpu in "${GPUS[@]}"; do
  snapshot="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits)"
  snapshot="${snapshot// /}"
  (( snapshot >= MIN_FREE_MIB )) || {
    echo "GPU ${gpu} has only ${snapshot} MiB free; need ${MIN_FREE_MIB}" >&2
    exit 3
  }
done

methods=(F0 F1 F2 F3 S1)
pids=()
for index in "${!methods[@]}"; do
  method="${methods[$index]}"
  run_id="${BATCH_ID}_${method}_seed0_val"
  (
    GPU="${GPUS[$index]}" METHOD="${method}" RUN_ID="${run_id}" \
      OUTPUT_BASE="${RESULT_BASE}" LOG_DIR="${LOG_DIR}" NO_AMP=1 \
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
  echo "FP32_INTERACTION_SELECTIVE_BATCH_FAILED" >&2
  exit 1
fi
printf 'F0 F1 F2 F3 S1 complete\n' > "${CONTROL_DIR}/status/complete.txt"
echo "FP32_INTERACTION_SELECTIVE_BATCH_COMPLETE batch=${BATCH_ID} output=${CONTROL_DIR}"
