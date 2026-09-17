#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BATCH_ID="${BATCH_ID:-view_private_components_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
[[ ${#GPUS[@]} -eq 8 ]] || { echo "GPU_LIST must contain exactly 8 GPUs" >&2; exit 2; }
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_view_private_components/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_view_private_components/${BATCH_ID}"
RESULT_BASE="${CONTROL_DIR}/runs"
mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}" "${RESULT_BASE}"

cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF
batch=${BATCH_ID}
selection=seed0 complete 8-task validation only; test forbidden
constant=shared frozen CLIP, view-specific Selector x10, EMOTIC protocol, fixed reliable-Face fusion and training schedule
P0=view-specific Selector; shared Prompt, shared Adapter, shared post-fusion classifier
P1=view-specific Prompt; shared Adapter, shared post-fusion classifier
P2=independent Full/Person/Face Image-token Adapter b32; shared Prompt and post-fusion classifier
P3=independent Prompt and Adapter; shared post-fusion classifier
H0=shared Prompt/Adapter; one shared per-view classifier; fixed logit fusion
H1=shared Prompt/Adapter; independent Full/Person/Face classifiers; fixed logit fusion
H2=independent Prompt/Adapter; one shared per-view classifier; fixed logit fusion
H3=independent Prompt/Adapter and independent Full/Person/Face classifiers; fixed logit fusion
evaluation=per-view score dumps and diagnostics; no checkpoints
EOF

for gpu in "${GPUS[@]}"; do
  snapshot="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits)"
  snapshot="${snapshot// /}"
  (( snapshot >= MIN_FREE_MIB )) || {
    echo "GPU ${gpu} has only ${snapshot} MiB free; need ${MIN_FREE_MIB}" >&2
    exit 3
  }
done

methods=(P0 P1 P2 P3 H0 H1 H2 H3)
pids=()
for index in "${!methods[@]}"; do
  method="${methods[$index]}"
  run_id="${BATCH_ID}_${method}_seed0_val"
  (
    GPU="${GPUS[$index]}" METHOD="${method}" RUN_ID="${run_id}" \
      OUTPUT_BASE="${RESULT_BASE}" LOG_DIR="${LOG_DIR}" \
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
  echo "VIEW_PRIVATE_COMPONENTS_BATCH_FAILED" >&2
  exit 1
fi
printf 'P0 P1 P2 P3 H0 H1 H2 H3 complete\n' > "${CONTROL_DIR}/status/complete.txt"
echo "VIEW_PRIVATE_COMPONENTS_BATCH_COMPLETE batch=${BATCH_ID} output=${CONTROL_DIR}"
