#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BATCH_ID="${BATCH_ID:-parax_shared_level_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
[[ ${#GPUS[@]} -eq 6 ]] || { echo "GPU_LIST must contain six GPUs" >&2; exit 2; }
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_parax_shared_level_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_parax/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_parax/${BATCH_ID}"
mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}" "${RESULT_BASE}"
cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF
batch=${BATCH_ID}
selection=seed0 complete 8-task EMOTIC validation only; test forbidden
dataset=EMOTIC Track-A; aligned Full/Person/Face views; fixed train/validation protocol
model=frozen OpenAI CLIP ViT-B/16; shared Selector/Prompt/head; existing image-token Adapter disabled
fusion=fixed reliable-Face three-view weights; joint BCE; auxiliary view loss 0.1
schedule=30 epochs/task; batch64 train/eval; Adam reset/task; main lr0.0125; cosine min0; no warmup; wd0; AMP+TF32
parax=token-only; rank=${PARAX_RANK:-32}; experts=${PARAX_EXPERTS:-3}; router_hidden=${PARAX_ROUTER_HIDDEN:-16}; residual_scale=${PARAX_SCALE:-0.1}; initialization=${PARAX_INITIALIZATION:-official}; adapter lr0.0004
arms=B0 P-post P-10 P-8:10 P-8:10-level Static-control
logs=${LOG_DIR}
results=${RESULT_BASE}
EOF
for gpu in "${GPUS[@]}"; do
  free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free_mib >= 5000 )) || { echo "GPU ${gpu} has only ${free_mib} MiB free; need 5000" >&2; exit 3; }
done
methods=(B0 P-post P-10 P-8:10 P-8:10-level Static-control)
pids=()
for index in "${!methods[@]}"; do
  method="${methods[$index]}"
  run_id="${BATCH_ID}_${method}_seed0_val"
  ( GPU="${GPUS[$index]}" METHOD="${method}" RUN_ID="${run_id}" OUTPUT_ROOT="${RESULT_BASE}" LOG_PATH="${LOG_DIR}/${method}.log" bash scripts/emotic/run_multilane_track_a_parax_val.sh; printf '0\n' > "${CONTROL_DIR}/status/${method}.exit_code" ) > "${LOG_DIR}/${method}.launcher.log" 2>&1 &
  pids+=("$!")
done
status=0
for index in "${!methods[@]}"; do
  wait "${pids[$index]}" || { printf '1\n' > "${CONTROL_DIR}/status/${methods[$index]}.exit_code"; status=1; }
done
(( status == 0 )) || { echo "PARAX_VALIDATION_FAILED" >&2; exit 1; }
printf '%s\n' "${methods[*]} complete" > "${CONTROL_DIR}/status/complete.txt"
echo "PARAX_VALIDATION_COMPLETE batch=${BATCH_ID} results=${RESULT_BASE}"
