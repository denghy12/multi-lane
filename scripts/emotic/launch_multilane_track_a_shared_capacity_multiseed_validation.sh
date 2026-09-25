#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BATCH_ID="${BATCH_ID:-shared_capacity_residual_scale_multiseed_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1}"
SEEDS="${SEEDS:-1,2}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_shared_capacity_residual_scale_multiseed_v0.1}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_shared_capacity_multiseed_validation/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_shared_capacity_multiseed_validation/${BATCH_ID}"

IFS=',' read -r -a gpus <<< "${GPU_LIST}"
IFS=',' read -r -a seeds <<< "${SEEDS}"
[[ ${#gpus[@]} -eq 2 ]] || { echo "GPU_LIST must contain exactly two GPUs" >&2; exit 2; }
[[ "${SEEDS}" == "1,2" ]] || { echo "SEEDS is locked to 1,2" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || {
  echo "Validation launch requires a clean Git worktree" >&2
  exit 2
}
[[ ! -e "${CONTROL_DIR}" ]] || { echo "Control directory already exists: ${CONTROL_DIR}" >&2; exit 2; }

for gpu in "${gpus[@]}"; do
  free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free_mib >= MIN_FREE_MIB )) || {
    echo "GPU ${gpu} has only ${free_mib} MiB free" >&2
    exit 3
  }
done

mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}" "${RESULT_BASE}"
cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF
batch=${BATCH_ID}
purpose=paired validation replication of the seed0 residual-scale capacity comparison
selection=seed1,seed2; tasks0-7; validation only; test forbidden
dataset=EMOTIC Track-A aligned Full/Person/Face
backbone=frozen OpenAI CLIP ViT-B/16
A0=shared Image-token Adapter b32, 49952 parameters/task, residual_scale=0.03
A-cap=shared Image-token Adapter b97, 149857 parameters/task, residual_scale=0.03
loss=shared BCE; Adapter ASL gamma_neg=9.8 gamma_pos=0 clip=0.05; auxiliary=0.1; joint view gradients
schedule=30 epochs/task; batch64; Adam reset/task; main_lr=0.0125; Adapter_lr=0.0004; cosine min0; AMP+TF32
fusion=fixed reliable-Face prior; no Router/level embedding/private bank/distillation
execution=seed1 pair on GPUs ${gpus[*]}, then seed2 pair on same GPUs
results=${RESULT_BASE}/${BATCH_ID}; logs=${LOG_DIR}; control=${CONTROL_DIR}
EOF

methods=(A0-shared-b32-scale003 A-cap-shared-b97-scale003)
for seed in "${seeds[@]}"; do
  pids=()
  for index in 0 1; do
    method="${methods[${index}]}"
    run_id="${BATCH_ID}_${method}_seed${seed}_val"
    (
      GPU="${gpus[${index}]}" SEED="${seed}" METHOD="${method}" RUN_ID="${run_id}" \
        OUTPUT_BASE="${RESULT_BASE}/${BATCH_ID}" LOG_DIR="${LOG_DIR}" \
        ADAPTER_RESIDUAL_SCALE=0.03 \
        bash scripts/emotic/run_multilane_track_a_shared_capacity_seed_validation.sh
      printf '%s\n' '0' > "${CONTROL_DIR}/status/${method}_seed${seed}.exit_code"
    ) > "${LOG_DIR}/${method}_seed${seed}.launcher.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for index in 0 1; do
    if ! wait "${pids[${index}]}"; then
      method="${methods[${index}]}"
      printf '%s\n' '1' > "${CONTROL_DIR}/status/${method}_seed${seed}.exit_code"
      failed=1
    fi
  done
  if (( failed )); then
    echo "SHARED_CAPACITY_MULTISEED_VALIDATION_FAILED seed=${seed}" >&2
    exit 1
  fi
done

printf '%s\n' "${methods[*]} seeds=${SEEDS} complete" > "${CONTROL_DIR}/status/complete.txt"
echo "SHARED_CAPACITY_MULTISEED_VALIDATION_BATCH_COMPLETE batch=${BATCH_ID} result=${RESULT_BASE}/${BATCH_ID}"
