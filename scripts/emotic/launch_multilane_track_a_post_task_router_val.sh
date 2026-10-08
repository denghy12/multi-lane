#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BATCH_ID="${BATCH_ID:-post_task_router_seed0_val_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-1 2}"
read -r -a GPUS <<< "${GPU_LIST}"
[[ "${#GPUS[@]}" -eq 2 ]] || {
  echo "GPU_LIST must contain exactly two GPU indices" >&2; exit 2;
}
shared_gpu=0
if [[ "${GPUS[0]}" == "${GPUS[1]}" ]]; then
  shared_gpu=1
fi
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-./datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-./models/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-./output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
RESULT_ROOT="${ROOT}/output/emotic_track_a_post_task_router_val"
LOG_ROOT="${ROOT}/logs/emotic_track_a_post_task_router_val"
CONTROL="${ROOT}/output/emotic_track_a_post_task_router_control/${BATCH_ID}"
METHODS=(B0 POST_TASK_ROUTER)
[[ ! -e "${CONTROL}" && ! -e "${RESULT_ROOT}/${BATCH_ID}" ]] || {
  echo "Batch output already exists: ${BATCH_ID}" >&2; exit 2;
}
for path in "${CLIP_CHECKPOINT}" "${DATA_ROOT}/CVPR17_Annotations.mat" \
  "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" \
  "${FACE_MANIFEST_ROOT}/manifests/val.jsonl"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
mkdir -p "${CONTROL}/status" "${LOG_ROOT}/${BATCH_ID}"
if (( shared_gpu )); then
  echo "Waiting for GPU ${GPUS[0]} before starting both methods concurrently"
  while true; do
    free_mib="$(nvidia-smi -i "${GPUS[0]}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    (( free_mib >= MIN_FREE_MIB )) && break
    sleep "${POLL_SECONDS}"
  done
fi
cat > "${CONTROL}/experiment_manifest.txt" <<EOF_MANIFEST
batch=${BATCH_ID}
branch=$(git branch --show-current)
code_commit=$(git rev-parse HEAD)
methods=${METHODS[*]}
gpu_indices=${GPUS[*]}
same_gpu_parallel=${shared_gpu}
dataset=EMOTIC Track-A train; tasks 0-2; validation only; test forbidden
inputs=${DATA_ROOT}; ${CLIP_CHECKPOINT}; ${FACE_MANIFEST_ROOT}
seed=0; epochs=30/task; train/eval batch=64; Adam reset/task; main lr=0.0125; ParaX lr=0.0004; cosine; weight decay=0; AMP+TF32
model=frozen CLIP ViT-B/16 and Frozen Forward; shared Selector/Prompt/classifier; no Image-token Adapter
post_route=after Task Forward final lane CLS features; shared rank32 center/three parameter-matrix experts; task0 learns center, then center frozen; task-local router hidden16 trains only on its task and then freezes; Full/Person/Face share center and per-task router
initialization=zero output; fixed residual scale=0.001; no Projector; fixed reliable Face fusion [0.64,0.16,0.20] or [0.80,0.20,0]
loss=legacy_full_zero BCE with auxiliary view weight 0.1; joint loss routing
output=${RESULT_ROOT}/${BATCH_ID}; logs=${LOG_ROOT}/${BATCH_ID}
EOF_MANIFEST
pids=()
for index in "${!METHODS[@]}"; do
  method="${METHODS[index]}"; gpu="${GPUS[index]}"
  (
    echo "gpu=${gpu}" > "${CONTROL}/status/${method}.waiting.txt"
    if (( ! shared_gpu )); then
      while true; do
        free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
        (( free_mib >= MIN_FREE_MIB )) && break
        sleep "${POLL_SECONDS}"
      done
    fi
    mv "${CONTROL}/status/${method}.waiting.txt" "${CONTROL}/status/${method}.started.txt"
    METHOD="${method}" GPU="${gpu}" RUN_ID="${BATCH_ID}" SEED=0 \
      MAX_TASKS=3 EPOCHS=30 PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" \
      CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" \
      OUTPUT_ROOT="${RESULT_ROOT}" LOG_DIR="${LOG_ROOT}" \
      bash scripts/emotic/run_multilane_track_a_post_task_router_val.sh \
      > "${LOG_ROOT}/${BATCH_ID}/${method}.launcher.log" 2>&1
    printf '0\n' > "${CONTROL}/status/${method}.exit_code"
  ) &
  pids+=("$!")
done
failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[index]}"; then
    printf '1\n' > "${CONTROL}/status/${METHODS[index]}.exit_code"
    failed=1
  fi
done
if (( failed )); then
  echo "Post-task router validation failed; inspect status and logs" >&2
  exit 1
fi
printf '%s\n' "${METHODS[*]} complete" > "${CONTROL}/status/complete.txt"
echo "POST_TASK_ROUTER_VALIDATION_COMPLETE batch=${BATCH_ID}"
