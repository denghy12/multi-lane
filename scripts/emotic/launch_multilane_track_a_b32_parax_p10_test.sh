#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BATCH_ID="${BATCH_ID:-b32_parax_p10_seed1_test_20260929_01}"
GPU_LIST="${GPU_LIST:-5,6}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-./datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-./models/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-./output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
RESULT_ROOT="${ROOT}/output/emotic_track_a_b32_parax_p10_test"
CONTROL_DIR="${ROOT}/output/emotic_track_a_b32_parax_p10_control/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_b32_parax_p10_test/${BATCH_ID}"
METHODS=(B32 B32_P10)
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
[[ ${#GPUS[@]} -eq ${#METHODS[@]} ]] || { echo "GPU_LIST must contain two GPU indices" >&2; exit 2; }
[[ ! -e "${CONTROL_DIR}" && ! -e "${RESULT_ROOT}/${BATCH_ID}" ]] || {
  echo "Batch output already exists: ${BATCH_ID}" >&2; exit 2;
}
for path in \
  "${CLIP_CHECKPOINT}" \
  "${DATA_ROOT}/CVPR17_Annotations.mat" \
  "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" \
  "${FACE_MANIFEST_ROOT}/manifests/test.jsonl"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done

mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}"
cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF_MANIFEST
batch=${BATCH_ID}
branch=$(git branch --show-current)
code_commit=$(git rev-parse HEAD)
methods=B32 B32_P10
dataset=EMOTIC Track-A train split; 8 incremental tasks; held-out test after each task
inputs=${DATA_ROOT}; ${CLIP_CHECKPOINT}; ${FACE_MANIFEST_ROOT}
seed=1; 30 epochs/task; train/eval batch64; Adam reset/task; main lr0.0125; Adapter/ParaX lr0.0004; cosine; weight decay0; AMP+TF32
model=frozen CLIP ViT-B/16; shared Selector/Prompt/classifier; shared b32 Image-token Adapter at index1, fixed scale0.03
P10=ParaX image-stream patch-only after index10; rank32; 3 experts; router hidden16; zero-output; fixed scale0.001; train center task0, freeze thereafter; image-only router
loss=main BCE; Adapter and ParaX ASL(negative gamma9.8, positive gamma0, clip0.05); auxiliary view loss0.1
fusion=fixed feature weights reliable Face [0.64,0.16,0.20], otherwise [0.80,0.20,0]
policy=skip validation evaluation; no test-side search; no checkpoints
results=${RESULT_ROOT}/${BATCH_ID}
logs=${LOG_DIR}
EOF_MANIFEST

echo "waiting for GPUs ${GPU_LIST} to each have ${MIN_FREE_MIB} MiB free" | tee "${CONTROL_DIR}/status/waiting.txt"
while true; do
  ready=1
  for gpu in "${GPUS[@]}"; do
    free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if (( free_mib < MIN_FREE_MIB )); then ready=0; fi
  done
  (( ready == 1 )) && break
  sleep "${POLL_SECONDS}"
done
mv "${CONTROL_DIR}/status/waiting.txt" "${CONTROL_DIR}/status/ready.txt"
date -Is > "${CONTROL_DIR}/status/started.txt"

pids=()
for index in "${!METHODS[@]}"; do
  method="${METHODS[$index]}"
  gpu="${GPUS[$index]}"
  (
    METHOD="${method}" RUN_ID="${BATCH_ID}" GPU="${gpu}" SEED=1 \
      PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" \
      CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" \
      OUTPUT_ROOT="${RESULT_ROOT}" LOG_DIR="${ROOT}/logs/emotic_track_a_b32_parax_p10_test" \
      bash scripts/emotic/run_multilane_track_a_b32_parax_p10_test.sh
    printf '0\n' > "${CONTROL_DIR}/status/${method}.exit_code"
  ) > "${LOG_DIR}/${method}.launcher.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    printf '1\n' > "${CONTROL_DIR}/status/${METHODS[$index]}.exit_code"
    status=1
  fi
done
(( status == 0 )) || { echo "B32_PARAX_P10_TEST_FAILED" >&2; exit 1; }
printf '%s\n' "${METHODS[*]} complete" > "${CONTROL_DIR}/status/complete.txt"
echo "B32_PARAX_P10_TEST_COMPLETE batch=${BATCH_ID} results=${RESULT_ROOT}/${BATCH_ID}"
