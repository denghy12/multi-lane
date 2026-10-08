#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BATCH_ID="${BATCH_ID:-parax_p10_projector_seed0_val_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-./datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-./models/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-./output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
RESULT_ROOT="${ROOT}/output/emotic_track_a_parax_projector_val"
CONTROL_DIR="${ROOT}/output/emotic_track_a_parax_projector_control/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_parax_projector_val/${BATCH_ID}"
METHODS=(B0 P10 PROJECTOR_ONLY P10_PROJECTOR P10_PROJECTOR_ALIGN)

[[ ! -e "${CONTROL_DIR}" && ! -e "${RESULT_ROOT}/${BATCH_ID}" ]] || {
  echo "Batch output already exists: ${BATCH_ID}" >&2; exit 2;
}
for path in \
  "${CLIP_CHECKPOINT}" \
  "${DATA_ROOT}/CVPR17_Annotations.mat" \
  "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" \
  "${FACE_MANIFEST_ROOT}/manifests/val.jsonl"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}"

cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF_MANIFEST
batch=${BATCH_ID}
branch=$(git branch --show-current)
code_commit=$(git rev-parse HEAD)
methods=${METHODS[*]}
dataset=EMOTIC Track-A train split; task0-2; validation after each task; test forbidden
inputs=${DATA_ROOT}; ${CLIP_CHECKPOINT}; ${FACE_MANIFEST_ROOT}
seed=0; epochs=30/task; train/eval batch=64; optimizer=Adam reset/task; main lr=0.0125; ParaX/Projector lr=0.0004; cosine; weight decay=0; AMP+TF32
model=frozen CLIP ViT-B/16; shared Selector/Prompt/classifier; no Image-token Adapter
P10=Frozen Forward after block 11 / code index 10; patch tokens only; rank 32; 3 expert pairs; router hidden 16; official initialization; initial scale 0.1 learnable; expert center and Router continually updated on all tasks
Projector=shared across Full/Person/Face and tasks; patch-token residual MLP 768-64-768; zero-initialized output; continually updated
Projector_only=ParaX residual exactly zero via fixed output scale 0
Projector_align=cosine alignment against pre-ParaX frozen patch tokens; weight 0.1
fusion=fixed reliable Face feature weights [0.64,0.16,0.20] or [0.80,0.20,0]
loss=legacy_full_zero BCE with auxiliary view weight 0.1; joint loss routing
results=${RESULT_ROOT}/${BATCH_ID}
logs=${LOG_DIR}
EOF_MANIFEST

echo "waiting for GPU ${GPU} with ${MIN_FREE_MIB} MiB free" > "${CONTROL_DIR}/status/waiting.txt"
while true; do
  free_mib="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free_mib >= MIN_FREE_MIB )) && break
  sleep "${POLL_SECONDS}"
done
mv "${CONTROL_DIR}/status/waiting.txt" "${CONTROL_DIR}/status/ready.txt"
date -Is > "${CONTROL_DIR}/status/started.txt"

for method in "${METHODS[@]}"; do
  echo "starting ${method}" > "${CONTROL_DIR}/status/${method}.started.txt"
  if METHOD="${method}" RUN_ID="${BATCH_ID}" GPU="${GPU}" SEED=0 \
      PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" \
      CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" \
      OUTPUT_ROOT="${RESULT_ROOT}" LOG_DIR="${ROOT}/logs/emotic_track_a_parax_projector_val" \
      bash scripts/emotic/run_multilane_track_a_parax_projector_val.sh \
      > "${LOG_DIR}/${method}.launcher.log" 2>&1; then
    printf '0\n' > "${CONTROL_DIR}/status/${method}.exit_code"
  else
    printf '1\n' > "${CONTROL_DIR}/status/${method}.exit_code"
    echo "Projector validation failed at ${method}" >&2
    exit 1
  fi
done
printf '%s\n' "${METHODS[*]} complete" > "${CONTROL_DIR}/status/complete.txt"
echo "PARAX_PROJECTOR_VALIDATION_COMPLETE batch=${BATCH_ID} results=${RESULT_ROOT}/${BATCH_ID}"
