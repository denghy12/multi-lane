#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BATCH_ID="${BATCH_ID:-parax_projector_full_dual_seed0_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-./datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-./models/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-./output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
GPU_LIST="${GPU_LIST:-1 2 3 4 5}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RESULT_ROOT="${ROOT}/output/emotic_track_a_parax_projector_full_dual"
LOG_ROOT="${ROOT}/logs/emotic_track_a_parax_projector_full_dual"
CONTROL="${ROOT}/output/emotic_track_a_parax_projector_full_control/${BATCH_ID}"
METHODS=(B0 P10 PROJECTOR_ONLY P10_PROJECTOR P10_PROJECTOR_ALIGN)
read -r -a GPUS <<< "${GPU_LIST}"
[[ "${#GPUS[@]}" == "${#METHODS[@]}" ]] || {
  echo "GPU_LIST must contain exactly five GPU indices" >&2; exit 2;
}
[[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -eq 5 ]] || {
  echo "GPU_LIST indices must be unique" >&2; exit 2;
}
[[ ! -e "${CONTROL}" && ! -e "${RESULT_ROOT}/${BATCH_ID}" ]] || {
  echo "Batch output already exists: ${BATCH_ID}" >&2; exit 2;
}
for path in \
  "${CLIP_CHECKPOINT}" \
  "${DATA_ROOT}/CVPR17_Annotations.mat" \
  "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" \
  "${FACE_MANIFEST_ROOT}/manifests/val.jsonl" \
  "${FACE_MANIFEST_ROOT}/manifests/test.jsonl"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
mkdir -p "${CONTROL}/status" "${LOG_ROOT}/${BATCH_ID}"
cat > "${CONTROL}/experiment_manifest.txt" <<EOF_MANIFEST
batch=${BATCH_ID}
branch=$(git branch --show-current)
code_commit=$(git rev-parse HEAD)
methods=${METHODS[*]}
gpu_indices=${GPUS[*]}
dataset=EMOTIC Track-A train; tasks 0-7; same checkpoints evaluated on validation and held-out test after each task
inputs=${DATA_ROOT}; ${CLIP_CHECKPOINT}; ${FACE_MANIFEST_ROOT}
seed=0; epochs=30/task; train/eval batch=64; Adam reset/task; main lr=0.0125; ParaX/Projector lr=0.0004; cosine; weight decay=0; AMP+TF32
model=frozen CLIP ViT-B/16; shared Selector/Prompt/classifier; no Image-token Adapter
ParaX=Frozen Forward after block 11/code index 10; patch tokens only; shared rank32 center with three parameter-matrix experts; router hidden16; official initialization; initial residual scale 0.1 learnable; center and Router update in every task
Projector=shared 768-64-768 residual MLP; zero-initialized output; updates in every task
Projector_only=ParaX residual exactly zero by fixed scale 0
Projector_align=normalized patch direction alignment against pre-ParaX tokens; weight 0.1
fusion=fixed reliable Face feature weights [0.64,0.16,0.20] or [0.80,0.20,0]
loss=legacy_full_zero BCE with auxiliary view weight 0.1; joint loss routing
output=${RESULT_ROOT}/${BATCH_ID}; logs=${LOG_ROOT}/${BATCH_ID}
test_use=locked reporting only; no test-based method or hyperparameter selection
EOF_MANIFEST

pids=()
for index in "${!METHODS[@]}"; do
  method="${METHODS[index]}"
  gpu="${GPUS[index]}"
  (
    echo "gpu=${gpu}" > "${CONTROL}/status/${method}.waiting.txt"
    while true; do
      free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
      (( free_mib >= MIN_FREE_MIB )) && break
      sleep "${POLL_SECONDS}"
    done
    mv "${CONTROL}/status/${method}.waiting.txt" "${CONTROL}/status/${method}.started.txt"
    METHOD="${method}" GPU="${gpu}" RUN_ID="${BATCH_ID}" SEED=0 \
      MAX_TASKS=8 EPOCHS=30 DUAL_REPORT=1 PYTHON="${PYTHON}" \
      DATA_ROOT="${DATA_ROOT}" CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
      FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" OUTPUT_ROOT="${RESULT_ROOT}" \
      LOG_DIR="${LOG_ROOT}" \
      bash scripts/emotic/run_multilane_track_a_parax_projector_val.sh \
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
  echo "At least one full dual-split run failed; inspect status and logs" >&2
  exit 1
fi
printf '%s\n' "${METHODS[*]} complete" > "${CONTROL}/status/complete.txt"
echo "PARAX_PROJECTOR_FULL_DUAL_COMPLETE batch=${BATCH_ID} results=${RESULT_ROOT}/${BATCH_ID}"
