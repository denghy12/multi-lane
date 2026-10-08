#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BATCH_ID="${BATCH_ID:-post_task_staged_seed0_val_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-./datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-./models/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-./output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
RESULT_ROOT="${ROOT}/output/emotic_track_a_post_task_staged_val"
LOG_ROOT="${ROOT}/logs/emotic_track_a_post_task_staged_val"
CONTROL="${ROOT}/output/emotic_track_a_post_task_staged_control/${BATCH_ID}"
METHODS=(B0 POST_TASK_STAGED POST_TASK_STAGED_STATIC)
[[ ! -e "${CONTROL}" && ! -e "${RESULT_ROOT}/${BATCH_ID}" ]] || {
  echo "Batch already exists: ${BATCH_ID}" >&2; exit 2;
}
for path in "${CLIP_CHECKPOINT}" "${DATA_ROOT}/CVPR17_Annotations.mat" \
  "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" "${FACE_MANIFEST_ROOT}/manifests/val.jsonl"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
free_mib="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
(( free_mib >= ${MIN_FREE_MIB:-16000} )) || { echo "Insufficient free GPU memory: ${free_mib} MiB" >&2; exit 2; }
mkdir -p "${CONTROL}/status" "${LOG_ROOT}/${BATCH_ID}"
cat > "${CONTROL}/experiment_manifest.txt" <<EOF_MANIFEST
batch=${BATCH_ID}
branch=$(git branch --show-current)
code_commit=$(git rev-parse HEAD)
methods=${METHODS[*]}
gpu=${GPU}; three arms concurrently
dataset=EMOTIC Track-A train; seed0; tasks0-2; validation only; test forbidden
inputs=${DATA_ROOT}; ${CLIP_CHECKPOINT}; ${FACE_MANIFEST_ROOT}
base_training=30 epochs/task; batch64; Adam reset/task; main lr0.0125; cosine; weight_decay0; AMP+TF32
model=frozen CLIP ViT-B/16; shared Selector/Prompt/classifier across views; no Image-token Adapter
route_training=after base training; cached train features with evaluation transforms; FP32; 5 epochs/task; Adam lr0.0004 cosine; gradient_clip1.0
route_position=normalized Task Forward final lane CLS features, before fixed feature fusion and shared classifier
center=randomly initialized shared expert_a/expert_b and LayerNorm fixed from start; rank32; 3 parameter-matrix experts
task_parameters=zero-initialized rank32->rank32 projection (1056 params/task) and image-only Router hidden16; old task modules frozen; uniform control freezes Router
bound=differentiable per-token residual/token norm ratio <0.02; fixed scale1.0 before bound; current-class BCE; reliable-view auxiliary BCE weight0.1; baseline fused/view logit MSE weight0.1
fusion=reliable Face [0.64,0.16,0.20], otherwise [0.80,0.20,0]; fixed weights
diagnostics=base scores before calibration; immutable base/center/old-route hashes; initial output difference; train feature caches; gate mean/std/entropy; residual ratios; compact checkpoints; fused/view validation scores
output=${RESULT_ROOT}/${BATCH_ID}; logs=${LOG_ROOT}/${BATCH_ID}
EOF_MANIFEST
pids=()
for method in "${METHODS[@]}"; do
  (
    date -Is > "${CONTROL}/status/${method}.started.txt"
    METHOD="${method}" GPU="${GPU}" RUN_ID="${BATCH_ID}" PYTHON="${PYTHON}" \
      SEED=0 MAX_TASKS=3 EPOCHS=30 DATA_ROOT="${DATA_ROOT}" CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
      FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" OUTPUT_ROOT="${RESULT_ROOT}" LOG_DIR="${LOG_ROOT}" \
      EXPORT_VIEW_SCORES=1 SAVE_COMPACT=1 \
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
(( failed == 0 )) || { echo "Staged validation failed; inspect logs" >&2; exit 1; }
date -Is > "${CONTROL}/status/complete.txt"
echo "POST_TASK_STAGED_VALIDATION_COMPLETE batch=${BATCH_ID}"
