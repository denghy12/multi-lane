#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BATCH_ID="${BATCH_ID:-joint26_parax_views_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-./datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-./models/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-./output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
RESULT_ROOT="${ROOT}/output/emotic_joint26_parax_view_ablation"
LOG_ROOT="${ROOT}/logs/emotic_joint26_parax_view_ablation"
CONTROL="${ROOT}/output/emotic_joint26_control/${BATCH_ID}"
METHODS=(FULL_ONLY THREE_VIEW FULL_ONLY_PARAX THREE_VIEW_PARAX)
[[ ! -e "${CONTROL}" && ! -e "${RESULT_ROOT}/${BATCH_ID}" ]] || {
  echo "Batch already exists: ${BATCH_ID}" >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Training worktree has tracked modifications" >&2; exit 2;
}
for path in "${DATA_ROOT}/CVPR17_Annotations.mat" "${CLIP_CHECKPOINT}" \
  "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" "${FACE_MANIFEST_ROOT}/manifests/val.jsonl"; do
  [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
free_mib="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
(( free_mib >= ${MIN_FREE_MIB:-20000} )) || {
  echo "Insufficient GPU${GPU} memory for four concurrent arms: ${free_mib} MiB" >&2; exit 2;
}
mkdir -p "${CONTROL}/status" "${LOG_ROOT}/${BATCH_ID}"
cat > "${CONTROL}/experiment_manifest.txt" <<EOF_MANIFEST
batch=${BATCH_ID}
branch=$(git branch --show-current)
commit=$(git rev-parse HEAD)
methods=${METHODS[*]}; four concurrent processes on GPU${GPU}
dataset=EMOTIC original train; all person instances and all26 labels simultaneously; one Task Forward lane; same validation sample IDs
inputs=${DATA_ROOT}; ${CLIP_CHECKPOINT}; ${FACE_MANIFEST_ROOT}
seed=0; epochs=30; batch=64; Adam; main_lr=0.0125; ParaX_lr=0.0004; weight_decay=0; cosine; no warmup; AMP+TF32
backbone=frozen OpenAI CLIP ViT-B/16; shared Selector/Prompt/classifier; Image-token Adapter disabled
ParaX=Frozen Forward after Block11/code10 before Block12; patch tokens only; CLS bypass; shared E_A/E_B parameter-matrix experts=3; rank32; router_hidden16; official initialization; initial learnable scale0.1; all ParaX parameters train throughout joint phase
excluded=Projector, level embedding, distillation, residual penalty/cap, dynamic fusion
fusion=reliable Face [0.64,0.16,0.20], otherwise [0.80,0.20,0]; same reliability rules
loss_three_view=BCE(fused)+0.1*mean(BCE(available views)); loss_full_only=1.1*BCE(Full)
budget=same complete training epochs and sample order across four groups; three views cost more compute; no task resets; 30 joint epochs is not 240 incremental epochs
selection=final epoch; validation only; test forbidden; no calibration holdout
audit=initial Selector/Prompt/classifier hash, first batch IDs, full train/val ID hashes, 26 label gradients in tests, frozen CLIP before/after hash, gate/residual/gradient stats, peak GPU memory, scores and compact checkpoint
outputs=${RESULT_ROOT}/${BATCH_ID}; logs=${LOG_ROOT}/${BATCH_ID}; state=${CONTROL}
EOF_MANIFEST
pids=()
for method in "${METHODS[@]}"; do
  (
    date -Is > "${CONTROL}/status/${method}.started.txt"
    set +e
    METHOD="${method}" GPU="${GPU}" RUN_ID="${BATCH_ID}" PYTHON="${PYTHON}" \
      DATA_ROOT="${DATA_ROOT}" CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
      FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" RESULT_ROOT="${RESULT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
      SEED=0 EPOCHS=30 bash scripts/emotic/run_joint26_parax_view_ablation.sh \
      > "${LOG_ROOT}/${BATCH_ID}/${method}.launcher.log" 2>&1
    result=$?
    printf '%s\n' "${result}" > "${CONTROL}/status/${method}.exit_code"
    exit "${result}"
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
(( failed == 0 )) || { echo "Joint26 run failed; inspect logs" >&2; exit 1; }
"${PYTHON}" -m multi_lane.track_a.compare_joint26 --batch-root "${RESULT_ROOT}/${BATCH_ID}"
date -Is > "${CONTROL}/status/complete.txt"
echo "JOINT26_PARAX_VIEW_ABLATION_COMPLETE batch=${BATCH_ID}"
