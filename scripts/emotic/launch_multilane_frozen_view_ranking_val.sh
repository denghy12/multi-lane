#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
GPU="${GPU:-0}"
[[ "${GPU}" == 0 ]] || { echo "This experiment is authorized for GPU0 only" >&2; exit 2; }
PYTHON="${PYTHON:-python}"
BATCH_ID="${BATCH_ID:-frozen_view_ranking_seed0_val_$(date +%Y%m%d_%H%M%S)}"
MAX_TASKS="${MAX_TASKS:-3}"
EPOCHS="${EPOCHS:-30}"
FUSION_EPOCHS="${FUSION_EPOCHS:-30}"
RESULT_ROOT="./output/emotic_frozen_view_ranking_val"
LOG_ROOT="./logs/emotic_frozen_view_ranking_val"
CONTROL="./output/emotic_frozen_view_ranking_control/${BATCH_ID}"
[[ ! -e "${CONTROL}" && ! -e "${RESULT_ROOT}/${BATCH_ID}" ]] || { echo "Batch exists" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Tracked source must be clean" >&2; exit 2; }
free_mib="$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
(( free_mib >= ${MIN_FREE_MIB:-5000} )) || { echo "Insufficient GPU0 free memory" >&2; exit 2; }
mkdir -p "${CONTROL}/status" "${LOG_ROOT}/${BATCH_ID}"
cat > "${CONTROL}/experiment_manifest.txt" <<EOF_MANIFEST
branch=$(git branch --show-current); code_commit=$(git rev-parse HEAD)
batch=${BATCH_ID}; GPU0 only; EMOTIC Track-A; seed0; tasks0-$((MAX_TASKS-1)); validation only; test forbidden
source=one B0 model; shared frozen CLIP ViT-B/16; no ParaX, Image-token Adapter or Projector
partition=stable SHA256 image-group train split: 80% base training, 20% excluded from ALL base tasks for fusion fitting
base=${EPOCHS} epochs/task; batch64; Adam main_lr0.0125; cosine; weight_decay0; AMP+TF32
inputs=${DATA_ROOT:-./datasets/EMOTIC}; ${CLIP_CHECKPOINT:-./models/clip/ViT-B-16.pt}; ${FACE_MANIFEST_ROOT:-./output/emotic_face_manifest/face_manifest_audit_v1_20260908}
controls=fixed source fusion; static BCE (2 params/task); dynamic BCE; dynamic class-balanced positive-negative ranking
fusion=${FUSION_EPOCHS} epochs/task; cached held-out TRAIN logits only; FP32 no TF32; batch256; Adam lr0.001; cosine; clip1.0; no validation selection
router=class-shared, task-owned frozen after fitting; inputs3 standardized view logits+3 disagreements+Face reliability; hidden8 tanh; output2 zero-init; 82 params/task
weights=Person/Face delta each <=0.05 via tanh; Full delta=-Person-Face; unreliable Face delta0 and weight0
objective=method objective +0.1 baseline logit MSE +1.0 mean squared weight delta; ranking temperature from train fixed-logit std clamp_min1; no BCE fallback when no pairs
inference=saved fixed fused logits + deltaP*(Person-Full) + deltaF*(Face-Full); no feature changes
outputs=${RESULT_ROOT}/${BATCH_ID}; logs=${LOG_ROOT}/${BATCH_ID}; frozen gate states, score dumps, metrics, pair changes, old sample drift, immutable source hash
EOF_MANIFEST
finish() { code=$?; printf '%s\n' "${code}" > "${CONTROL}/status/exit_code"; if (( code != 0 )); then date -Is > "${CONTROL}/status/failed.txt"; fi; }
trap finish EXIT
date -Is > "${CONTROL}/status/started.txt"
METHOD=B0 GPU=0 RUN_ID="${BATCH_ID}" PYTHON="${PYTHON}" \
  MAX_TASKS="${MAX_TASKS}" EPOCHS="${EPOCHS}" SEED=0 \
  OUTPUT_ROOT="${RESULT_ROOT}" LOG_DIR="${LOG_ROOT}" \
  EXPORT_VIEW_SCORES=1 SAVE_COMPACT=1 EXPORT_HELDOUT_VIEWS=1 \
  bash scripts/emotic/run_multilane_track_a_post_task_router_val.sh
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m multi_lane.track_a.frozen_view_ranking_fusion \
  --source "${RESULT_ROOT}/${BATCH_ID}/B0" --output "${RESULT_ROOT}/${BATCH_ID}/fusion" \
  --tasks "${MAX_TASKS}" --epochs "${FUSION_EPOCHS}" --batch-size 256 --device cuda:0 \
  2>&1 | tee "${LOG_ROOT}/${BATCH_ID}/fusion.log"
date -Is > "${CONTROL}/status/complete.txt"
echo "FROZEN_VIEW_RANKING_VALIDATION_COMPLETE batch=${BATCH_ID}"
