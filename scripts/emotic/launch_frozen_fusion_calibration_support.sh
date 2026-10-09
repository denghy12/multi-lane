#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
[[ "${GPU:-0}" == 0 ]] || { echo "GPU0 only" >&2; exit 2; }
PYTHON="${PYTHON:-python}"
SOURCE="${SOURCE:?Completed heldout B0 source required}"
DATA_ROOT="${DATA_ROOT:?Existing EMOTIC path required}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:?Existing CLIP path required}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:?Existing audited Face manifest required}"
BATCH_ID="${BATCH_ID:-calibration_support_seed0_val_$(date +%Y%m%d_%H%M%S)}"
TASKS="${TASKS:-3}"
EPOCHS="${EPOCHS:-30}"
OUTPUT="./output/emotic_frozen_fusion_calibration_support/${BATCH_ID}"
LOG_DIR="./logs/emotic_frozen_fusion_calibration_support/${BATCH_ID}"
[[ ! -e "${OUTPUT}" && ! -e "${LOG_DIR}" ]] || { echo "Batch exists" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean code required" >&2; exit 2; }
free="$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
(( free >= 5000 )) || { echo "GPU0 free memory below 5GiB" >&2; exit 2; }
mkdir -p "${LOG_DIR}"
cat > "${LOG_DIR}/manifest.txt" <<EOF_MANIFEST
branch=$(git branch --show-current); commit=$(git rev-parse HEAD)
source=${SOURCE}; source model NOT retrained; seed0; frozen CLIP/Selectors/Prompt/classifier
GPU0 only; tasks0-$((TASKS-1)); validation-only; test forbidden
inputs=${DATA_ROOT}; ${CLIP_CHECKPOINT}; ${FACE_MANIFEST_ROOT}
change=excluded TRAIN calibration expands current-positive pool to all seen-class image groups; historical train images revisited; this is calibration memory, NOT replay-free
cache=keep all previous calibration rows exactly; export ONLY added rows with source task checkpoints; no future classes used for fit
training=${EPOCHS} epochs; original samples/epoch 1031/817/160; original updates/epoch 5/4/1; batch256; Adam0.001; cosine; FP32 no TF32; clip1
controls=fixed; static BCE; dynamic BCE; dynamic ranking; same gate hidden8 zero output, Person/Face bounded deltas0.05, invalid Face blocked, MSE0.1, delta penalty1.0
output=${OUTPUT}; logs=${LOG_DIR}; source/new support, fitted gates, score dumps, metrics, pair changes, fixed old sample drift, immutable model/source hashes
EOF_MANIFEST
finish() { code=$?; printf '%s\n' "${code}" > "${LOG_DIR}/exit_code"; }
trap finish EXIT
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m multi_lane.track_a.frozen_fusion_calibration_support \
 --source "${SOURCE}" --output "${OUTPUT}" --data-root "${DATA_ROOT}" \
 --clip-checkpoint "${CLIP_CHECKPOINT}" --face-manifest-root "${FACE_MANIFEST_ROOT}" \
 --tasks "${TASKS}" --epochs "${EPOCHS}" 2>&1 | tee "${LOG_DIR}/run.log"
