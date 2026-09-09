#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
LOG_PATH="${LOG_PATH:?LOG_PATH is required}"

[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Descriptor output exists: ${OUTPUT_DIR}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC data: ${DATA_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${FACE_MANIFEST_ROOT}/audit_summary.json" ]] || { echo "Missing Face audit: ${FACE_MANIFEST_ROOT}" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Descriptor export requires a clean worktree" >&2; exit 2; }
mkdir -p "$(dirname "${LOG_PATH}")" "$(dirname "${OUTPUT_DIR}")"

echo "Three-view descriptors: gpu=${GPU} splits=train_calibration,val test=forbidden views=full_center_crop,person_margin15_letterbox,face_manifest_letterbox feature=three_frozen_clip_cosines batch=128"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.export_three_view_descriptors \
  --data-root "${DATA_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --batch-size 128 \
  --workers 2 \
  2>&1 | tee "${LOG_PATH}"
