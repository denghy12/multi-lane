#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:?RUN_ID is required}"
PYTHON="${PYTHON:-/opt/conda/envs/cocoer-preprocess/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
DETECTOR_CHECKPOINT="${DETECTOR_CHECKPOINT:-/mnt/haoyuan/workspace/baseline_sources/cocoer_insightface/models/buffalo_l/det_10g.onnx}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT}/output/emotic_face_manifest}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_face_manifest}"
OUTPUT_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"
VISUAL_SAMPLES="${VISUAL_SAMPLES:-200}"
MAX_IMAGES="${MAX_IMAGES:-}"

[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }
[[ -f "${DETECTOR_CHECKPOINT}" ]] || { echo "Missing face detector: ${DETECTOR_CHECKPOINT}" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}"

ARGS=(
  --data-root "${DATA_ROOT}"
  --detector-checkpoint "${DETECTOR_CHECKPOINT}"
  --output-root "${OUTPUT_ROOT}"
  --splits train val
  --det-size 640
  --det-threshold 0.5
  --face-margin 0.15
  --ambiguity-gap 0.08
  --visual-samples "${VISUAL_SAMPLES}"
  --resume
)
if [[ -n "${MAX_IMAGES}" ]]; then
  ARGS+=(--max-images "${MAX_IMAGES}")
fi

echo "Face manifest audit: run_id=${RUN_ID} dataset=EMOTIC splits=train,val detector=InsightFace-SCRFD-det_10g provider=CPU det_size=640 threshold=0.5 face_margin=0.15 ambiguity_gap=0.08 visual_samples=${VISUAL_SAMPLES} max_images=${MAX_IMAGES:-all} training=disabled"
"${PYTHON}" -m multi_lane.track_a.face_manifest "${ARGS[@]}" 2>&1 | tee -a "${LOG_PATH}"

echo "EMOTIC_FACE_MANIFEST_AUDIT_COMPLETE output=${OUTPUT_ROOT} log=${LOG_PATH}"
