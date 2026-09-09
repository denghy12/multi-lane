#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:?RUN_ID is required}"
PYTHON="${PYTHON:-/opt/conda/envs/cocoer-preprocess/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
DETECTOR_CHECKPOINT="${DETECTOR_CHECKPOINT:-/mnt/haoyuan/workspace/baseline_sources/cocoer_insightface/models/buffalo_l/det_10g.onnx}"
SOURCE_MANIFEST_ROOT="${SOURCE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_face_test_manifest}"
OUTPUT_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Manifest destination exists: ${OUTPUT_ROOT}" >&2; exit 2; }
[[ -f "${SOURCE_MANIFEST_ROOT}/audit_summary.json" ]] || { echo "Missing source audit" >&2; exit 2; }
[[ -f "${DETECTOR_CHECKPOINT}" ]] || { echo "Missing detector checkpoint" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Manifest preparation requires a clean worktree" >&2; exit 2; }

mkdir -p "${OUTPUT_ROOT}/detections" "${OUTPUT_ROOT}/manifests" \
  "${OUTPUT_ROOT}/summaries" "${LOG_DIR}"
cp "${SOURCE_MANIFEST_ROOT}/detector_config.json" "${OUTPUT_ROOT}/detector_config.json"
for split in train val; do
  cp "${SOURCE_MANIFEST_ROOT}/detections/${split}.jsonl" "${OUTPUT_ROOT}/detections/${split}.jsonl"
  cp "${SOURCE_MANIFEST_ROOT}/manifests/${split}.jsonl" "${OUTPUT_ROOT}/manifests/${split}.jsonl"
  cp "${SOURCE_MANIFEST_ROOT}/summaries/${split}.json" "${OUTPUT_ROOT}/summaries/${split}.json"
done

echo "Face test manifest: reuse audited train/val detector cache; detect test only; SCRFD det_10g CPU; det_size=640 threshold=0.5 margin=0.15 ambiguity_gap=0.08"
"${PYTHON}" -m multi_lane.track_a.face_manifest \
  --data-root "${DATA_ROOT}" \
  --detector-checkpoint "${DETECTOR_CHECKPOINT}" \
  --output-root "${OUTPUT_ROOT}" \
  --splits train val test \
  --det-size 640 \
  --det-threshold 0.5 \
  --face-margin 0.15 \
  --ambiguity-gap 0.08 \
  --visual-samples 0 \
  --resume 2>&1 | tee "${LOG_PATH}"

echo "FACE_TEST_MANIFEST_COMPLETE output=${OUTPUT_ROOT} log=${LOG_PATH}"
