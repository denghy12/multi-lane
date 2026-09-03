#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
SOURCE_RUN="${SOURCE_RUN:?SOURCE_RUN is required}"
SELECTION="${SELECTION:?SELECTION is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_learned_reliability_gate}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Output already exists: ${OUTPUT_ROOT}" >&2; exit 2; }
[[ -f "${SELECTION}" ]] || { echo "Missing validation selection: ${SELECTION}" >&2; exit 2; }
[[ -f "${SOURCE_RUN}/seed_summary.json" ]] || { echo "Missing source run: ${SOURCE_RUN}" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Test export requires a clean Git worktree" >&2; exit 2; }
mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_ROOT}")"

echo "Locked compact-state test export: gpu=${GPU} source=${SOURCE_RUN} selection=${SELECTION} output=${OUTPUT_ROOT}; optimization=off search=off"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.export_compact_test_scores \
  --source-run "${SOURCE_RUN}" \
  --validation-selection "${SELECTION}" \
  --data-root "${DATA_ROOT}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda \
  2>&1 | tee "${LOG_PATH}"
