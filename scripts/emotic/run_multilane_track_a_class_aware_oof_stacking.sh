#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
GPU="${GPU:-0}"
SOURCE_BATCH="${SOURCE_BATCH:-three_view_oof_seed0_20260910_160250}"
OOF_BASE="${OOF_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_three_view_oof_v0.1/${SOURCE_BATCH}}"
OOF_ROOT="${OOF_ROOT:-${OOF_BASE}/oof_sources}"
DESCRIPTOR_ROOT="${DESCRIPTOR_ROOT:-${OOF_BASE}/descriptors}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
FULL_RUN="${FULL_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_full_anchor}"
PERSON_RUN="${PERSON_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_person_letterbox}"
FACE_RUN="${FACE_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_endpoint_val_v0.1/face_endpoint_val_seed0_20260909_1320/face_endpoint_val_seed0_20260909_1320_face}"
RUN_ID="${RUN_ID:-class_aware_oof_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/output/emotic_track_a_class_aware_oof/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_class_aware_oof}"

[[ -x "${PYTHON}" ]] || { echo "Missing restored ddp Python: ${PYTHON}" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Stacking run requires a clean worktree" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Output exists: ${OUTPUT_DIR}" >&2; exit 2; }
for path in "${OOF_ROOT}" "${DESCRIPTOR_ROOT}" "${FACE_MANIFEST_ROOT}" \
  "${FULL_RUN}" "${PERSON_RUN}" "${FACE_RUN}"; do
  [[ -e "${path}" ]] || { echo "Missing source: ${path}" >&2; exit 2; }
done

free_mib="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
[[ "${free_mib}" -ge 2000 ]] || { echo "GPU ${GPU} has only ${free_mib} MiB free" >&2; exit 2; }
mkdir -p "${LOG_DIR}"
echo "Class-aware OOF stacking: C0 fixed R1; C1 class bias priors 1/3/10; C2 rank2 priors 1/3/10; seed0 complete 8-task validation; margin 0.05 mAP; test forbidden; GPU=${GPU}"
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="${GPU}" \
  "${PYTHON}" -m multi_lane.track_a.class_aware_oof_stacking \
  --oof-root "${OOF_ROOT}" --full-run "${FULL_RUN}" --person-run "${PERSON_RUN}" \
  --face-run "${FACE_RUN}" --data-root "${DATA_ROOT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" --descriptor-root "${DESCRIPTOR_ROOT}" \
  --output-dir "${OUTPUT_DIR}" --device cuda:0 \
  2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
echo "CLASS_AWARE_OOF_STACKING_RUN_COMPLETE output=${OUTPUT_DIR}"
