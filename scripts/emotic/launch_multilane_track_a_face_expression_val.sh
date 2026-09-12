#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
EXPRESSION_CHECKPOINT="${EXPRESSION_CHECKPOINT:-/mnt/haoyuan/workspace/pretrained/emotiefflib/enet_b0_8_best_afew.pt}"
EXPECTED_SHA256="${EXPECTED_SHA256:?EXPECTED_SHA256 is required}"
BATCH_ID="${BATCH_ID:-face_expression_seed0_$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_expression_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_face_expression/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_face_expression/${BATCH_ID}"
FULL_RUN="${FULL_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_full_anchor}"
PERSON_RUN="${PERSON_RUN:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720/image_token_layer1_full_person_letterbox_val_seed0_20260903_134720_person_letterbox}"
ANCHOR_FACE="${ANCHOR_FACE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_face_endpoint_val_v0.1/face_endpoint_val_seed0_20260909_1320/face_endpoint_val_seed0_20260909_1320_face}"
GPUS=( ${GPUS:-0 1} )

[[ "${#GPUS[@]}" -eq 2 ]] || { echo "Exactly two GPUs are required" >&2; exit 2; }
[[ ! -e "${RESULT_BASE}" && ! -e "${CONTROL_DIR}" ]] || { echo "Batch output exists" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean worktree required" >&2; exit 2; }
for path in "${FULL_RUN}" "${PERSON_RUN}" "${ANCHOR_FACE}"; do
  [[ -f "${path}/seed_summary.json" ]] || { echo "Missing source run: ${path}" >&2; exit 2; }
done
for gpu in "${GPUS[@]}"; do
  free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  [[ "${free_mib}" -ge 8000 ]] || { echo "GPU ${gpu} has only ${free_mib} MiB free" >&2; exit 2; }
done

mkdir -p "${RESULT_BASE}" "${CONTROL_DIR}/status" "${LOG_DIR}"
"${PYTHON}" -m multi_lane.track_a.audit_face_alignment \
  --data-root "${DATA_ROOT}" --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --output "${CONTROL_DIR}/alignment_audit.json" \
  2>&1 | tee "${LOG_DIR}/alignment_audit.log"

printf 'variant\tgpu\tseed\trun\n' > "${CONTROL_DIR}/manifest.tsv"
pids=()
variants=(projection bottleneck_adapter)
for index in 0 1; do
  variant="${variants[$index]}"
  gpu="${GPUS[$index]}"
  run_id="${BATCH_ID}_${variant}"
  printf '%s\t%s\t0\t%s/%s\n' "${variant}" "${gpu}" "${RESULT_BASE}" "${run_id}" >> "${CONTROL_DIR}/manifest.tsv"
  (
    set +e
    GPU="${gpu}" RUN_ID="${run_id}" VARIANT="${variant}" OUTPUT_BASE="${RESULT_BASE}" \
      EXPECTED_SHA256="${EXPECTED_SHA256}" PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" \
      FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" EXPRESSION_CHECKPOINT="${EXPRESSION_CHECKPOINT}" \
      LOG_DIR="${LOG_DIR}" bash scripts/emotic/run_multilane_track_a_face_expression_val.sh
    code=$?
    printf '%s\n' "${code}" > "${CONTROL_DIR}/status/${variant}.exit_code"
    exit "${code}"
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
[[ "${failed}" -eq 0 ]] || { printf 'failed\n' > "${CONTROL_DIR}/batch_status.txt"; exit 1; }

"${PYTHON}" -m multi_lane.track_a.compare_face_expression_validation \
  --full-run "${FULL_RUN}" --person-run "${PERSON_RUN}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" --anchor "clip_face=${ANCHOR_FACE}" \
  --candidate "expression_projection=${RESULT_BASE}/${BATCH_ID}_projection" \
  --candidate "expression_adapter=${RESULT_BASE}/${BATCH_ID}_bottleneck_adapter" \
  --output "${CONTROL_DIR}/validation_summary.json" 2>&1 | tee "${LOG_DIR}/summary.log"
printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "FACE_EXPRESSION_BATCH_COMPLETE batch=${BATCH_ID} result=${RESULT_BASE}"
