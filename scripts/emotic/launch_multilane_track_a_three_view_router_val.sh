#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
read -r -a GPU_LIST <<< "${GPUS:-0 1}"
[[ ${#GPU_LIST[@]} -eq 2 ]] || { echo "Exactly two GPUs are required" >&2; exit 2; }
[[ "${GPU_LIST[0]}" != "${GPU_LIST[1]}" ]] || { echo "GPUs must be distinct" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Launcher requires a clean worktree" >&2; exit 2; }

BATCH_ID="${BATCH_ID:-three_view_router_seed0_$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_three_view_router_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_three_view_router/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_three_view_router/${BATCH_ID}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-/mnt/haoyuan/workspace/multi-lane-main-face-manifest/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
OLD_BASE="/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_learned_reliability_gate_v0.1/learned_reliability_gate_seed012_20260904_021636/validation_sources"
FULL_RUN="${FULL_RUN:-${OLD_BASE}/learned_reliability_gate_seed012_20260904_021636_seed0_full}"
PERSON_RUN="${PERSON_RUN:-${OLD_BASE}/learned_reliability_gate_seed012_20260904_021636_seed0_person}"
FACE_RUN_ID="${BATCH_ID}_seed0_face"
FACE_RUN="${RESULT_BASE}/validation_sources/${FACE_RUN_ID}"
DESCRIPTOR_ROOT="${RESULT_BASE}/descriptors"
SELECTION_DIR="${CONTROL_DIR}/router_selection"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"

[[ ! -e "${RESULT_BASE}" ]] || { echo "Result batch exists: ${RESULT_BASE}" >&2; exit 2; }
[[ ! -e "${CONTROL_DIR}" ]] || { echo "Control batch exists: ${CONTROL_DIR}" >&2; exit 2; }
for run in "${FULL_RUN}" "${PERSON_RUN}"; do
  [[ -f "${run}/seed_summary.json" ]] || { echo "Missing reusable source: ${run}" >&2; exit 2; }
done
mkdir -p "${RESULT_BASE}/validation_sources" "${CONTROL_DIR}/status" "${LOG_DIR}"

cat > "${CONTROL_DIR}/protocol.txt" <<EOF
Selection only: seed0, EMOTIC, 8-task validation; test access forbidden.
Sources: stable image-group 90% fit / disjoint 10% calibration.
Full/Person: reuse audited 20260904 scores and compact checkpoints.
Face: same split, valid&&!ambiguous loss mask, 30 epochs/task, layer1 b32 Adapter-ASL.
R0: Full0.8+Person0.2. R1: reliable Face beta0.20.
R2: quality and three-view prediction statistics. R3: R2 plus three frozen-CLIP view cosines.
Routers: task-local, hidden16, masked softmax, invalid Face weight0, AdamW1e-3/wd1e-4,
80 epochs/task, current-task calibration labels only, priors 0/0.1/1.
Advance only when best R3 final validation mAP exceeds both R1 and best R2.
EOF
printf 'batch_id\tface_gpu\tdescriptor_gpu\tfull_run\tperson_run\tface_run\tdescriptors\ttest_access\n' > "${CONTROL_DIR}/manifest.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tforbidden\n' \
  "${BATCH_ID}" "${GPU_LIST[0]}" "${GPU_LIST[1]}" "${FULL_RUN}" "${PERSON_RUN}" \
  "${FACE_RUN}" "${DESCRIPTOR_ROOT}" >> "${CONTROL_DIR}/manifest.tsv"

consecutive=0
while (( consecutive < 2 )); do
  ready=1
  for gpu in "${GPU_LIST[@]}"; do
    free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    echo "GPU ${gpu} free_mib=${free_mib} utilization=${utilization}"
    if (( free_mib < MIN_FREE_MIB || utilization > MAX_UTILIZATION )); then ready=0; fi
  done
  if (( ready )); then consecutive=$((consecutive + 1)); else consecutive=0; fi
  if (( consecutive < 2 )); then sleep 30; fi
done

SEED=0 VIEW=face GPU="${GPU_LIST[0]}" RUN_ID="${FACE_RUN_ID}" \
  OUTPUT_BASE="${RESULT_BASE}/validation_sources" LOG_DIR="${LOG_DIR}" \
  PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
  FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" \
  bash scripts/emotic/run_multilane_track_a_reliability_source_val.sh &
face_pid=$!
GPU="${GPU_LIST[1]}" OUTPUT_DIR="${DESCRIPTOR_ROOT}" \
  LOG_PATH="${LOG_DIR}/descriptor_export.log" PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" \
  CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" \
  bash scripts/emotic/run_multilane_track_a_three_view_descriptors.sh &
descriptor_pid=$!
if wait "${face_pid}"; then face_rc=0; else face_rc=$?; fi
if wait "${descriptor_pid}"; then descriptor_rc=0; else descriptor_rc=$?; fi
printf '%s\n' "${face_rc}" > "${CONTROL_DIR}/status/face_source.exit_code"
printf '%s\n' "${descriptor_rc}" > "${CONTROL_DIR}/status/descriptors.exit_code"
if (( face_rc || descriptor_rc )); then
  echo "Face source or descriptor export failed; router not started" >&2
  exit 1
fi

"${PYTHON}" -m multi_lane.track_a.three_view_router \
  --full-run "${FULL_RUN}" \
  --person-run "${PERSON_RUN}" \
  --face-run "${FACE_RUN}" \
  --data-root "${DATA_ROOT}" \
  --face-manifest-root "${FACE_MANIFEST_ROOT}" \
  --descriptor-root "${DESCRIPTOR_ROOT}" \
  --output-dir "${SELECTION_DIR}" \
  2>&1 | tee "${LOG_DIR}/router_selection.log"
printf '0\n' > "${CONTROL_DIR}/status/router.exit_code"
printf 'validation_complete_test_not_accessed\n' > "${CONTROL_DIR}/batch_status.txt"
echo "THREE_VIEW_ROUTER_BATCH_COMPLETE batch=${BATCH_ID} selection=${SELECTION_DIR}/validation_selection.json test_accessed=false"
