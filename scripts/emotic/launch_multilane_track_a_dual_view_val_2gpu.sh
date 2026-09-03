#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPUS_TEXT="${GPUS:-0 1}"
read -r -a GPU_LIST <<< "${GPUS_TEXT}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-18000}"
GPU_READY_CHECKS="${GPU_READY_CHECKS:-2}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-30}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"

[[ "${#GPU_LIST[@]}" -eq 2 ]] || { echo "Dual-view validation requires two GPUs" >&2; exit 2; }
[[ "${GPU_LIST[0]}" != "${GPU_LIST[1]}" ]] || { echo "GPUs must be distinct" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Dual-view launch requires a clean Git worktree" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }

BATCH_ID="${BATCH_ID:-image_token_layer1_full_person_letterbox_val_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_val_v0.1/${BATCH_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_dual_view_val}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_dual_view_val/${BATCH_ID}"
STATUS_DIR="${CONTROL_DIR}/launcher_status"
MANIFEST="${CONTROL_DIR}/manifest.tsv"
FULL_RUN_ID="${BATCH_ID}_full_anchor"
PERSON_RUN_ID="${BATCH_ID}_person_letterbox"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}" "${STATUS_DIR}"

printf 'view\tgpu\trun_id\tinput\tmargin\ttransform\tjitter\treporting\n' > "${MANIFEST}"
printf 'full\t%s\t%s\tfull\t0\tlegacy_crop\t0@0\tval\n' "${GPU_LIST[0]}" "${FULL_RUN_ID}" >> "${MANIFEST}"
printf 'person\t%s\t%s\tperson_crop\t0.15\tletterbox\t0.10@0.20\tval\n' "${GPU_LIST[1]}" "${PERSON_RUN_ID}" >> "${MANIFEST}"

consecutive=0
while (( consecutive < GPU_READY_CHECKS )); do
  ready=1
  for gpu in "${GPU_LIST[@]}"; do
    free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -dc '0-9')"
    echo "GPU ${gpu}: free_mib=${free_mib} utilization=${utilization}%"
    if (( free_mib < GPU_MIN_FREE_MIB || utilization > 10 )); then ready=0; fi
  done
  if (( ready == 1 )); then
    consecutive=$((consecutive + 1))
    echo "Both GPUs ready ${consecutive}/${GPU_READY_CHECKS}"
  else
    consecutive=0
  fi
  if (( consecutive < GPU_READY_CHECKS )); then sleep "${GPU_WAIT_SECONDS}"; fi
done

run_view() {
  local view="$1"
  local gpu="$2"
  local run_id="$3"
  if GPU="${gpu}" RUN_ID="${run_id}" OUTPUT_BASE="${OUTPUT_BASE}" VIEW="${view}" \
    PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
    LOG_DIR="${LOG_DIR}" bash scripts/emotic/run_multilane_track_a_dual_view_val.sh; then
    printf '0\n' > "${STATUS_DIR}/${view}.exit_code"
  else
    printf '1\n' > "${STATUS_DIR}/${view}.exit_code"
    return 1
  fi
}

run_view full "${GPU_LIST[0]}" "${FULL_RUN_ID}" &
full_pid="$!"
run_view person "${GPU_LIST[1]}" "${PERSON_RUN_ID}" &
person_pid="$!"
failed=0
if ! wait "${full_pid}"; then failed=1; fi
if ! wait "${person_pid}"; then failed=1; fi
if (( failed != 0 )); then
  echo "At least one dual-view validation run failed; completed outputs are preserved" >&2
  exit 1
fi

"${PYTHON}" -m multi_lane.track_a.fuse_validation_scores \
  --full-run "${OUTPUT_BASE}/${FULL_RUN_ID}" \
  --person-run "${OUTPUT_BASE}/${PERSON_RUN_ID}" \
  --output "${CONTROL_DIR}/fusion_summary.json" \
  --alpha-step 0.05
printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "MULTI_LANE_DUAL_VIEW_VALIDATION_COMPLETE batch_id=${BATCH_ID} output=${OUTPUT_BASE} fusion=${CONTROL_DIR}/fusion_summary.json"
