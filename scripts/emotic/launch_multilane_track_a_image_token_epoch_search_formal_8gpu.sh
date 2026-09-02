#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPUS_TEXT="${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_LIST <<< "${GPUS_TEXT}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-18000}"
GPU_READY_CHECKS="${GPU_READY_CHECKS:-2}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-30}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"

[[ "${#GPU_LIST[@]}" -eq 8 ]] || { echo "Epoch search requires exactly eight GPUs" >&2; exit 2; }
[[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l | tr -d ' ')" -eq 8 ]] || {
  echo "Epoch search requires eight distinct GPUs" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || { echo "Epoch search requires a clean Git worktree" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }

BATCH_ID="${BATCH_ID:-image_token_asl_layer1_epoch_search_formal_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_image_token_epoch_search_formal_v0.1/${BATCH_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_image_token_epoch_search_formal}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_image_token_epoch_search_formal/${BATCH_ID}"
STATUS_DIR="${CONTROL_DIR}/launcher_status"
MANIFEST="${CONTROL_DIR}/search_manifest.tsv"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}" "${STATUS_DIR}"

epochs=(18 22 26 30 34 38 42 48)
printf 'index\tlabel\tepochs_per_task\tgpu\treporting\n' > "${MANIFEST}"
for index in "${!epochs[@]}"; do
  printf '%s\tepochs%s\t%s\t%s\ttest\n' \
    "${index}" "${epochs[$index]}" "${epochs[$index]}" "${GPU_LIST[$index]}" >> "${MANIFEST}"
done

wait_for_all_gpus() {
  local consecutive=0
  while (( consecutive < GPU_READY_CHECKS )); do
    local ready=1
    for gpu in "${GPU_LIST[@]}"; do
      free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
      if (( free_mib < GPU_MIN_FREE_MIB )); then ready=0; fi
      echo "GPU ${gpu}: free_mib=${free_mib} required=${GPU_MIN_FREE_MIB}"
    done
    if (( ready == 1 )); then
      consecutive=$((consecutive + 1))
      echo "All GPUs ready ${consecutive}/${GPU_READY_CHECKS}"
    else
      consecutive=0
      echo "Waiting for all GPUs before atomic 8-run launch"
    fi
    if (( consecutive < GPU_READY_CHECKS )); then sleep "${GPU_WAIT_SECONDS}"; fi
  done
}

wait_for_all_gpus

run_one() {
  local index="$1"
  local epoch_count="${epochs[$index]}"
  local label="epochs${epoch_count}"
  local run_id="${BATCH_ID}_${label}"
  if GPU="${GPU_LIST[$index]}" RUN_ID="${run_id}" OUTPUT_BASE="${OUTPUT_BASE}" \
    EPOCHS="${epoch_count}" PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" \
    CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" LOG_DIR="${LOG_DIR}" \
    bash scripts/emotic/run_multilane_track_a_image_token_epoch_search_formal.sh; then
    printf '0\n' > "${STATUS_DIR}/${label}.exit_code"
  else
    printf '1\n' > "${STATUS_DIR}/${label}.exit_code"
    return 1
  fi
}

pids=()
for index in "${!epochs[@]}"; do
  run_one "${index}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if (( failed != 0 )); then
  echo "At least one epoch-search run failed; completed results were preserved" >&2
  exit 1
fi

"${PYTHON}" -m multi_lane.track_a.summarize_image_token_epoch_search \
  --batch-root "${OUTPUT_BASE}" \
  --batch-id "${BATCH_ID}" \
  --output-dir "${CONTROL_DIR}"
printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "MULTI_LANE_IMAGE_TOKEN_EPOCH_SEARCH_COMPLETE batch_id=${BATCH_ID} output=${OUTPUT_BASE} manifest=${MANIFEST}"
