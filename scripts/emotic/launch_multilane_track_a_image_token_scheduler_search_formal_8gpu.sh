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

[[ "${#GPU_LIST[@]}" -eq 8 ]] || { echo "Scheduler search requires exactly eight GPUs" >&2; exit 2; }
[[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l | tr -d ' ')" -eq 8 ]] || {
  echo "Scheduler search requires eight distinct GPUs" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || { echo "Scheduler search requires a clean Git worktree" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations: ${DATA_ROOT}" >&2; exit 2; }

BATCH_ID="${BATCH_ID:-image_token_asl_layer1_scheduler_search_formal_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_image_token_scheduler_search_formal_v0.1/${BATCH_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/emotic_track_a_image_token_scheduler_search_formal}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_image_token_scheduler_search_formal/${BATCH_ID}"
STATUS_DIR="${CONTROL_DIR}/launcher_status"
MANIFEST="${CONTROL_DIR}/search_manifest.tsv"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}" "${STATUS_DIR}"

labels=(cosine_anchor cosine_min001 cosine_min010 cosine_warmup005 cosine_warmup010 linear constant multistep)
modes=(cosine cosine cosine cosine cosine linear constant multistep)
min_ratios=(0 0.01 0.10 0 0 0 0 0)
warmup_ratios=(0 0 0 0.05 0.10 0 0 0)

for values_name in modes min_ratios warmup_ratios; do
  declare -n values_ref="${values_name}"
  [[ "${#values_ref[@]}" -eq "${#labels[@]}" ]] || {
    echo "Internal scheduler array length mismatch: ${values_name}" >&2
    exit 2
  }
done

printf 'index\tlabel\tscheduler_mode\tmin_lr_ratio\twarmup_ratio\tmilestones\tgamma\tgpu\treporting\n' > "${MANIFEST}"
for index in "${!labels[@]}"; do
  printf '%s\t%s\t%s\t%s\t%s\t0.6,0.85\t0.1\t%s\ttest\n' \
    "${index}" "${labels[$index]}" "${modes[$index]}" "${min_ratios[$index]}" \
    "${warmup_ratios[$index]}" "${GPU_LIST[$index]}" >> "${MANIFEST}"
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
  local label="${labels[$index]}"
  local run_id="${BATCH_ID}_${label}"
  if GPU="${GPU_LIST[$index]}" RUN_ID="${run_id}" OUTPUT_BASE="${OUTPUT_BASE}" \
    SCHEDULER_MODE="${modes[$index]}" MIN_LR_RATIO="${min_ratios[$index]}" \
    WARMUP_RATIO="${warmup_ratios[$index]}" PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" \
    CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" LOG_DIR="${LOG_DIR}" \
    bash scripts/emotic/run_multilane_track_a_image_token_scheduler_search_formal.sh; then
    printf '0\n' > "${STATUS_DIR}/${label}.exit_code"
  else
    printf '1\n' > "${STATUS_DIR}/${label}.exit_code"
    return 1
  fi
}

pids=()
for index in "${!labels[@]}"; do
  run_one "${index}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if (( failed != 0 )); then
  echo "At least one scheduler-search run failed; completed results were preserved" >&2
  exit 1
fi

"${PYTHON}" -m multi_lane.track_a.summarize_image_token_scheduler_search \
  --batch-root "${OUTPUT_BASE}" \
  --batch-id "${BATCH_ID}" \
  --output-dir "${CONTROL_DIR}"
printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "MULTI_LANE_IMAGE_TOKEN_SCHEDULER_SEARCH_COMPLETE batch_id=${BATCH_ID} output=${OUTPUT_BASE} manifest=${MANIFEST}"
