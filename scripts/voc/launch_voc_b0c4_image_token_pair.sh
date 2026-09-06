#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BATCH_ID="${BATCH_ID:-$(date +%Y%m%d-%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/datasets}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
MAX_UTIL="${MAX_UTIL:-10}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_CONTROL="${GPU_CONTROL:-}"
GPU_ADAPTER="${GPU_ADAPTER:-}"

cd "${ROOT_DIR}"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing to launch from a tracked-dirty worktree" >&2
  exit 3
fi

if [[ ! -d "${DATA_ROOT}/VOCdevkit/VOC2007" ]]; then
  echo "VOC2007 not found at ${DATA_ROOT}/VOCdevkit/VOC2007" >&2
  exit 4
fi

select_gpus() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | \
    awk -F, -v free="${MIN_FREE_MIB}" -v util="${MAX_UTIL}" '
      {gsub(/ /, "", $0); if ($2 >= free && $3 <= util) print $1}' | head -n 2
}

if [[ -z "${GPU_CONTROL}" || -z "${GPU_ADAPTER}" ]]; then
  echo "Waiting for two GPUs with >=${MIN_FREE_MIB} MiB free and <=${MAX_UTIL}% utilization"
  while true; do
    mapfile -t first_probe < <(select_gpus)
    if (( ${#first_probe[@]} >= 2 )); then
      sleep 10
      mapfile -t second_probe < <(select_gpus)
      if (( ${#second_probe[@]} >= 2 )) && \
         [[ " ${second_probe[*]} " == *" ${first_probe[0]} "* ]] && \
         [[ " ${second_probe[*]} " == *" ${first_probe[1]} "* ]]; then
        GPU_CONTROL="${first_probe[0]}"
        GPU_ADAPTER="${first_probe[1]}"
        break
      fi
    fi
    sleep "${POLL_SECONDS}"
  done
elif [[ "${GPU_CONTROL}" == "${GPU_ADAPTER}" ]]; then
  echo "Explicit control and adapter GPUs must be different" >&2
  exit 5
else
  echo "Using explicitly assigned GPUs ${GPU_CONTROL}/${GPU_ADAPTER}"
fi

echo "Launching control on GPU ${GPU_CONTROL}, adapter on GPU ${GPU_ADAPTER}; batch=${BATCH_ID}"
BATCH_ID="${BATCH_ID}" DATA_ROOT="${DATA_ROOT}" VARIANT=control GPU_ID="${GPU_CONTROL}" \
  bash scripts/voc/run_voc_b0c4_image_token_formal.sh &
CONTROL_PID=$!
BATCH_ID="${BATCH_ID}" DATA_ROOT="${DATA_ROOT}" VARIANT=adapter GPU_ID="${GPU_ADAPTER}" \
  bash scripts/voc/run_voc_b0c4_image_token_formal.sh &
ADAPTER_PID=$!

status=0
wait "${CONTROL_PID}" || status=1
wait "${ADAPTER_PID}" || status=1
if (( status != 0 )); then
  echo "At least one paired VOC run failed; inspect logs before retrying" >&2
  exit "${status}"
fi

python scripts/voc/summarize_voc_b0c4_image_token_pair.py \
  --batch-root "${ROOT_DIR}/output/voc_b0c4_image_token/${BATCH_ID}"
echo "VOC paired run complete: ${BATCH_ID}"
