#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:-face_manifest_audit_v1_20260908}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
MANIFEST="${MANIFEST:-${ROOT}/output/emotic_face_manifest/${RUN_ID}/manifests/train.jsonl}"
OUTPUT="${OUTPUT:-${ROOT}/output/emotic_face_manifest/${RUN_ID}/face_gpu_smoke.json}"
LOG="${LOG:-${ROOT}/logs/emotic_face_manifest/${RUN_ID}_gpu_smoke.log}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
READY_CHECKS="${READY_CHECKS:-2}"

[[ -f "${MANIFEST}" ]] || { echo "Missing manifest: ${MANIFEST}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${LOG}")"

ready_gpu=""
consecutive=0
last_candidate=""
while [[ -z "${ready_gpu}" ]]; do
  while IFS=', ' read -r gpu free_mib utilization; do
    if [[ "${free_mib}" -ge "${MIN_FREE_MIB}" && "${utilization}" -le "${MAX_UTILIZATION}" ]]; then
      candidate="${gpu}"
      break
    fi
    candidate=""
  done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
  if [[ -n "${candidate:-}" ]]; then
    if [[ "${candidate}" == "${last_candidate}" ]]; then
      consecutive=$((consecutive + 1))
    else
      consecutive=1
      last_candidate="${candidate}"
    fi
    if [[ "${consecutive}" -ge "${READY_CHECKS}" ]]; then
      ready_gpu="${candidate}"
      break
    fi
  else
    consecutive=0
    last_candidate=""
  fi
  echo "Waiting for Face GPU smoke: candidate=${candidate:-none} consecutive=${consecutive}/${READY_CHECKS}" | tee -a "${LOG}"
  sleep "${WAIT_SECONDS}"
done

echo "Running Face GPU smoke on physical GPU ${ready_gpu}" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES="${ready_gpu}" "${PYTHON}" -m multi_lane.track_a.face_manifest_gpu_smoke \
  --manifest "${MANIFEST}" \
  --clip-checkpoint "${CLIP_CHECKPOINT}" \
  --output "${OUTPUT}" \
  --samples 4 \
  --device cuda:0 2>&1 | tee -a "${LOG}"
