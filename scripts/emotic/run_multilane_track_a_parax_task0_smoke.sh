#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
GPU="${GPU:?GPU must be set}"
METHOD="${METHOD:?METHOD must be set}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
LOG_PATH="${LOG_PATH:?LOG_PATH must be set}"
mkdir -p "$(dirname "${LOG_PATH}")"
case "${METHOD}" in
  B0) PARAX_MODE=disabled; PARAX_LAYERS=(10) ;;
  P-post) PARAX_MODE=post; PARAX_LAYERS=(10) ;;
  P-10) PARAX_MODE=image; PARAX_LAYERS=(10) ;;
  P-8:10) PARAX_MODE=image; PARAX_LAYERS=(8 9 10) ;;
  P-8:10-level) PARAX_MODE=image_level; PARAX_LAYERS=(8 9 10) ;;
  Static-control) PARAX_MODE=static; PARAX_LAYERS=(8 9 10) ;;
  *) echo "Unknown METHOD=${METHOD}" >&2; exit 2 ;;
esac
echo "ParaX task0 smoke method=${METHOD} gpu=${GPU} mode=${PARAX_MODE} layers=${PARAX_LAYERS[*]}"
precision_args=()
if [[ "${NO_AMP:-0}" == 1 ]]; then precision_args+=(--no-amp); fi
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m multi_lane.track_a.smoke --clip-checkpoint "${CLIP_CHECKPOINT}" --view-fusion fixed_three_view --adapter-mode disabled --parax-mode "${PARAX_MODE}" --parax-rank "${PARAX_RANK:-32}" --parax-num-experts "${PARAX_EXPERTS:-3}" --parax-layer-indices "${PARAX_LAYERS[@]}" --parax-router-hidden "${PARAX_ROUTER_HIDDEN:-16}" --parax-residual-scale "${PARAX_SCALE:-0.1}" --parax-initialization "${PARAX_INITIALIZATION:-official}" "${precision_args[@]}"
