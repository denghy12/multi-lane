#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

read -r -a GPUS <<< "${GPU_LIST:-3 4 7}"
METHODS=(model_asl adapter_asl both_asl)
SESSION="${SESSION:-mla_image_token_asl_seed0_$(date +%H%M%S)}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"
STATE_DIR="${ROOT}/logs/emotic_track_a_image_token_asl_formal/${STAMP}_launchers"

[[ "${#GPUS[@]}" -eq 3 ]] || { echo "GPU_LIST must contain exactly three GPUs" >&2; exit 2; }
[[ "${GPUS[0]}" != "${GPUS[1]}" && "${GPUS[0]}" != "${GPUS[2]}" && "${GPUS[1]}" != "${GPUS[2]}" ]] || {
  echo "GPU_LIST must contain three distinct GPUs" >&2; exit 2;
}
[[ -z "$(git status --porcelain)" ]] || { echo "Formal launch requires a clean Git worktree" >&2; exit 2; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 2; }
mkdir -p "${STATE_DIR}"

for gpu in "${GPUS[@]}"; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
  [[ "${free_mib}" =~ ^[0-9]+$ ]] || { echo "Cannot read free memory for GPU${gpu}" >&2; exit 2; }
  (( free_mib >= MIN_FREE_MIB )) || {
    echo "GPU${gpu} has only ${free_mib} MiB free; require ${MIN_FREE_MIB} MiB" >&2; exit 2;
  }
done

for index in 0 1 2; do
  method="${METHODS[$index]}"
  gpu="${GPUS[$index]}"
  run_id="image_token_${method}_b32_layer8_seed0_${STAMP}"
  launcher_log="${STATE_DIR}/${method}_gpu${gpu}.launcher.log"
  command="cd '${ROOT}' && GPU='${gpu}' LOSS_ROUTING='${method}' RUN_ID='${run_id}' bash scripts/emotic/run_multilane_track_a_image_token_asl_seed0.sh > '${launcher_log}' 2>&1; code=\$?; echo LAUNCHER_EXIT_CODE=\$code >> '${launcher_log}'; exit \$code"
  if [[ "${index}" -eq 0 ]]; then
    tmux new-session -d -s "${SESSION}" -n "${method}" "${command}"
  else
    tmux new-window -t "${SESSION}" -n "${method}" "${command}"
  fi
  echo "started method=${method} gpu=${gpu} run_id=${run_id} launcher_log=${launcher_log}"
done

echo "session=${SESSION} state_dir=${STATE_DIR}"
