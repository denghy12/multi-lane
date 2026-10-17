#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
BATCH_ID="${BATCH_ID:-seed01_20261017_01}"
PHASE="${PHASE:-formal}"
[[ "${PHASE}" == formal || "${PHASE}" == smoke ]] || exit 2
[[ -z "$(git status --porcelain)" ]] || { echo "Clean worktree required" >&2; exit 2; }
batch_root="${ROOT}/logs/moe_task_ffn_v4/batch_${PHASE}_${BATCH_ID}"
[[ ! -e "${batch_root}" ]] || { echo "Batch ID already exists" >&2; exit 2; }
# Use every currently idle GPU; never terminate or compete with another job.
gpu_text="$("${PYTHON}" - <<'PY'
import subprocess
query = subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
    '--format=csv,noheader,nounits'], text=True)
busy = set(subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid',
    '--format=csv,noheader'], text=True).splitlines())
for row in query.splitlines():
    index, uuid, free, util = (x.strip() for x in row.split(','))
    if uuid not in busy and int(free) >= 16000 and int(util) <= 5:
        print(index)
PY
)"
gpus=()
while IFS= read -r gpu; do
  [[ -n "${gpu}" ]] && gpus+=("${gpu}")
done <<< "${gpu_text}"
[[ ${#gpus[@]} -gt 0 ]] || { echo "No idle GPU available" >&2; exit 2; }
mkdir -p "${batch_root}"
git rev-parse HEAD > "${batch_root}/source_commit.txt"
jobs=(baseline:0 shared_fixed:0 view_fixed:0 dynamic:0 baseline:1 shared_fixed:1 view_fixed:1 dynamic:1)
worker() {
  local slot="$1" gpu="$2" index job mode seed code
  for ((index=slot; index<${#jobs[@]}; index+=${#gpus[@]})); do
    job="${jobs[index]}"; mode="${job%:*}"; seed="${job#*:}"
    echo "START gpu=${gpu} mode=${mode} seed=${seed} phase=${PHASE}"
    code=0
    if [[ "${PHASE}" == smoke ]]; then
      GPU="${gpu}" ROUTE_MODE="${mode}" SEED="${seed}" SMOKE_UPDATES=2 \
        RUN_ID="smoke_${mode}_seed${seed}_${BATCH_ID}" \
        bash scripts/emotic/run_moe_task_ffn_v4.sh \
        > "${batch_root}/${mode}_seed${seed}.log" 2>&1 || code=$?
    else
      GPU="${gpu}" ROUTE_MODE="${mode}" SEED="${seed}" \
        RUN_ID="formal_${mode}_seed${seed}_${BATCH_ID}" \
        bash scripts/emotic/run_moe_task_ffn_v4.sh \
        > "${batch_root}/${mode}_seed${seed}.log" 2>&1 || code=$?
    fi
    printf '%s\t%s\t%s\t%s\n' "${mode}" "${seed}" "${gpu}" "${code}" \
      >> "${batch_root}/exit_status.tsv"
    [[ ${code} == 0 ]] || return "${code}"
  done
}
pids=()
for ((slot=0; slot<${#gpus[@]} && slot<${#jobs[@]}; slot++)); do
  worker "${slot}" "${gpus[slot]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ ${status} == 0 ]] || exit 1
echo "MOE_TASK_FFN_V4_BATCH_COMPLETE phase=${PHASE} batch=${BATCH_ID}"
