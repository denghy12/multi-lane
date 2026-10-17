#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
BATCH_ID="${BATCH_ID:-seed0_20261017_01}"
PHASE="${PHASE:-formal}"
[[ "${PHASE}" == formal || "${PHASE}" == smoke ]] || exit 2
[[ -z "$(git status --porcelain)" ]] || { echo "Clean worktree required" >&2; exit 2; }
batch_root="${ROOT}/logs/moe_task_ffn_v4/batch_${PHASE}_${BATCH_ID}"
[[ ! -e "${batch_root}" ]] || { echo "Batch ID already exists" >&2; exit 2; }
# User's final scope: GPU0 only. Four ~3GiB jobs fit a free 24GiB 4090.
"${PYTHON}" - <<'PY'
import subprocess
row = subprocess.check_output([
    'nvidia-smi', '-i', '0', '--query-gpu=uuid,memory.free,utilization.gpu',
    '--format=csv,noheader,nounits'], text=True).strip()
uuid, free, util = (x.strip() for x in row.split(','))
busy = set(subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid',
    '--format=csv,noheader'], text=True).splitlines())
if uuid in busy or int(free) < 20000 or int(util) > 5:
    raise SystemExit('GPU0 must be idle with >=20000MiB free; no existing job will be stopped')
PY
mkdir -p "${batch_root}"
git rev-parse HEAD > "${batch_root}/source_commit.txt"
worker() {
  local mode="$1" code=0
  echo "START gpu=0 mode=${mode} seed=0 phase=${PHASE}"
  if [[ "${PHASE}" == smoke ]]; then
    GPU=0 ROUTE_MODE="${mode}" SEED=0 SMOKE_UPDATES=2 \
      RUN_ID="smoke_${mode}_${BATCH_ID}" \
      bash scripts/emotic/run_moe_task_ffn_v4.sh \
      > "${batch_root}/${mode}.log" 2>&1 || code=$?
  else
    GPU=0 ROUTE_MODE="${mode}" SEED=0 \
      RUN_ID="formal_${mode}_${BATCH_ID}" \
      bash scripts/emotic/run_moe_task_ffn_v4.sh \
      > "${batch_root}/${mode}.log" 2>&1 || code=$?
  fi
  printf '%s\t0\t0\t%s\n' "${mode}" "${code}" >> "${batch_root}/exit_status.tsv"
  return "${code}"
}
pids=()
for mode in baseline shared_fixed view_fixed dynamic; do
  worker "${mode}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ ${status} == 0 ]] || exit 1
echo "MOE_TASK_FFN_V4_BATCH_COMPLETE phase=${PHASE} batch=${BATCH_ID}"
