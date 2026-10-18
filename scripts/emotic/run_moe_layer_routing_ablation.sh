#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
RUN_ID="${RUN_ID:-formal_seed0_20261018_01}"
[[ "${SEED:-0}" == 0 ]] || { echo "Locked to seed0" >&2; exit 2; }
RUN_ROOT="${ROOT}/output/moe_layer_routing_ablation/${RUN_ID}"
LOG_ROOT="${ROOT}/logs/moe_layer_routing_ablation/${RUN_ID}"
[[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "Batch ID already exists" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean Git worktree required" >&2; exit 2; }
# One check before starting all arms, allowing intentional sharing within this batch.
"${PYTHON}" - <<'GPU_CHECK'
import subprocess
busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
for gpu in (0,1):
    uuid,free,util=(v.strip() for v in subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
    if uuid in busy or int(free)<22000 or int(util)>5:
        raise SystemExit(f'GPU{gpu} must be idle with >=22000 MiB free; no existing process will be stopped')
GPU_CHECK
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
trap 'batch_exit=$?; echo "${batch_exit}" > "${LOG_ROOT}/exit_status.txt"' EXIT
arms=(L1_K2_F2 L5_K2_F2 L11_K2_F2 L1_K2_F3 L1_K3_F2 L1_K3_F3)
export PYTHON RUN_ID ABLATION_BATCH_PRECHECKED=1 OMP_NUM_THREADS=1
"${PYTHON}" - "${RUN_ROOT}" <<'MANIFEST'
import json,os,subprocess,sys
from pathlib import Path
arms=[dict(arm=f'L{l}_K{k}_F{f}',gpu=g,layer=l,router_top_k=k,freeze_count=f)
      for l,k,f,g in [(1,2,2,0),(5,2,2,0),(11,2,2,0),(1,2,3,1),(1,3,2,1),(1,3,3,1)]]
manifest=dict(run_id=os.environ['RUN_ID'],seed=0,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              smoke_updates=os.environ.get('SMOKE_UPDATES'),arms=arms,status='running',
              layer_group=['L1_K2_F2','L5_K2_F2','L11_K2_F2'],
              routing_freezing_group=['L1_K2_F2','L1_K2_F3','L1_K3_F2','L1_K3_F3'],
              shared_control='L1_K2_F2',freeze_metric='gate_mass',test_used_for_selection=False)
(Path(sys.argv[1])/'batch_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
MANIFEST
pids=()
for arm in "${arms[@]}"; do
  ARM="${arm}" bash scripts/emotic/run_moe_layer_routing_arm.sh > "${LOG_ROOT}/${arm}.launch.log" 2>&1 &
  pids+=("$!")
done
# Wait every child even when an earlier arm fails; retain all exit codes.
failed=0
for i in "${!arms[@]}"; do
  status=0
  wait "${pids[$i]}" || status=$?
  printf '%s\t%s\t%s\n' "${arms[$i]}" "${pids[$i]}" "${status}" >> "${LOG_ROOT}/arm_status.tsv"
  [[ "${status}" == 0 ]] || failed=1
done
"${PYTHON}" - "${RUN_ROOT}" "${LOG_ROOT}/arm_status.tsv" <<'SUMMARY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'batch_manifest.json'
manifest=json.loads(p.read_text())
statuses={a:dict(pid=int(pid),exit_code=int(code)) for a,pid,code in (line.split() for line in Path(sys.argv[2]).read_text().splitlines())}
manifest['arm_status']=statuses
manifest['status']='complete' if all(x['exit_code']==0 for x in statuses.values()) else 'failed'
p.write_text(json.dumps(manifest,indent=2)+'\n')
SUMMARY
exit "${failed}"
