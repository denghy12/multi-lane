#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export GPU=0
export PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
export RUN_ID="${RUN_ID:-formal_seed0_20261021_01}"
LOG_ROOT="logs/person_conditioned_cross_view/${RUN_ID}"
RUN_ROOT="output/person_conditioned_cross_view/${RUN_ID}"
[[ ! -e "${LOG_ROOT}" && ! -e "${RUN_ROOT}" ]] || { echo "Batch already exists" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean Git required" >&2; exit 2; }
"${PYTHON}" - <<'GPU_CHECK'
import subprocess
uuid,free,util=(v.strip() for v in subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
assert uuid not in busy and int(free)>=22000 and int(util)<=5,(uuid,free,util)
GPU_CHECK
mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"
trap 'batch_exit=$?; echo "${batch_exit}" > "${LOG_ROOT}/batch_exit_status.txt"' EXIT
export BASELINE_RUN="${BASELINE_RUN:-../multi-lane-main-joint-view-summary-20261020/output/joint_view_summary/formal_seed0_20261020_01/baseline_noaux/train}"
"${PYTHON}" - "${BASELINE_RUN}" "${RUN_ROOT}/baseline_reference.json" <<'REFERENCE'
import hashlib,json,sys
from pathlib import Path
train=Path(sys.argv[1]); root=train.parent
read=lambda p:json.loads(p.read_text())
c=read(train/'config.json'); s=read(train/'seed_summary.json'); a=read(train/'paired_protocol_audit.json')
assert s['status']=='complete' and s['config']==c and c['git']['dirty'] is False
assert c['seed']==0 and c['max_tasks']==8 and c['epochs_per_task']==30
assert c['view_auxiliary_loss_weight']==0 and c['calibration_fraction']==0
assert c['joint_summary_mode']=='disabled' and c['adapter_mode']=='disabled' and c['parax_mode']=='disabled'
h=read(train/'training_history.json')
assert sum(r['optimizer_steps'] for rows in h.values() for r in rows)==13950
assert sum(r['skipped_optimizer_steps'] for rows in h.values() for r in rows)==0
assert a['frozen_visual_unchanged']
for split,n in [('full_val',2397),('full_test',5368)]:
    r=read(root/split/'result.json')
    assert r['full_cohort'] and r['cohort_samples']==n and len(r['task_metrics'])==8
files=[train/'config.json',train/'seed_summary.json',train/'paired_protocol_audit.json']
files+=list((train/'compact_checkpoints').glob('task*.pth'))
files+=list((root/'full_val/scores').glob('*.npz'))+list((root/'full_test/scores').glob('*.npz'))
assert len(files)==27,len(files)
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(p.read_bytes()).hexdigest()
record=dict(status='reference_audited',source_root=str(root.resolve()),source_git=c['git'],test_used_for_selection=False,sha256={str(p.resolve()):sha(p) for p in files},validation=read(root/'full_val/result.json')['summary'],test=read(root/'full_test/result.json')['summary'])
Path(sys.argv[2]).write_text(json.dumps(record,indent=2)+'\n')
print('BASELINE_REFERENCE_AUDITED',len(files),flush=True)
REFERENCE
pids=()
for arm in person_self task_full person_full; do
  ARM="${arm}" bash scripts/emotic/run_person_conditioned_cross_view.sh > "${LOG_ROOT}/${arm}.log" 2>&1 &
  pids+=("$!")
done
batch_failed=0
for i in "${!pids[@]}"; do
  status=0
  wait "${pids[i]}" || status=$?
  echo "index=${i} pid=${pids[i]} exit=${status}" >> "${LOG_ROOT}/arm_status.txt"
  [[ "${status}" == 0 ]] || batch_failed=1
done
[[ "${batch_failed}" == 0 ]] || exit 1
"${PYTHON}" - "${RUN_ROOT}/baseline_reference.json" <<'IMMUTABLE'
import hashlib,json,sys
from pathlib import Path
for name,expected in json.loads(Path(sys.argv[1]).read_text())['sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
print('BASELINE_SOURCE_UNCHANGED')
IMMUTABLE
echo "CROSS_VIEW_BATCH_COMPLETE ${RUN_ID}"
