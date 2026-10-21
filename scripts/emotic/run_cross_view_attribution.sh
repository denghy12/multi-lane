#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
GPU="${GPU:-0}"
[[ "${GPU}" == 0 ]] || { echo "Locked attribution uses GPU0" >&2; exit 2; }
RUN_ID="${RUN_ID:-formal_seed0_20261021_01}"
RUN_ROOT="output/cross_view_attribution/${RUN_ID}"
LOG_ROOT="logs/cross_view_attribution/${RUN_ID}"
[[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "Existing run ID" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean Git required" >&2; exit 2; }
"${PYTHON}" - <<'GPU'
import subprocess
uuid,free,util=(x.strip() for x in subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
assert uuid not in apps and int(free)>=16000 and int(util)<=5,(free,util)
GPU
SOURCE_ROOT="${SOURCE_ROOT:-../multi-lane-main-person-conditioned-cross-view-20261021/output/person_conditioned_cross_view/formal_seed0_20261021_01}"
DATA_ROOT="${DATA_ROOT:-../multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-../CODE_DDP/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
trap 'result=$?; echo "${result}" > "${LOG_ROOT}/batch_exit_status.txt"' EXIT
export OMP_NUM_THREADS=1
args=()
splits=(val test)
if [[ "${SMOKE:-0}" == 1 ]]; then
  args=(--checkpoint-tasks 0 2 7 --smoke-batches 2)
  splits=(val)
fi
pids=()
for arm in task_full person_full; do
  for split in "${splits[@]}"; do
    name="${arm}_${split}"
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u -m multi_lane.track_a.evaluate_cross_view_attribution \
      --source-run "${SOURCE_ROOT}/${arm}/train" --output "${RUN_ROOT}/${name}" \
      --split "${split}" --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" \
      --face-manifest-root "${FACE_MANIFEST_ROOT}" --batch-size 64 --minimum-visibility 0.9 \
      "${args[@]}" > "${LOG_ROOT}/${name}.log" 2>&1 &
    pids+=("$!")
  done
done
failed=0
for i in "${!pids[@]}"; do
  status=0
  wait "${pids[i]}" || status=$?
  echo "index=${i} pid=${pids[i]} exit=${status}" >> "${LOG_ROOT}/job_status.txt"
  [[ "${status}" == 0 ]] || failed=1
done
[[ "${failed}" == 0 ]] || exit 1
"${PYTHON}" - "${RUN_ROOT}" "${SMOKE:-0}" <<'VERIFY'
from pathlib import Path
import json,sys
root=Path(sys.argv[1]);smoke=sys.argv[2]=='1';splits=('val',) if smoke else ('val','test')
rows=[]
for split in splits:
    plans=[]
    for arm in ('task_full','person_full'):
        p=root/(arm+'_'+split);r=json.loads((p/'result.json').read_text())
        assert r['status']=='complete' and r['source_unchanged'] and not r['training_performed']
        assert r['method_weights_unchanged'] and r['frozen_visual_unchanged']
        assert r['full_and_face_endpoints_bitwise_unchanged'] and r['noneligible_query_swap_bitwise_unchanged']
        assert r['checkpoint_tasks']==([0,2,7] if smoke else list(range(8)))
        if arm=='person_full':
            assert all(r['details']['on'][str(t)]['identity_query_bitwise_checked'] for t in r['checkpoint_tasks'])
            assert all(r['details']['query_swap'][str(t)]['subsets']['eligible']['samples']>0 for t in r['checkpoint_tasks'])
        plans.append(json.loads((p/'query_pairs.json').read_text()))
        rows.append(dict(arm=arm,split=split,summary=r['summary'],query_eligible_people=r['query_eligible_people']))
    assert plans[0]==plans[1],'Cross-model pairing mismatch'
(root/'batch_verification.json').write_text(json.dumps(dict(status='verified',smoke=smoke,checks=rows),indent=2)+'\n')
print('ALL_CROSS_VIEW_ATTRIBUTION_CHECKS_PASSED')
VERIFY
echo "CROSS_VIEW_ATTRIBUTION_BATCH_COMPLETE ${RUN_ID}"
