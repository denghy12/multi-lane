#!/usr/bin/env bash
# Run only the protected expert-freezing study; source checkpoints are read-only.
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/ddp/bin/python}
GPU=${GPU:-0}
BATCH=${BATCH:-fixed_baseline_cached_routes_20261011_01}
DATA_ROOT=${DATA_ROOT:-../multi-lane-main/datasets/EMOTIC}
CLIP_CHECKPOINT=${CLIP_CHECKPOINT:-../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}
FACE_MANIFEST_ROOT=${FACE_MANIFEST_ROOT:-../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}
OLD_ROOT=${OLD_ROOT:-../multi-lane-main-parax-audit-paired-staged/output/emotic_protected_parax_val/protected_parax_audit_v2_seed0_val_20261010_01}
BASE_ROOT=${BASE_ROOT:-../multi-lane-main-parax-freezing-eight-task-relaunch/output/emotic_protected_parax_full/protected_parax_freezing_8task_seed012_valtest_20261011_01}
OUTPUT_ROOT=./output/emotic_fixed_baseline_routes/$BATCH
LOG_ROOT=./logs/emotic_fixed_baseline_routes/$BATCH
mkdir -p "$LOG_ROOT"
# Do not overwrite any prior batch. Shared GPU use requires explicit opt-in.
test ! -e "$OUTPUT_ROOT"
"$PYTHON_BIN" - "$GPU" "$LOG_ROOT" <<'PY'
import json, os, subprocess, sys, time
from pathlib import Path
index = sys.argv[1]
policy = os.environ.get('GPU_START_POLICY', 'idle')
if policy not in ('idle', 'shared_memory'):
    raise SystemExit('Unknown GPU_START_POLICY: '+policy)
minimum_free = int(os.environ.get('GPU_MIN_FREE_MIB', '8192'))
path = Path(sys.argv[2]) / 'resource_wait.json'
deadline = time.monotonic() + int(os.environ.get('GPU_IDLE_TIMEOUT_SECONDS', '86400'))
announced = False
while True:
    apps = subprocess.check_output(['nvidia-smi','-i',index,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    used, free, utilization = map(int, subprocess.check_output(
        ['nvidia-smi','-i',index,'--query-gpu=memory.used,memory.free,utilization.gpu',
         '--format=csv,noheader,nounits'],text=True).strip().split(','))
    state = dict(gpu=index, timestamp=time.time(), policy=policy, compute_pids=apps.splitlines(),
                 memory_used_mib=used, memory_free_mib=free, utilization=utilization)
    ready = (free >= minimum_free if policy == 'shared_memory'
             else not apps and used < 512 and free >= 18000 and utilization <= 5)
    if ready:
        path.write_text(json.dumps(dict(state, status='ready'), indent=2)+'\n')
        break
    if policy == 'shared_memory':
        path.write_text(json.dumps(dict(state, status='insufficient_memory'), indent=2)+'\n')
        raise SystemExit('Insufficient free GPU memory for both workers: '+str(free)+' MiB')
    if not announced:
        path.write_text(json.dumps(dict(state, status='waiting'), indent=2)+'\n')
        print('Waiting for GPU '+index+' to be idle; other processes are untouched.', flush=True)
        announced = True
    if time.monotonic() >= deadline:
        path.write_text(json.dumps(dict(state, status='timed_out'), indent=2)+'\n')
        raise SystemExit('GPU remained occupied for 24h (or configured timeout); no experiment started.')
    time.sleep(30)
PY
mkdir -p "$OUTPUT_ROOT"
common=(--data-root "$DATA_ROOT" --clip-checkpoint "$CLIP_CHECKPOINT" --face-manifest-root "$FACE_MANIFEST_ROOT")
old_tests() {
    for method in baseline frozen_pool fresh_only reuse_old; do
        CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -u -m multi_lane.track_a.fixed_baseline_routes \
            --mode evaluate-old --source "$OLD_ROOT/$method" --output "$OUTPUT_ROOT/old_seed0_test/$method" \
            "${common[@]}" > "$LOG_ROOT/old_test_${method}.log" 2>&1 || return $?
    done
}
new_fits() {
    for seed in 0 1 2; do
        CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -u -m multi_lane.track_a.fixed_baseline_routes \
            --mode fit --source "$BASE_ROOT/seed$seed/baseline" --max-tasks 8 \
            --output "$OUTPUT_ROOT/cached_fit/seed$seed" "${common[@]}" > "$LOG_ROOT/cached_fit_seed${seed}.log" 2>&1 || return $?
    done
}
# Two inference/cache-fitting workers fit within the 24GB GPU in real smoke.
old_tests & old_pid=$!
new_fits & fit_pid=$!
printf 'old_test_pid=%s cached_fit_pid=%s gpu=%s\n' "$old_pid" "$fit_pid" "$GPU" > "$OUTPUT_ROOT/launch_state.txt"
old_rc=0; wait "$old_pid" || old_rc=$?
fit_rc=0; wait "$fit_pid" || fit_rc=$?
printf 'old_test_exit=%s cached_fit_exit=%s\n' "$old_rc" "$fit_rc" > "$OUTPUT_ROOT/completion_state.txt"
if [ "$old_rc" -ne 0 ] || [ "$fit_rc" -ne 0 ]; then exit 1; fi
