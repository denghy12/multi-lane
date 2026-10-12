#!/usr/bin/env bash
# Fixed seed0 baselines; complete expert outputs; old full functions are frozen.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/ddp/bin/python}
GPU=${GPU:-0}
SMOKE=${SMOKE:-0}
GPU_POLICY=${GPU_POLICY:-idle}
BATCH=${BATCH:?Set a new immutable BATCH}
SOURCE=${SOURCE:-../multi-lane-main-parax-freezing-eight-task-relaunch/output/emotic_protected_parax_full/protected_parax_freezing_8task_seed012_valtest_20261011_01/seed0/baseline}
CACHE_SOURCE=${CACHE_SOURCE:-../multi-lane-main-parax-fixed-baseline-cached-routing/output/emotic_fixed_baseline_routes/fixed_baseline_cached_routes_20261011_01/cached_fit/seed0}
DATA_ROOT=${DATA_ROOT:-../multi-lane-main/datasets/EMOTIC}
CLIP_CHECKPOINT=${CLIP_CHECKPOINT:-../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}
FACE_MANIFEST_ROOT=${FACE_MANIFEST_ROOT:-../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}
OUTPUT=./output/emotic_staged_functional_experts/$BATCH
LOGS=./logs/emotic_staged_functional_experts/$BATCH
test ! -e "$OUTPUT"
test ! -e "$LOGS"
mkdir -p "$LOGS"
"$PYTHON_BIN" - "$GPU" "$LOGS/resource.json" "$GPU_POLICY" <<'PY'
import json,subprocess,sys,time
from pathlib import Path
used,free,util=map(int,subprocess.check_output(['nvidia-smi','-i',sys.argv[1],
    '--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
policy=sys.argv[3]
if policy not in ('idle','shared'):raise SystemExit('GPU_POLICY must be idle or shared')
record=dict(gpu=sys.argv[1],used_mib=used,free_mib=free,utilization=util,time=time.time(),
            policy=policy,existing_processes_untouched=True,minimum_free_mib=8192,maximum_idle_used_mib=512,maximum_idle_utilization=5)
Path(sys.argv[2]).write_text(json.dumps(record,indent=2)+'\n')
if free<8192 or (policy=='idle' and (used>512 or util>5)):
    raise SystemExit('Selected GPU violates the declared resource policy; no training started')
PY
extra=()
case "$SMOKE" in 0) ;; 1) extra+=(--smoke) ;; *) echo 'SMOKE must be 0 or 1' >&2; exit 2 ;; esac
set +e
CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 "$PYTHON_BIN" -u -m multi_lane.track_a.fixed_baseline_staged_functions \
 --source "$SOURCE" --cache-source "$CACHE_SOURCE" --output "$OUTPUT" \
 --data-root "$DATA_ROOT" --clip-checkpoint "$CLIP_CHECKPOINT" --face-manifest-root "$FACE_MANIFEST_ROOT" \
 "${extra[@]}" > "$LOGS/experiment.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$LOGS/exit_code"
exit "$rc"
