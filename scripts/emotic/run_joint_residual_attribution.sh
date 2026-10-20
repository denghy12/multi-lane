#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
GPU="${GPU:-5}"
ARM="${ARM:-dense}"
[[ "$GPU" =~ ^[56]$ && "$ARM" =~ ^(dense|moe)$ ]] || exit 2
[[ -z "$(git status --porcelain)" ]] || { echo 'Clean source required'; exit 2; }
RUN_ID="${RUN_ID:-formal_seed0_20261020_01}"
SOURCE="../multi-lane-main-joint-view-summary-20261020/output/joint_view_summary/formal_seed0_20261020_01/${ARM}"
OUT="output/joint_residual_attribution/${RUN_ID}/${ARM}"
LOG="logs/joint_residual_attribution/${RUN_ID}/${ARM}"
[[ ! -e "$OUT" && ! -e "$LOG" ]] || exit 2
mkdir -p "$LOG"
trap 'status=$?; echo "$status" > "$LOG/exit_status.txt"' EXIT
export OMP_NUM_THREADS=1
TASKS=(0 2 7)
[[ "${SMOKE:-0}" == 0 ]] || TASKS=(0)
for mode in on off; do
 args=()
 [[ "$mode" == on ]] || args=(--disable-joint-residual)
 CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u -m multi_lane.track_a.evaluate_moe_task_ffn \
  --source-run "$SOURCE/train" --output "$OUT/$mode" --split val --full-cohort \
  --checkpoint-tasks "${TASKS[@]}" --expected-tasks 8 --batch-size 64 --workers 0 \
  --data-root ../multi-lane-main/datasets/EMOTIC \
  --clip-checkpoint ../CODE_DDP/pretrained/clip/ViT-B-16.pt \
  --face-manifest-root ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909 \
  "${args[@]}" > "$LOG/$mode.log" 2>&1
 if [[ "$mode" == on ]]; then
 "$PYTHON" - "$SOURCE" "$OUT/on" "${TASKS[@]}" <<'PY'
import sys,json
from pathlib import Path
import numpy as np
source,out=map(Path,sys.argv[1:3]); checks={}
for t in map(int,sys.argv[3:]):
 a=np.load(source/f'full_val/scores/task{t}.npz',allow_pickle=False)
 b=np.load(out/f'scores/task{t}.npz',allow_pickle=False)
 assert set(a.files)==set(b.files),(a.files,b.files)
 for k in a.files:
  assert np.array_equal(a[k],b[k]),(t,k)
 checks[str(t)]='all_arrays_bitwise_equal'
(out/'original_replay.json').write_text(json.dumps(checks,indent=2)+'\n')
print('ORIGINAL_REPLAY_BITWISE_EQUAL',checks)
PY
 fi
done
