#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
BATCH_ID="${BATCH_ID:?Set fresh BATCH_ID}"
GPU="${GPU:-0}"
SOURCE_ROOT="${SOURCE_ROOT:-../multi-lane-main-three-view-joint-to-incremental/output/emotic_three_view_joint_transfer/three_view_joint_to_incremental_val_test_20261010_01/seed0}"
PILOT_REFERENCE="${PILOT_REFERENCE:-../multi-lane-main-whole-parax-freeze/output/emotic_whole_parax_control/three_view_whole_parax_freeze_seed0_val_20261011_01/frozen}"
DATA_ROOT="${DATA_ROOT:-../multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}"
result="./output/emotic_whole_parax_eight_task/${BATCH_ID}"
logs="./logs/emotic_whole_parax_eight_task/${BATCH_ID}"
[[ ! -e "${result}" && ! -e "${logs}" ]] || { echo 'Batch exists' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'Tracked source dirty' >&2; exit 2; }
mkdir -p "${result}/control" "${logs}"
trap 'echo failed > "${result}/control/state"' ERR
cat > "${result}/control/manifest.txt" <<EOF
branch=$(git branch --show-current);commit=$(git rev-parse HEAD)
plan=docs/three_view_whole_parax_eight_task_val_test_plan_20261012.md
seed0 only;8 tasks;30epochs/task;13950 updates;batch64;workers2;Adam reset/task;mainLR0.0125;ParaX task0 LR0.0004;WD0;cosine;AMP1024 growth1e9
Frozen Forward index10;rank32;3 low-rank matrix pairs;router16;official init;learnable scale0.1 in task0;entire ParaX frozen task1 onward
Image-token Adapter disabled;shared three views;fixed feature fusion;no level/distillation/new experts
validation-only training then full2397-person validation and full5368-person test of all8 immutable checkpoints
no checkpoint selection;no test weight search;reuse baseline/live checkpoint sources=${SOURCE_ROOT};pilot=${PILOT_REFERENCE}
inputs=${DATA_ROOT},${CLIP_CHECKPOINT},${FACE_MANIFEST_ROOT};outputs=${result};logs=${logs};GPU=${GPU}
EOF
echo waiting_for_idle_gpu > "${result}/control/state"
deadline=$(( $(date +%s) + ${WAIT_SECONDS:-43200} ))
while true; do
  IFS=',' read -r free util < <(nvidia-smi -i "${GPU}" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)
  free="${free// /}";util="${util// /}"
  (( free >= 14000 && util <= 15 )) && break
  (( $(date +%s) < deadline )) || { echo 'GPU deadline expired' >&2; exit 2; }
  sleep 45
done
echo training > "${result}/control/state"
set +e
MAX_TASKS=8 SEED=0 EPOCHS=30 GPU="${GPU}" PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" \
  CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" \
  RUN_ROOT="${result}/frozen" LOG_PATH="${logs}/frozen.log" \
  bash scripts/emotic/run_whole_parax_freeze.sh > "${logs}/frozen.launcher.log" 2>&1
code=$?
set -e
echo "${code}" > "${result}/control/train.exit_code"
(( code == 0 )) || { echo failed > "${result}/control/state"; exit "${code}"; }
for split in val test; do
  echo "evaluating_${split}" > "${result}/control/state"
  extra=()
  if [[ "${split}" == test ]]; then
    extra+=(--validation-result "${result}/full_val/comparison.json")
  fi
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" OMP_NUM_THREADS=1 "${PYTHON}" -m multi_lane.track_a.evaluate_whole_parax_eight_task \
    --source "${result}/frozen" --reference-root "${SOURCE_ROOT}" --pilot-reference "${PILOT_REFERENCE}" \
    --output "${result}/full_${split}" --split "${split}" --data-root "${DATA_ROOT}" \
    --clip-checkpoint "${CLIP_CHECKPOINT}" --face-manifest-root "${FACE_MANIFEST_ROOT}" "${extra[@]}" \
    > "${logs}/full_${split}.log" 2>&1
  code=$?
  set -e
  echo "${code}" > "${result}/control/${split}.exit_code"
  (( code == 0 )) || { echo failed > "${result}/control/state"; exit "${code}"; }
done
echo complete > "${result}/control/state"
date -Is > "${result}/control/complete"
