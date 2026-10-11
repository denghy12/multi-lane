#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
BATCH_ID="${BATCH_ID:?Set fresh freeze BATCH_ID}"
TEST_BATCH_ID="${TEST_BATCH_ID:?Set fresh TEST_BATCH_ID}"
source_root='../multi-lane-main-three-view-joint-to-incremental/output/emotic_three_view_joint_transfer/three_view_joint_to_incremental_val_test_20261010_01/seed0'
result="./output/emotic_whole_parax_control/${BATCH_ID}"
logs="./logs/emotic_whole_parax_control/${BATCH_ID}"
test_result="./output/emotic_snapshot_test/${TEST_BATCH_ID}"
test_logs="./logs/emotic_snapshot_test/${TEST_BATCH_ID}"
[[ ! -e "${result}" && ! -e "${test_result}" ]] || { echo 'Batch exists' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'Tracked source dirty' >&2; exit 2; }
mkdir -p "${result}/control" "${logs}" "${test_result}/control" "${test_logs}"
trap 'echo failed > "${result}/control/state"' ERR
cat > "${result}/control/manifest.txt" <<EOF
branch=$(git branch --show-current); commit=$(git rev-parse HEAD)
configuration=docs/three_view_whole_parax_freeze_and_seed0_test_plan_20261011.md
seed0; frozen control task0-2 only; eight-task allocation; original official init/learnable scale0.1; image index10/rank32/expert3/router16; all ParaX frozen task1 onward
30epochs/task;batch64;workers2;Adam reset/task;mainLR0.0125;ParaXLR0.0004;WD0;cosine;AMP1024 growth1e9;no new test in training
separate locked test: seed0 baseline and live-ParaX eight existing snapshots/all5368people/26classes; no training or weight search
sources=${source_root};outputs=${result},${test_result};logs=${logs},${test_logs}
resources=GPU0 only; wait<=12h for >=14000MiB free/util<=15%; two processes inference+training; no interruption of other jobs
EOF
cp "${result}/control/manifest.txt" "${test_result}/control/manifest.txt"
echo waiting_for_gpu0 > "${result}/control/state"
echo waiting_for_gpu0 > "${test_result}/control/state"
deadline=$(( $(date +%s) + ${WAIT_SECONDS:-43200} ))
while true; do
  IFS=',' read -r free util < <(nvidia-smi -i 0 --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)
  free="${free// /}";util="${util// /}"
  (( free >= 14000 && util <= 15 )) && break
  (( $(date +%s) < deadline )) || { echo 'GPU0 deadline expired' >&2; exit 2; }
  sleep 45
done
(
  set +e
  echo evaluating_test > "${test_result}/control/state"
  code=0
  for method in THREE_VIEW THREE_VIEW_PARAX; do
    EVAL_SPLIT=test SEED=0 METHOD="${method}" GPU=0 PYTHON="${PYTHON}" \
      RESULT_ROOT="${test_result}" LOG_ROOT="${test_logs}" \
      bash scripts/emotic/run_snapshot_validation.sh > "${test_logs}/${method}.launcher.log" 2>&1
    run_code=$?;echo "${run_code}" > "${test_result}/control/${method}.exit_code"
    if (( run_code != 0 ));then code="${run_code}";break;fi
  done
  if (( code == 0 ));then
    "${PYTHON}" -m multi_lane.track_a.compare_snapshot_validation --root "${test_result}" --split test --seeds 0 > "${test_logs}/comparison.log" 2>&1
    code=$?
  fi
  echo "${code}" > "${result}/control/test.exit_code"
  if (( code == 0 ));then echo complete > "${test_result}/control/state";date -Is > "${test_result}/control/complete";else echo failed > "${test_result}/control/state";fi
  exit "${code}"
) &
test_pid=$!
(
  set +e
  echo training > "${result}/control/state"
  SEED=0 GPU=0 PYTHON="${PYTHON}" RUN_ROOT="${result}/frozen" LOG_PATH="${logs}/frozen.log" \
    bash scripts/emotic/run_whole_parax_freeze.sh > "${logs}/frozen.launcher.log" 2>&1
  code=$?;echo "${code}" > "${result}/control/train.exit_code"
  if (( code == 0 ));then
    echo full_validation_comparison > "${result}/control/state"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 "${PYTHON}" -m multi_lane.track_a.compare_whole_parax_freeze \
      --source "${result}/frozen" --reference-root "${source_root}" --output "${result}/full_validation" \
      --data-root ../multi-lane-main/datasets/EMOTIC \
      --clip-checkpoint ../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt \
      --face-manifest-root ../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909 \
      > "${logs}/full_validation_comparison.log" 2>&1
    code=$?
  fi
  echo "${code}" > "${result}/control/freeze_and_comparison.exit_code"
  exit "${code}"
) &
train_pid=$!
failed=0
wait "${test_pid}" || failed=1
wait "${train_pid}" || failed=1
(( failed == 0 )) || { echo 'A declared job failed; artifacts retained' >&2;exit 1; }
echo complete > "${result}/control/state"
date -Is > "${result}/control/complete"
