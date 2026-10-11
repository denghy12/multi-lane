#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BATCH_ID="${BATCH_ID:?Set fresh BATCH_ID}"
PYTHON="${PYTHON:-python}"
result="./output/emotic_snapshot_validation/${BATCH_ID}"
logs="./logs/emotic_snapshot_validation/${BATCH_ID}"
[[ ! -e "${result}" ]] || { echo 'Batch exists' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'Dirty source' >&2; exit 2; }
mkdir -p "${result}/control" "${logs}"
trap 'echo failed > "${result}/control/state"' ERR
cat > "${result}/control/manifest.txt" <<EOF
batch=${BATCH_ID}; branch=$(git branch --show-current); commit=$(git rev-parse HEAD)
configuration=docs/three_view_snapshot_full_validation_plan_20261011.md
source=../multi-lane-main-three-view-joint-to-incremental/output/emotic_three_view_joint_transfer/three_view_joint_to_incremental_val_test_20261010_01
scope=original internal image-stream ParaX and no-ParaX; seeds0/1/2; 8 immutable task checkpoints; all2397validation people and26classes; fixed class-owner composition, no best-AP selection
precision=source AMP/TF32; batch64/workers2; OMP1; no optimizer/no training/no new test images or scores
outputs=${result}; logs=${logs}; one GPU sequential jobs; wait for >=20000MiB free and util<=15%, deadline12h; no interruption of other jobs
EOF
echo waiting_for_idle_gpu > "${result}/control/state"
deadline=$(( $(date +%s) + ${WAIT_SECONDS:-43200} ))
gpu=''
while [[ -z "${gpu}" ]]; do
  while IFS=',' read -r index free util; do
    index="${index// /}"; free="${free// /}"; util="${util// /}"
    if (( free >= 20000 && util <= 15 )); then gpu="${index}"; break; fi
  done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
  [[ -n "${gpu}" ]] && break
  (( $(date +%s) < deadline )) || { echo 'No idle GPU within deadline' >&2; exit 2; }
  sleep 45
done
printf 'evaluating GPU%s\n' "${gpu}" > "${result}/control/state"
for seed in 0 1 2; do
  for method in THREE_VIEW THREE_VIEW_PARAX; do
    set +e
    PYTHON="${PYTHON}" SEED="${seed}" METHOD="${method}" GPU="${gpu}" \
      RESULT_ROOT="${result}" LOG_ROOT="${logs}" \
      bash scripts/emotic/run_snapshot_validation.sh > "${logs}/seed${seed}_${method}.launcher.log" 2>&1
    code=$?
    set -e
    echo "${code}" > "${result}/control/seed${seed}_${method}.exit_code"
    (( code == 0 )) || { echo 'Evaluation failed; outputs retained' >&2; exit 1; }
  done
done
"${PYTHON}" -m multi_lane.track_a.compare_snapshot_validation --root "${result}" > "${logs}/comparison.log" 2>&1
echo complete > "${result}/control/state"
date -Is > "${result}/control/complete"
