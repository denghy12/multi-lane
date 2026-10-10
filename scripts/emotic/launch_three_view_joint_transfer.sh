#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BATCH_ID="${BATCH_ID:?Set fresh BATCH_ID}"
PYTHON="${PYTHON:-python}"
result="./output/emotic_three_view_joint_transfer/${BATCH_ID}"
logs="./logs/emotic_three_view_joint_transfer/${BATCH_ID}"
[[ ! -e "${result}" ]] || { echo 'Batch already exists' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'Tracked source is dirty' >&2; exit 2; }
mkdir -p "${result}/control" "${logs}"
trap 'echo failed > "${result}/control/state"' ERR
cat > "${result}/control/manifest.txt" <<EOF
batch=${BATCH_ID}; branch=$(git branch --show-current); commit=$(git rev-parse HEAD)
scope=original joint26 THREE_VIEW vs THREE_VIEW_PARAX transferred to incremental 8 tasks; NOT protected post-task ParaX
seeds=${SEEDS:-0 1 2}; GPUs=${GPUS:-0 1 2 3 4 5}; one process per GPU; smoke_updates=${SMOKE_UPDATES:-0}
configuration=docs/three_view_joint_to_incremental_val_test_plan_20261010.md
training=8tasks[5,3,3,3,3,3,3,3];30epochs/task;batch64/eval64/workers2;Adam reset/task;mainLR0.0125;ParaXLR0.0004;WD0;cosine;AMP1024 growth1e9;OMP1
architecture=CLIP frozen;Image-token Adapter off;shared selectors/prompts/head;fixed reliable three-view FEATURE fusion;ParaX only image stream index10 rank32 experts3 router16 official init learnable scale0.1, all shared ParaX parameters live
reporting=task-end validation+test from identical weights; smoke validation only; no test selection/no early stopping
inputs=../multi-lane-main/datasets/EMOTIC;../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt;../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909
outputs=${result};logs=${logs};all task compact checkpoints + fused/per-view scores + fixed val-anchor FP32 drift + old parameter hashes
EOF
"${PYTHON}" -c 'import json; from pathlib import Path; from multi_lane.track_a.evaluate_joint26_checkpoint import audit_face_manifest; c=json.loads(Path("../multi-lane-main-joint26-parax-seed-replication/output/emotic_joint26_parax_view_ablation/joint26_stableamp_multiseed_val_gpu0_20261009_01_seed0/THREE_VIEW/config.json").read_text()); audit_face_manifest(c,Path("../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909")); print("FACE_MANIFEST_EQUIVALENCE_PASSED")'
read -r -a gpu_list <<< "${GPUS:-0 1 2 3 4 5}"
read -r -a seed_list <<< "${SEEDS:-0 1 2}"
(( ${#gpu_list[@]} == 2 * ${#seed_list[@]} )) || { echo 'Need one GPU per run' >&2; exit 2; }
# Refuse occupied devices; never kill or evict another experiment.
for gpu in "${gpu_list[@]}"; do
  free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free >= 20000 )) || { echo "GPU${gpu} is occupied: ${free}MiB free" >&2; exit 2; }
done
echo training > "${result}/control/state"
pids=(); slot=0
for seed in "${seed_list[@]}"; do
  [[ "${seed}" =~ ^[012]$ ]] || exit 2
  mkdir -p "${logs}/seed${seed}"
  for method in THREE_VIEW THREE_VIEW_PARAX; do
    gpu="${gpu_list[${slot}]}"; slot=$((slot+1))
    (
      set +e
      METHOD="${method}" SEED="${seed}" GPU="${gpu}" PYTHON="${PYTHON}" \
        RUN_ROOT="${result}/seed${seed}/${method}" LOG_PATH="${logs}/seed${seed}/${method}.log" \
        bash scripts/emotic/run_three_view_joint_transfer.sh > "${logs}/seed${seed}/${method}.launcher.log" 2>&1
      code=$?
      echo "${code}" > "${result}/control/seed${seed}_${method}.exit_code"
      exit "${code}"
    ) &
    pids+=("$!")
  done
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo 'A run failed; artifacts retained' >&2; exit 1; }
for seed in "${seed_list[@]}"; do
  "${PYTHON}" -m multi_lane.track_a.compare_three_view_joint_transfer --root "${result}/seed${seed}" --smoke-updates "${SMOKE_UPDATES:-0}"
done
if [[ -z "${SMOKE_UPDATES:-}" && "${SEEDS:-0 1 2}" == '0 1 2' ]]; then
  "${PYTHON}" -m multi_lane.track_a.compare_three_view_joint_transfer --root "${result}" --summarize
fi
echo complete > "${result}/control/state"
date -Is > "${result}/control/complete"
echo "THREE_VIEW_TRANSFER_COMPLETE batch=${BATCH_ID}"
