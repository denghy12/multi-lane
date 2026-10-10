#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BATCH_ID="${BATCH_ID:?Set fresh BATCH_ID}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
MAX_TASKS="${MAX_TASKS:-8}"
result="./output/emotic_incremental_view_contribution/${BATCH_ID}"
logs="./logs/emotic_incremental_view_contribution/${BATCH_ID}"
[[ ! -e "${result}" ]] || { echo "Batch already exists" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Tracked modifications present" >&2; exit 2; }
mkdir -p "${result}/control" "${logs}"
trap 'printf "failed\n" > "${result}/control/state"' ERR
cat > "${result}/control/manifest.txt" <<EOF
batch=${BATCH_ID}; branch=$(git branch --show-current); commit=$(git rev-parse HEAD)
dataset=EMOTIC train only; input=../multi-lane-main/datasets/EMOTIC; CLIP=../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt
face=../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909; verified identical train/val to prior manifest
models=Full,Full+Person,Full+Face,Full+Person+Face; frozen shared CLIP ViT-B/16; Image-token Adapter=off; ParaX=off; shared Selector10/Prompt10 first5 layers/classifier across active views
protocol=8 incremental tasks[5,3,3,3,3,3,3,3]; seeds=${SEEDS}; max_tasks=${MAX_TASKS}; 30epochs/task; batch64/eval64/workers2; Adam reset/task; lr0.0125 WD0 cosine no warmup; temperature1 threshold0.5
precision=AMP+TF32 init_scale1024 growth1e9; require zero skipped; fixed per-task sampler seed=seed+1009*task, identical within each cohort; legacy default sampler is unchanged outside this explicit protocol
loss=Full:1.1*BCE; active multi-view:BCE(fused)+0.1*mean(BCE(active views)), Face loss only reliable rows
fusion=Full+Person[0.8,0.2]; Full+Face reliable[0.64/0.84,0.20/0.84] else[1,0]; three reliable[0.64,0.16,0.20] else[0.8,0.2,0]; shared feature fusion before head
reliability=validFace and nonambiguous and shortside>=24 and det_score>=0.6
evaluation=formal test-only after each task, no validation forward/test selection; smoke uses validation only and ${SMOKE_UPDATES:-0}updates/task
audit=initial parameters/CLIP hashes, per-task train/test IDs/first batch, same model capacity, all epochs/updates/scales, fixed task0 cohort old5class logit/AP drift
outputs=${result}; logs=${logs}; compact checkpoints at every task; no full CLIP checkpoint
execution=GPU${GPU} four concurrent per seed, sequential seeds; wait finite12h for >=20GB free, no automatic retuning or reruns
EOF
"${PYTHON}" -c 'import json; from pathlib import Path; from multi_lane.track_a.evaluate_joint26_checkpoint import audit_face_manifest; c=json.loads(Path("../multi-lane-main-joint26-parax-seed-replication/output/emotic_joint26_parax_view_ablation/joint26_stableamp_multiseed_val_gpu0_20261009_01_seed0/THREE_VIEW/config.json").read_text()); audit_face_manifest(c,Path("../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909")); print("FACE_MANIFEST_EQUIVALENCE_PASSED")'
for seed in ${SEEDS}; do
  [[ "${seed}" =~ ^[012]$ ]] || { echo "Invalid seed" >&2; exit 2; }
  deadline=$(( $(date +%s) + 43200 ))
  while true; do
    free="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    (( free >= 20000 )) && break
    printf 'waiting_gpu seed=%s free_mib=%s\n' "${seed}" "${free}" > "${result}/control/state"
    (( $(date +%s) < deadline )) || { echo "GPU wait timed out" >&2; exit 2; }
    sleep 45
  done
  printf 'training seed=%s\n' "${seed}" > "${result}/control/state"
  mkdir -p "${logs}/seed${seed}"
  pids=()
  for method in FULL_ONLY FULL_PERSON FULL_FACE THREE_VIEW; do
    (
      set +e
      METHOD="${method}" SEED="${seed}" GPU="${GPU}" PYTHON="${PYTHON}" MAX_TASKS="${MAX_TASKS}" \
        RUN_ROOT="${result}/seed${seed}/${method}" LOG_PATH="${logs}/seed${seed}/${method}.log" \
        bash scripts/emotic/run_incremental_view_contribution.sh > "${logs}/seed${seed}/${method}.launcher.log" 2>&1
      code=$?
      printf '%s\n' "${code}" > "${result}/control/seed${seed}_${method}.exit_code"
      exit "${code}"
    ) &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || { echo "Training failed; artifacts retained" >&2; exit 1; }
  "${PYTHON}" -m multi_lane.track_a.compare_incremental_view_contribution \
    --root "${result}/seed${seed}" --smoke-updates "${SMOKE_UPDATES:-0}"
  date -Is > "${result}/control/seed${seed}.complete"
done
if [[ -z "${SMOKE_UPDATES:-}" && "${SEEDS}" == "0 1 2" ]]; then
  "${PYTHON}" -m multi_lane.track_a.compare_incremental_view_contribution --root "${result}" --summarize
fi
printf 'complete\n' > "${result}/control/state"
date -Is > "${result}/control/complete"
echo "INCREMENTAL_VIEW_CONTRIBUTION_COMPLETE batch=${BATCH_ID}"
