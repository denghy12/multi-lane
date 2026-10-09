#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BATCH_ID="${BATCH_ID:?Set a fresh BATCH_ID}"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
TEST_GPU="${TEST_GPU:-1}"
SEEDS="${SEEDS:-0 1 2}"
export AMP_INITIAL_SCALE=1024 AMP_GROWTH_INTERVAL=1000000000
DATA_ROOT="${DATA_ROOT:-../multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT:-../multi-lane-main/output/emotic_face_manifest/face_manifest_audit_v1_20260908}"
TEST_FACE_ROOT="${TEST_FACE_ROOT:-../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909}"
REFERENCE_ROOT=../multi-lane-main-joint26-parax-seed-replication/output/emotic_joint26_parax_view_ablation
REFERENCE_PREFIX=joint26_stableamp_multiseed_val_gpu0_20261009_01
REFERENCE_TEST_ROOT=../multi-lane-main-joint26-checkpoint-locked-test/output/emotic_joint26_locked_test/joint26_final_checkpoint_test_20261009_01
RESULT_ROOT="./output/emotic_joint26_auxiliary_view_contribution/${BATCH_ID}"
LOG_ROOT="./logs/emotic_joint26_auxiliary_view_contribution/${BATCH_ID}"
[[ ! -e "${RESULT_ROOT}" ]] || { echo "Batch already exists" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Tracked modifications present" >&2; exit 2; }
for path in "${DATA_ROOT}/CVPR17_Annotations.mat" "${CLIP_CHECKPOINT}" \
  "${FACE_MANIFEST_ROOT}/manifests/train.jsonl" "${FACE_MANIFEST_ROOT}/manifests/val.jsonl" \
  "${TEST_FACE_ROOT}/manifests/test.jsonl"; do
  [[ -f "${path}" ]] || { echo "Missing input ${path}" >&2; exit 2; }
done
mkdir -p "${RESULT_ROOT}/control" "${LOG_ROOT}"
cat > "${RESULT_ROOT}/control/manifest.txt" <<EOF
batch=${BATCH_ID}; branch=$(git branch --show-current); commit=$(git rev-parse HEAD)
new_models=Full+Person and Full+Face; ParaX/Image-token Adapter disabled; references=reuse existing matched-seed Full and three-view models, never retrain them
dataset=EMOTIC train16001/val2397; test5368 only after final checkpoint; all26 labels; one Task Forward pathway
seed=${SEEDS}; epochs30; batch64/eval64; workers2; Adam main_lr0.0125 WD0 cosine no-warmup; temperature1 threshold0.5; AMP+TF32 initialscale1024 growth1e9
fusion=Full+Person[0.8,0.2]; Full+Face reliable[0.64/0.84,0.20/0.84] else[1,0]; same Face reliability valid/nonambiguous/shortside24/score0.6
loss=BCE(fused)+0.1*mean(BCE(active reliable views)); inactive views never encoded or supervised; same ThreeViewTransform for paired augmentation
parameters=shared frozen CLIP ViT-B/16, shared Selector10/Prompt10 first5 layers/classifier, pre-head normalization
train_GPU=${GPU} two concurrent; test_GPU=${TEST_GPU} two concurrent after each seed; source final epoch30 only; no weight or threshold search; smoke_updates=${SMOKE_UPDATES:-none}
input=${DATA_ROOT}; ${CLIP_CHECKPOINT}; ${FACE_MANIFEST_ROOT}; test_face=${TEST_FACE_ROOT}
references=${REFERENCE_ROOT}/${REFERENCE_PREFIX}_seedN; reference_test=${REFERENCE_TEST_ROOT}/stableamp_seedN
output=${RESULT_ROOT}; logs=${LOG_ROOT}; audits=base_init/first_batch/all_ID/CLIP hashes, zero skipped, frozen parameters, identical val/test rows, checkpoint replay
EOF
for seed in ${SEEDS}; do
  [[ "${seed}" =~ ^[012]$ ]] || { echo "Invalid seed" >&2; exit 2; }
  free="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free >= 14000 )) || { echo "Insufficient free training GPU memory" >&2; exit 2; }
  seed_root="${RESULT_ROOT}/seed${seed}"
  mkdir -p "${seed_root}" "${LOG_ROOT}/seed${seed}"
  pids=()
  for method in FULL_PERSON FULL_FACE; do
    (
      set +e
      METHOD="${method}" GPU="${GPU}" RUN_ID="seed${seed}" PYTHON="${PYTHON}" SEED="${seed}" EPOCHS=30 \
        DATA_ROOT="${DATA_ROOT}" CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" FACE_MANIFEST_ROOT="${FACE_MANIFEST_ROOT}" \
        RESULT_ROOT="${RESULT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
        bash scripts/emotic/run_joint26_parax_view_ablation.sh > "${LOG_ROOT}/seed${seed}/${method}.launcher.log" 2>&1
      result=$?
      printf '%s\n' "${result}" > "${RESULT_ROOT}/control/seed${seed}_${method}.exit_code"
      exit "${result}"
    ) &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || { echo "Training failed" >&2; exit 1; }
  "${PYTHON}" -m multi_lane.track_a.compare_joint26_view_contribution \
    --root "${seed_root}" --reference "${REFERENCE_ROOT}/${REFERENCE_PREFIX}_seed${seed}" \
    --smoke-updates "${SMOKE_UPDATES:-0}"
  if [[ -n "${SMOKE_UPDATES:-}" ]]; then continue; fi
  free="$(nvidia-smi -i "${TEST_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free >= 14000 )) || { echo "Insufficient free test GPU memory; final models preserved" >&2; exit 2; }
  pids=()
  for method in FULL_PERSON FULL_FACE; do
    CUDA_VISIBLE_DEVICES="${TEST_GPU}" "${PYTHON}" -m multi_lane.track_a.evaluate_joint26_checkpoint \
      --source "${seed_root}/${method}" --output "${seed_root}/test/${method}" \
      --data-root "${DATA_ROOT}" --clip-checkpoint "${CLIP_CHECKPOINT}" --face-manifest-root "${TEST_FACE_ROOT}" \
      > "${LOG_ROOT}/seed${seed}/${method}.test.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || { echo "Test evaluation failed" >&2; exit 1; }
  "${PYTHON}" -m multi_lane.track_a.compare_joint26_view_contribution \
    --root "${seed_root}" --reference "${REFERENCE_ROOT}/${REFERENCE_PREFIX}_seed${seed}" \
    --reference-test "${REFERENCE_TEST_ROOT}/stableamp_seed${seed}" --include-test
  date -Is > "${RESULT_ROOT}/control/seed${seed}.complete"
done
if [[ -z "${SMOKE_UPDATES:-}" && "${SEEDS}" == "0 1 2" ]]; then
  "${PYTHON}" -m multi_lane.track_a.compare_joint26_view_contribution --root "${RESULT_ROOT}" --summarize
fi
date -Is > "${RESULT_ROOT}/control/complete"
echo "JOINT26_VIEW_CONTRIBUTION_COMPLETE batch=${BATCH_ID}"
