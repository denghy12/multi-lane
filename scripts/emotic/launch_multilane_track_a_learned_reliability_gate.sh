#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5}"
[[ ${#GPU_LIST[@]} -eq 6 ]] || { echo "Exactly six GPUs are required" >&2; exit 2; }
[[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l)" -eq 6 ]] || { echo "GPUs must be distinct" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Clean Git worktree required" >&2; exit 2; }

BATCH_ID="${BATCH_ID:-learned_reliability_gate_seed012_$(date +%Y%m%d_%H%M%S)}"
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_learned_reliability_gate_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_learned_reliability_gate/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_learned_reliability_gate/${BATCH_ID}"
SOURCE_BASE="${RESULT_BASE}/validation_sources"
TEST_BASE="${RESULT_BASE}/locked_test_exports"
SELECTION_DIR="${CONTROL_DIR}/gate_selection"
SELECTION="${SELECTION_DIR}/validation_selection.json"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"

[[ ! -e "${RESULT_BASE}" ]] || { echo "Result batch already exists: ${RESULT_BASE}" >&2; exit 2; }
[[ ! -e "${CONTROL_DIR}" ]] || { echo "Control batch already exists: ${CONTROL_DIR}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC data: ${DATA_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CHECKPOINT}" >&2; exit 2; }
mkdir -p "${SOURCE_BASE}" "${LOG_DIR}" "${CONTROL_DIR}/status"

printf '%s\n' \
  'Dataset: EMOTIC; seeds: 0,1,2; views: Full and Person letterbox' \
  'Base fitting: deterministic 90% train image groups; calibration: disjoint image-group 10%' \
  'Base model: 30 epochs/task, batch 64, Adam reset/task, main LR 0.0125,' \
  'per-task CosineAnnealingLR eta_min=0, no warmup, CLIP normalization, AMP/TF32 on' \
  'Image-token Adapter: zero-based layer1, bottleneck32, LR4e-4, scale0.1,' \
  'ReLU, independent; main BCE + Adapter ASL gamma_neg9.8/gamma_pos0/clip0.05' \
  'Gate: shared MLP; bbox geometry plus two-view confidence, entropy and disagreement;' \
  'one weight/sample in [0.10,0.35], anchor0.20; hidden={8,16},' \
  'normalized prior={0.1,0.3,1,3}; AdamW1e-3/wd1e-4, 80 epochs/task, batch64,' \
  'current calibration task only, no replay' \
  'Selection: 8-task validation over all seeds; tie/beat fixed0.20 final mAP per seed,' \
  'then rank mean final mAP and average mAP.' \
  'Test: skipped if no eligible winner; otherwise one locked gate, no test search.' \
  'Artifacts: val/calibration probabilities, compact method states (frozen CLIP omitted),' \
  'selection JSON, locked test probabilities and three-seed summary.' \
  > "${CONTROL_DIR}/protocol.txt"

echo -e "seed\tview\tgpu\trun_id" > "${CONTROL_DIR}/manifest.tsv"
source_runs=()
index=0
for seed in 0 1 2; do
  for view in full person; do
    run_id="${BATCH_ID}_seed${seed}_${view}"
    source_runs+=("${SOURCE_BASE}/${run_id}")
    echo -e "${seed}\t${view}\t${GPU_LIST[$index]}\t${run_id}" >> "${CONTROL_DIR}/manifest.tsv"
    index=$((index + 1))
  done
done

consecutive=0
while (( consecutive < 2 )); do
  ready=1
  for gpu in "${GPU_LIST[@]}"; do
    free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    echo "GPU ${gpu} free_mib=${free_mib} utilization=${utilization}"
    if (( free_mib < 18000 || utilization > 10 )); then ready=0; fi
  done
  if (( ready )); then consecutive=$((consecutive + 1)); else consecutive=0; fi
  if (( consecutive < 2 )); then sleep 30; fi
done

run_source() {
  local seed="$1" view="$2" gpu="$3" run_id="$4" rc=0
  SEED="${seed}" VIEW="${view}" GPU="${gpu}" RUN_ID="${run_id}" \
    OUTPUT_BASE="${SOURCE_BASE}" LOG_DIR="${LOG_DIR}" PYTHON="${PYTHON}" \
    DATA_ROOT="${DATA_ROOT}" CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
    bash scripts/emotic/run_multilane_track_a_reliability_source_val.sh || rc=$?
  printf '%s\n' "${rc}" > "${CONTROL_DIR}/status/source_seed${seed}_${view}.exit_code"
  return "${rc}"
}

pids=()
index=0
for seed in 0 1 2; do
  for view in full person; do
    run_source "${seed}" "${view}" "${GPU_LIST[$index]}" "${BATCH_ID}_seed${seed}_${view}" &
    pids+=("$!")
    index=$((index + 1))
  done
done
failed=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
if (( failed )); then echo "At least one validation source failed; outputs preserved" >&2; exit 1; fi

full_runs=("${source_runs[0]}" "${source_runs[2]}" "${source_runs[4]}")
person_runs=("${source_runs[1]}" "${source_runs[3]}" "${source_runs[5]}")
"${PYTHON}" -m multi_lane.track_a.learned_reliability_gate select \
  --full-runs "${full_runs[@]}" \
  --person-runs "${person_runs[@]}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${SELECTION_DIR}" \
  2>&1 | tee "${LOG_DIR}/gate_validation_selection.log"

advance="$("${PYTHON}" -c 'import json,sys; print("yes" if json.load(open(sys.argv[1]))["advance_to_locked_test"] else "no")' "${SELECTION}")"
if [[ "${advance}" != yes ]]; then
  printf 'validation_complete_no_eligible_gate_test_skipped\n' > "${CONTROL_DIR}/batch_status.txt"
  echo "LEARNED_GATE_VALIDATION_REJECTED test_not_accessed=true batch=${BATCH_ID} selection=${SELECTION}"
  exit 0
fi

mkdir -p "${TEST_BASE}"
run_export() {
  local seed="$1" view="$2" gpu="$3" source_run="$4" rc=0
  local run_id="${BATCH_ID}_seed${seed}_${view}_locked_test"
  local output_root="${TEST_BASE}/${run_id}"
  GPU="${gpu}" SOURCE_RUN="${source_run}" SELECTION="${SELECTION}" \
    OUTPUT_ROOT="${output_root}" RUN_ID="${run_id}" LOG_DIR="${LOG_DIR}" \
    PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
    bash scripts/emotic/run_multilane_track_a_export_compact_test_scores.sh || rc=$?
  printf '%s\n' "${rc}" > "${CONTROL_DIR}/status/test_seed${seed}_${view}.exit_code"
  return "${rc}"
}

pids=()
index=0
test_runs=()
for seed in 0 1 2; do
  for view in full person; do
    test_run="${TEST_BASE}/${BATCH_ID}_seed${seed}_${view}_locked_test"
    test_runs+=("${test_run}")
    run_export "${seed}" "${view}" "${GPU_LIST[$index]}" "${source_runs[$index]}" &
    pids+=("$!")
    index=$((index + 1))
  done
done
failed=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
if (( failed )); then echo "At least one locked test export failed; outputs preserved" >&2; exit 1; fi

"${PYTHON}" -m multi_lane.track_a.learned_reliability_gate evaluate-test \
  --selection "${SELECTION}" \
  --full-runs "${test_runs[0]}" "${test_runs[2]}" "${test_runs[4]}" \
  --person-runs "${test_runs[1]}" "${test_runs[3]}" "${test_runs[5]}" \
  --data-root "${DATA_ROOT}" \
  --output "${CONTROL_DIR}/locked_test_evaluation.json" \
  2>&1 | tee "${LOG_DIR}/locked_test_evaluation.log"
printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "LEARNED_RELIABILITY_GATE_COMPLETE batch=${BATCH_ID} result=${RESULT_BASE} control=${CONTROL_DIR}"
