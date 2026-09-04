#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
SOURCE_BATCH="${SOURCE_BATCH:-learned_reliability_gate_seed012_20260904_021636}"
SOURCE_BASE="${SOURCE_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_learned_reliability_gate_v0.1/${SOURCE_BATCH}/validation_sources}"
RESUME_ID="${RESUME_ID:-learned_gate_split_fix_$(date +%Y%m%d_%H%M%S)}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_learned_reliability_gate/${RESUME_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_learned_reliability_gate/${RESUME_ID}"
SELECTION="${CONTROL_DIR}/gate_selection/validation_selection.json"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5}"
[[ ${#GPU_LIST[@]} -eq 6 ]] || { echo "Six GPUs required for conditional test export" >&2; exit 2; }
[[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l)" -eq 6 ]] || exit 2
[[ -z "$(git status --porcelain)" ]] || { echo "Clean worktree required" >&2; exit 2; }
[[ ! -e "${CONTROL_DIR}" ]] || { echo "Refusing to overwrite existing results" >&2; exit 2; }
full_runs=()
person_runs=()
for seed in 0 1 2; do
  full_runs+=("${SOURCE_BASE}/${SOURCE_BATCH}_seed${seed}_full")
  person_runs+=("${SOURCE_BASE}/${SOURCE_BATCH}_seed${seed}_person")
done
for run in "${full_runs[@]}" "${person_runs[@]}"; do
  [[ -f "${run}/seed_summary.json" ]] || { echo "Missing source: ${run}" >&2; exit 2; }
done
mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}"
trap 'rc=$?; if (( rc != 0 )); then printf "failed_exit_%s\n" "$rc" > "${CONTROL_DIR}/batch_status.txt"; fi' EXIT
git rev-parse HEAD > "${CONTROL_DIR}/analysis_commit.txt"
printf '%s\n' "${full_runs[@]}" "${person_runs[@]}" > "${CONTROL_DIR}/source_runs.txt"
printf 'Selecting original 8 gate candidates; no base training; train/val only.\n'
"${PYTHON}" -m multi_lane.track_a.learned_reliability_gate select \
  --full-runs "${full_runs[@]}" --person-runs "${person_runs[@]}" \
  --data-root "${DATA_ROOT}" --output-dir "${CONTROL_DIR}/gate_selection" \
  2>&1 | tee "${LOG_DIR}/gate_validation_selection.log"
advance="$("${PYTHON}" -c 'import json,sys; print("yes" if json.load(open(sys.argv[1]))["advance_to_locked_test"] else "no")' "${SELECTION}")"
if [[ "${advance}" != yes ]]; then
  printf 'validation_complete_no_eligible_gate_test_skipped\n' > "${CONTROL_DIR}/batch_status.txt"
  echo "LEARNED_GATE_VALIDATION_REJECTED test_not_accessed=true"
  exit 0
fi

# Test is reachable only after the immutable selection artifact is written.
consecutive=0
while (( consecutive < 2 )); do
  ready=1
  for gpu in "${GPU_LIST[@]}"; do
    free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    utilization="$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    if (( free_mib < 18000 || utilization > 10 )); then ready=0; fi
  done
  if (( ready )); then consecutive=$((consecutive + 1)); else consecutive=0; fi
  if (( consecutive < 2 )); then sleep 30; fi
done
run_export() {
  local seed="$1" view="$2" gpu="$3" source="$4" rc=0
  GPU="$gpu" SOURCE_RUN="$source" SELECTION="${SELECTION}" \
    OUTPUT_ROOT="${CONTROL_DIR}/test_exports/seed${seed}_${view}" \
    RUN_ID="${RESUME_ID}_seed${seed}_${view}_test" LOG_DIR="${LOG_DIR}" \
    PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
    bash scripts/emotic/run_multilane_track_a_export_compact_test_scores.sh || rc=$?
  printf '%s\n' "$rc" > "${CONTROL_DIR}/status/test_seed${seed}_${view}.exit_code"
  return "$rc"
}
pids=()
for seed in 0 1 2; do
  run_export "$seed" full "${GPU_LIST[$((seed * 2))]}" "${full_runs[$seed]}" &
  pids+=("$!")
  run_export "$seed" person "${GPU_LIST[$((seed * 2 + 1))]}" "${person_runs[$seed]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then echo "Test export failed; preserving outputs" >&2; exit 1; fi
"${PYTHON}" -m multi_lane.track_a.learned_reliability_gate evaluate-test \
  --selection "${SELECTION}" \
  --full-runs "${CONTROL_DIR}/test_exports/seed0_full" "${CONTROL_DIR}/test_exports/seed1_full" "${CONTROL_DIR}/test_exports/seed2_full" \
  --person-runs "${CONTROL_DIR}/test_exports/seed0_person" "${CONTROL_DIR}/test_exports/seed1_person" "${CONTROL_DIR}/test_exports/seed2_person" \
  --data-root "${DATA_ROOT}" --output "${CONTROL_DIR}/locked_test_evaluation.json" \
  2>&1 | tee "${LOG_DIR}/locked_test_evaluation.log"
printf 'complete\n' > "${CONTROL_DIR}/batch_status.txt"
echo "LEARNED_GATE_RESUME_COMPLETE ${RESUME_ID}"
