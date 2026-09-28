#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BATCH_ID="${BATCH_ID:-parax_all_test_rerun_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}"
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_parax_all_test_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_parax_test/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_parax_test/${BATCH_ID}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
[[ ${#GPUS[@]} -ge 5 ]] || { echo "GPU_LIST must contain at least five GPUs" >&2; exit 2; }

mkdir -p "${CONTROL_DIR}/status" "${LOG_DIR}" "${RESULT_BASE}"
cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF_MANIFEST
batch=${BATCH_ID}
protocol=ParaX validation candidates rerun as exploratory held-out-test-only comparisons
dataset=EMOTIC Track-A; train split for all 8 incremental tasks; held-out test per task
seed=0
model=frozen OpenAI CLIP ViT-B/16; shared Selector/Prompt/classifier; existing image-token Adapter disabled
fusion=fixed reliable-Face three-view weights; joint BCE; auxiliary view loss 0.1
schedule=30 epochs/task; batch64 train/eval; Adam reset/task; main lr0.0125; cosine min0; no warmup; wd0; AMP+TF32
parax=rank32; experts3; router_hidden16; ParaX lr0.0004; no checkpoints
test_policy=skip validation evaluation; no test-side threshold/fusion/structure search; no candidate selection from test
suites=level,paired,center,frozen,post
results=${RESULT_BASE}
logs=${LOG_DIR}
EOF_MANIFEST

for gpu in "${GPUS[@]}"; do
  free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free_mib >= 5000 )) || { echo "GPU ${gpu} has only ${free_mib} MiB free; need 5000" >&2; exit 3; }
done

run_suite() {
  local suite="$1"
  local -a methods=()
  case "${suite}" in
    level) methods=(B0 P-post P-10 P-8:10 P-8:10-level Static-control) ;;
    paired) methods=(B0-paired P10-identity P10-small P10-zeroB) ;;
    center) methods=(B0 Shared-live Frozen-center Task-local-delta) ;;
    frozen) methods=(B0 Frozen-center-small Frozen-center-penalty) ;;
    post) methods=(B0 P-post-zero P-post-small P-post-tiny P-post-static) ;;
    *) echo "Unknown suite=${suite}" >&2; return 2 ;;
  esac
  mkdir -p "${CONTROL_DIR}/status/${suite}" "${LOG_DIR}/${suite}"
  local -a pids=()
  for index in "${!methods[@]}"; do
    local method="${methods[$index]}"
    local safe_method="${method//:/_}"
    local gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
    local run_id="${BATCH_ID}_${suite}_${safe_method}_seed0_test"
    (
      SUITE="${suite}" METHOD="${method}" GPU="${gpu}" RUN_ID="${run_id}" \
        OUTPUT_ROOT="${RESULT_BASE}/${suite}" LOG_PATH="${LOG_DIR}/${suite}/${safe_method}.log" \
        bash scripts/emotic/run_multilane_track_a_parax_test_rerun.sh
      printf '0\n' > "${CONTROL_DIR}/status/${suite}/${safe_method}.exit_code"
    ) > "${LOG_DIR}/${suite}/${safe_method}.launcher.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}" || {
      printf '1\n' > "${CONTROL_DIR}/status/${suite}/${methods[$index]//:/_}.exit_code"
      status=1
    }
  done
  (( status == 0 )) || { echo "PARAX_TEST_SUITE_FAILED suite=${suite}" >&2; return 1; }
  printf '%s\n' "${methods[*]} complete" > "${CONTROL_DIR}/status/${suite}/complete.txt"
}

for suite in level paired center frozen post; do
  echo "PARAX_TEST_SUITE_START suite=${suite} batch=${BATCH_ID}"
  run_suite "${suite}"
  echo "PARAX_TEST_SUITE_COMPLETE suite=${suite} batch=${BATCH_ID}"
done

printf '%s\n' 'level paired center frozen post complete' > "${CONTROL_DIR}/status/complete.txt"
echo "PARAX_ALL_TEST_COMPLETE batch=${BATCH_ID} results=${RESULT_BASE}"
