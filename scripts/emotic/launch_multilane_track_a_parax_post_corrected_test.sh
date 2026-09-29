#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BATCH_ID="${BATCH_ID:-parax_post_corrected_test_20260929_01}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RESULT_BASE="${RESULT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_parax_all_test_v0.1/${BATCH_ID}}"
CONTROL_DIR="${ROOT}/output/emotic_track_a_parax_test/${BATCH_ID}"
LOG_DIR="${ROOT}/logs/emotic_track_a_parax_test/${BATCH_ID}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
METHODS=(B0 P-post-zero P-post-small P-post-tiny P-post-static)
[[ ${#GPUS[@]} -eq ${#METHODS[@]} ]] || { echo "GPU_LIST must contain five GPUs" >&2; exit 2; }
[[ ! -e "${CONTROL_DIR}" && ! -e "${RESULT_BASE}" ]] || { echo "Batch output already exists" >&2; exit 2; }

mkdir -p "${CONTROL_DIR}/status/post" "${LOG_DIR}/post" "${RESULT_BASE}/post"
cat > "${CONTROL_DIR}/experiment_manifest.txt" <<EOF_MANIFEST
batch=${BATCH_ID}
code_commit=$(git rev-parse HEAD)
dataset=EMOTIC Track-A; train split for all 8 incremental tasks; held-out test after each task
seed=0
model=frozen OpenAI CLIP ViT-B/16; shared Selector/Prompt/classifier; image-token Adapter disabled
fusion=fixed reliable-Face three-view weights; joint BCE; auxiliary view loss 0.1
schedule=30 epochs/task; batch64 train/eval; Adam reset/task; main lr0.0125; cosine min0; no warmup; wd0; AMP+TF32
post_parax=rank32; experts3; router_hidden16; zero_output; router-only; fixed output scale; freeze center after task0; ParaX lr0.0004
methods=B0 P-post-zero(scale0) P-post-small(scale0.001) P-post-tiny(scale0.0001) P-post-static(scale0.001)
inputs=./datasets/EMOTIC; OpenAI CLIP ViT-B/16 checkpoint; fixed Face train/val/test manifest
test_policy=skip validation evaluation; no test-side threshold/fusion/structure search; no checkpoints
results=${RESULT_BASE}
logs=${LOG_DIR}
EOF_MANIFEST

echo "waiting for GPUs ${GPU_LIST} to each have ${MIN_FREE_MIB} MiB free" | tee "${CONTROL_DIR}/status/post/waiting.txt"
while true; do
  ready=1
  for gpu in "${GPUS[@]}"; do
    free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if (( free_mib < MIN_FREE_MIB )); then ready=0; fi
  done
  (( ready == 1 )) && break
  sleep "${POLL_SECONDS}"
done
rm "${CONTROL_DIR}/status/post/waiting.txt"
date -Is > "${CONTROL_DIR}/status/post/started.txt"

pids=()
for index in "${!METHODS[@]}"; do
  method="${METHODS[$index]}"
  gpu="${GPUS[$index]}"
  run_id="${BATCH_ID}_post_${method}_seed0_test"
  (
    SUITE=post METHOD="${method}" GPU="${gpu}" RUN_ID="${run_id}" \
      OUTPUT_ROOT="${RESULT_BASE}/post" LOG_PATH="${LOG_DIR}/post/${method}.log" \
      bash scripts/emotic/run_multilane_track_a_parax_test_rerun.sh
    printf '0\n' > "${CONTROL_DIR}/status/post/${method}.exit_code"
  ) > "${LOG_DIR}/post/${method}.launcher.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    printf '1\n' > "${CONTROL_DIR}/status/post/${METHODS[$index]}.exit_code"
    status=1
  fi
done
(( status == 0 )) || { echo "PARAX_POST_CORRECTED_TEST_FAILED" >&2; exit 1; }
printf '%s\n' "${METHODS[*]} complete" > "${CONTROL_DIR}/status/post/complete.txt"
echo "PARAX_POST_CORRECTED_TEST_COMPLETE batch=${BATCH_ID} results=${RESULT_BASE}"
