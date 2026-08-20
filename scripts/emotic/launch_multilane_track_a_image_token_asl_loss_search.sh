#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

GPUS_TEXT="${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_LIST <<< "${GPUS_TEXT}"
[[ "${#GPU_LIST[@]}" -gt 0 ]] || { echo "At least one GPU is required" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "ASL loss search requires a clean Git worktree" >&2; exit 2; }

BATCH_ID="${BATCH_ID:-image_token_asl_loss_seed0_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT}/output/emotic_track_a_image_token_asl_hparam/${BATCH_ID}}"
mkdir -p "${OUTPUT_BASE}"
MANIFEST="${OUTPUT_BASE}/search_manifest.tsv"

labels=(joint_bce)
routings=(joint_bce)
gamma_negs=(9.8)
gamma_poss=(0.0)
clips=(0.05)
for gamma_neg in 1 2 4 6 9.8; do
  for asl_clip in 0 0.025 0.05 0.1; do
    gamma_tag="${gamma_neg//./p}"
    clip_tag="${asl_clip//./p}"
    labels+=("adapter_asl_gn${gamma_tag}_clip${clip_tag}")
    routings+=(adapter_asl)
    gamma_negs+=("${gamma_neg}")
    gamma_poss+=(0.0)
    clips+=("${asl_clip}")
  done
done

printf 'index\tlabel\tloss_routing\tgamma_neg\tgamma_pos\tclip\tgpu\n' > "${MANIFEST}"
for index in "${!labels[@]}"; do
  gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${index}" "${labels[$index]}" "${routings[$index]}" \
    "${gamma_negs[$index]}" "${gamma_poss[$index]}" \
    "${clips[$index]}" "${gpu}" >> "${MANIFEST}"
done

run_gpu_lane() {
  local lane_index="$1"
  local gpu="${GPU_LIST[$lane_index]}"
  local job_index
  for ((job_index=lane_index; job_index<${#labels[@]}; job_index+=${#GPU_LIST[@]})); do
    local label="${labels[$job_index]}"
    local run_id="${BATCH_ID}_${label}"
    echo "Starting ${label} on physical GPU ${gpu}"
    SEED=0 GPU="${gpu}" LOSS_ROUTING="${routings[$job_index]}" \
      ASL_GAMMA_NEG="${gamma_negs[$job_index]}" \
      ASL_GAMMA_POS="${gamma_poss[$job_index]}" \
      ASL_CLIP="${clips[$job_index]}" \
      RUN_ID="${run_id}" OUTPUT_BASE="${OUTPUT_BASE}" \
      bash scripts/emotic/run_multilane_track_a_image_token_hparam_val.sh
  done
}

pids=()
for lane_index in "${!GPU_LIST[@]}"; do
  run_gpu_lane "${lane_index}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
[[ "${failed}" == "0" ]] || { echo "At least one ASL loss-search lane failed" >&2; exit 1; }

summary_arguments=()
for label in "${labels[@]}"; do
  summary_arguments+=(--run "${label}=${OUTPUT_BASE}/${BATCH_ID}_${label}")
done
"${PYTHON:-/opt/conda/envs/ddp/bin/python}" -m multi_lane.track_a.summarize_image_token_asl_search \
  "${summary_arguments[@]}" \
  --output "${OUTPUT_BASE}/loss_search_summary.json"
echo "MULTI_LANE_IMAGE_TOKEN_ASL_LOSS_SEARCH_COMPLETE ${OUTPUT_BASE}"
