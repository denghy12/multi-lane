#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:-multi_lane_main_track_a_seed012_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_main_track_a_v0.1}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt}"
GPUS=(0 1 2)

[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }
[[ -f "${CLIP_CHECKPOINT}" ]] || { echo "Missing CLIP checkpoint" >&2; exit 2; }
[[ -f "${DATA_ROOT}/CVPR17_Annotations.mat" ]] || { echo "Missing EMOTIC annotations" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Formal run requires a clean Git worktree" >&2; exit 2; }
mkdir -p "${RUN_ROOT}"

echo "Running GPU smoke on physical GPU ${GPUS[0]}"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" -m multi_lane.track_a.smoke \
  --clip-checkpoint "${CLIP_CHECKPOINT}"

pids=()
for seed in 0 1 2; do
  echo "Starting seed ${seed} on physical GPU ${GPUS[$seed]}"
  SEED="${seed}" GPU="${GPUS[$seed]}" RUN_ROOT="${RUN_ROOT}" \
    PYTHON="${PYTHON}" DATA_ROOT="${DATA_ROOT}" \
    CLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
    bash scripts/emotic/run_multilane_track_a_seed.sh &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
[[ "${failed}" == "0" ]] || { echo "At least one seed failed" >&2; exit 1; }

"${PYTHON}" -m multi_lane.track_a.aggregate \
  --run-root "${RUN_ROOT}" \
  --project-summary "${ROOT}/output/emotic_track_a/${RUN_ID}/formal_seed_summary.json"

echo "MULTI_LANE_TRACK_A_SEED012_COMPLETE ${RUN_ROOT}"
