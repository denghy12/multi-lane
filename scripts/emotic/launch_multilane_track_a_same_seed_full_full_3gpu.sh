#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2}"
[[ ${#GPU_LIST[@]} -eq 3 ]] || { echo 'Three GPUs required' >&2; exit 2; }
[[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l)" -eq 3 ]] || exit 2
[[ -z "$(git status --porcelain)" ]] || { echo 'Clean orchestration worktree required' >&2; exit 2; }
BATCH_ID="${BATCH_ID:-same_seed_full_full_test_$(date +%Y%m%d_%H%M%S)}"
CONTROL="$ROOT/output/emotic_track_a_same_seed_full_full/$BATCH_ID"
LOG_DIR="$ROOT/logs/emotic_track_a_same_seed_full_full/$BATCH_ID"
[[ ! -e "$CONTROL" && ! -e "$LOG_DIR" ]] || { echo 'Batch exists; refusing overwrite' >&2; exit 2; }
mkdir -p "$CONTROL/runs" "$LOG_DIR"
trap 'rc=$?; if (( rc != 0 )); then printf "failed_exit_%s\n" "$rc" > "$CONTROL/batch_status.txt"; fi' EXIT
TRAIN0="$ROOT/../multi-lane-main-dual-view-formal-test"
TRAIN12="$ROOT/../multi-lane-main-dual-view-seed12"
[[ "$(git -C "$TRAIN0" rev-parse --short=7 HEAD)" == c9fa74c ]] || exit 2
[[ "$(git -C "$TRAIN12" rev-parse --short=7 HEAD)" == d2442ee ]] || exit 2
for tree in "$TRAIN0" "$TRAIN12"; do
  [[ -z "$(git -C "$tree" status --porcelain)" ]] || { echo "Dirty training source $tree" >&2; exit 2; }
done
SEED0_ID=image_token_layer1_full_person_letterbox_formal_test_seed0_20260903_145816
SEED12_ID=dual_view_locked_formal_seed12_20260903_174938
SEED0_ROOT=/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_formal_v0.1/$SEED0_ID
SEED12_ROOT=/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_dual_view_seed_confirmation_v0.1/$SEED12_ID
FULL=("$SEED0_ROOT/${SEED0_ID}_full_anchor" "$SEED12_ROOT/seed1/${SEED12_ID}_seed1_full_anchor" "$SEED12_ROOT/seed2/${SEED12_ID}_seed2_full_anchor")
PERSON=("$SEED0_ROOT/${SEED0_ID}_person_letterbox" "$SEED12_ROOT/seed1/${SEED12_ID}_seed1_person_letterbox" "$SEED12_ROOT/seed2/${SEED12_ID}_seed2_person_letterbox")
REPEAT=()
for seed in 0 1 2; do REPEAT+=("$CONTROL/runs/${BATCH_ID}_seed${seed}_full_repeat"); done
"$PYTHON" -m unittest discover -s tests > "$LOG_DIR/unit_tests.log" 2>&1
"$PYTHON" -m multi_lane.track_a.compare_same_seed_full_full --audit-only \
  --full-runs "${FULL[@]}" --person-runs "${PERSON[@]}" > "$LOG_DIR/source_audit.log" 2>&1
# Refuse busy cards rather than silently launching competing workloads.
for check in 1 2; do
  for gpu in "${GPU_LIST[@]}"; do
    free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    util="$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    echo "GPU=$gpu free_mib=$free utilization=$util"
    (( free >= 18000 && util <= 10 )) || { echo 'GPU is busy; no training started' >&2; exit 2; }
  done
  if (( check == 1 )); then sleep 3; fi
done
CLIP=/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt
for source_index in 0 1; do
  tree="$TRAIN0"; if (( source_index == 1 )); then tree="$TRAIN12"; fi
  (cd "$tree"; CUDA_VISIBLE_DEVICES="${GPU_LIST[$source_index]}" "$PYTHON" -m multi_lane.track_a.smoke \
    --clip-checkpoint "$CLIP" --adapter-mode image_token --adapter-layer-indices 1 \
    --adapter-bottleneck-dim 32 --adapter-residual-scale 0.1 --adapter-task-init independent \
    --loss-routing adapter_asl) > "$LOG_DIR/source${source_index}_gpu_smoke.log" 2>&1
done
printf 'seed\tgpu\tprimary\trepeat\ttraining_source\n' > "$CONTROL/manifest.tsv"
for seed in 0 1 2; do
  tree="$TRAIN12"; if (( seed == 0 )); then tree="$TRAIN0"; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$seed" "${GPU_LIST[$seed]}" "${FULL[$seed]}" "${REPEAT[$seed]}" "$tree" >> "$CONTROL/manifest.tsv"
done
printf 'training\n' > "$CONTROL/batch_status.txt"
run_repeat() {
  local seed="$1" tree="$TRAIN12" rc=0
  if (( seed == 0 )); then tree="$TRAIN0"; fi
  SEED="$seed" GPU="${GPU_LIST[$seed]}" VIEW=full PYTHON="$PYTHON" \
    RUN_ID="${BATCH_ID}_seed${seed}_full_repeat" OUTPUT_BASE="$CONTROL/runs" LOG_DIR="$LOG_DIR" \
    bash "$tree/scripts/emotic/run_multilane_track_a_dual_view_formal_test.sh" || rc=$?
  printf '%s\n' "$rc" > "$CONTROL/seed${seed}.exit_code"
  return "$rc"
}
pids=()
for seed in 0 1 2; do run_repeat "$seed" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
(( failed == 0 )) || { echo 'A repeat failed; no aggregate published' >&2; exit 1; }
printf 'analyzing\n' > "$CONTROL/batch_status.txt"
"$PYTHON" -m multi_lane.track_a.compare_same_seed_full_full --full-runs "${FULL[@]}" \
  --repeat-runs "${REPEAT[@]}" --person-runs "${PERSON[@]}" --output "$CONTROL/ensemble_comparison.json" \
  > "$LOG_DIR/analysis.log" 2>&1
printf 'complete\n' > "$CONTROL/batch_status.txt"
echo "SAME_SEED_FULL_FULL_BATCH_COMPLETE $CONTROL"
