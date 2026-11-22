#!/usr/bin/env bash
set -euo pipefail
phase=${1:?smoke or formal}
coverage=${2:?equal90 or full}
export CUDA_VISIBLE_DEVICES=${3:-0} CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
cd "$(dirname "$0")/../.."
historical=${HISTORICAL_SAMPLE_ROOT:-../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118}
equal=${HISTORICAL_EQUAL_ROOT:-../multi-lane-main-d9-holdout-equal-updates/output/d9_equal_updates/seed1_20261119}
base=${FILL_OUTPUT_ROOT:-output/d9_fill_fusion_tables/seed1_20261122}
logs=${FILL_LOG_ROOT:-logs/d9_fill_fusion_tables/seed1_20261122}
args=(); batch=64
case "$phase" in
  smoke) args+=(--smoke); batch=4 ;;
  formal) ;;
  *) exit 2 ;;
esac
case "$coverage" in
  equal90)
    mode=sample_holdout; gates=all; reference="$equal/$phase"
    args+=(--equal-updates-reference "$historical/sample_full" --equal-updates-holdout "$historical/sample_holdout") ;;
  full)
    mode=sample_full; gates=cosine; reference="$historical/sample_full"
    [[ "$phase" != smoke ]] || reference="$historical/smoke/sample_full" ;;
  *) exit 2 ;;
esac
root="$base/$phase/replay_$coverage"
mkdir -p "$root" "$logs/$phase"
exec 9>"$root/.pipeline.lock"
flock -n 9 || { echo "Another pipeline owns $root" >&2; exit 1; }
exec > >(tee -a "$logs/$phase/replay_$coverage.log") 2>&1
trap 'status=$?; echo "$status" > "$root/exit.status.tmp"; mv "$root/exit.status.tmp" "$root/exit.status"' EXIT
[[ -f "$reference/compact_checkpoints/task0.pth" ]] || { echo "Missing exact replay reference $reference"; exit 1; }
if [[ "$phase" == formal && "$(cat "$base/smoke/replay_$coverage/exit.status" 2>/dev/null || true)" != 0 ]]; then
  echo "Complete $coverage smoke first" >&2; exit 1
fi
echo "phase=$phase coverage=$coverage gpu=$CUDA_VISIBLE_DEVICES git=$(git rev-parse HEAD) time=$(date -Is)"
"$py" -u -m scripts.emotic.run_d9_relation_training --mode "$mode" --protect-branches \
  --output "$root" --precision bf16 --workers 2 --micro-batch-size "$batch" \
  --fill-table-gates "$gates" --fill-replay-reference "$reference" "${args[@]}"
eval_args=(); [[ "$phase" != smoke ]] || eval_args+=(--smoke)
for split in val test; do
  "$py" -u -m scripts.emotic.evaluate_d9_sample_fusion --root "$root" --split "$split" \
    --batch-size "$batch" --workers 2 "${eval_args[@]}"
  "$py" -u -m scripts.emotic.d9_fill_replay --root "$root" --reference "$reference" --split "$split" "${eval_args[@]}"
done
echo "Finished $coverage/$phase $(date -Is)"
