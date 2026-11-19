#!/usr/bin/env bash
set -euo pipefail
phase=${1:?specify smoke or formal}
export CUDA_VISIBLE_DEVICES=${2:-0} CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
cd "$(dirname "$0")/../.."
historical=../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118
run=output/d9_equal_updates/seed1_20261119
logs=logs/d9_equal_updates/seed1_20261119
args=()
case "$phase" in
  smoke) last=1; batch=4; args=(--smoke) ;;
  formal) last=7; batch=64 ;;
  *) exit 2 ;;
esac
model="$run/$phase"
mkdir -p "$model" "$logs"
exec 9>"$model/.pipeline.lock"
flock -n 9 || { echo "Another pipeline owns $model" >&2; exit 1; }
exec > >(tee -a "$logs/$phase.log") 2>&1
status_file="$model/exit.status"
trap 'code=$?; echo "$code" > "$status_file.tmp"; mv "$status_file.tmp" "$status_file"' EXIT
echo "phase=$phase gpu=$CUDA_VISIBLE_DEVICES git=$(git rev-parse HEAD) time=$(date -Is)"
paired=("$py" -u -m scripts.emotic.d9_equal_updates_paired --root "$model" --out "$model/paired_A")
if [[ "$phase" == formal && "$(cat "$run/smoke/exit.status" 2>/dev/null || true)" != 0 ]]; then
  echo "Complete the GPU smoke first." >&2
  exit 1
fi
for ((task=0;task<=last;task++)); do
  "$py" -u -m scripts.emotic.run_d9_relation_training --mode sample_holdout --protect-branches \
    --output "$model" --precision bf16 --micro-batch-size "$batch" --workers 2 \
    --equal-updates-reference "$historical/sample_full" --equal-updates-holdout "$historical/sample_holdout" \
    --stop-after-task "$task" "${args[@]}"
  "${paired[@]}" --phase fit --task "$task"
done
for split in val test; do
  "$py" -u -m scripts.emotic.evaluate_d9_sample_fusion --root "$model" --split "$split" \
    --batch-size "$batch" --workers 2 "${args[@]}"
  if [[ "$phase" == formal ]]; then
    "$py" -u -m scripts.emotic.d9_equal_updates_compare --root "$model" --historical "$historical" \
      --out "$model/comparison" --split "$split"
  fi
  "${paired[@]}" --phase "$split"
done
echo "Finished $phase $(date -Is)"
