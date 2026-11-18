#!/usr/bin/env bash
set -euo pipefail
phase=${1:?supply smoke or formal}
mode=${2:?supply sample_full or sample_holdout}
case "$mode" in sample_full|sample_holdout) ;; *) exit 2 ;; esac
export CUDA_VISIBLE_DEVICES=${3:-0} CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
batch=64
out=output/d9_sample_fusion/seed1_20261118
logs=logs/d9_sample_fusion/seed1_20261118
ref=../multi-lane-main-d9-protected-fixed-20261117/output/d9_protected_fixed/seed1_20261117
args=()
case "$phase" in
    smoke) out="$out/smoke"; logs="$logs/smoke"; ref="$ref/smoke"; batch=4; args+=(--smoke) ;;
    formal) ;;
    *) exit 2 ;;
esac
root="$out/$mode"
mkdir -p "$logs" "$root"
exec 9>"$root/.pipeline.lock"
flock -n 9 || { echo "Pipeline already running for $root" >&2; exit 1; }
exec > >(tee -a "$logs/$mode.log") 2>&1
trap 'status=$?; echo "$status" > "$root/exit.status"' EXIT
echo "phase=$phase mode=$mode gpu=$CUDA_VISIBLE_DEVICES $(git rev-parse HEAD) $(date -Is)"
"$py" -u -m scripts.emotic.run_d9_relation_training --mode "$mode" --protect-branches \
   --output "$root" --precision bf16 --workers 2 --micro-batch-size "$batch" "${args[@]}"
ref_args=()
if [ "$mode" = sample_full ]; then
    "$py" -u -m scripts.emotic.verify_d9_protected_replay --root "$root" --reference-root "$ref/fixed" "${args[@]}"
    ref_args=(--reference-root "$ref/fixed")
fi
for split in val test; do
    "$py" -u -m scripts.emotic.evaluate_d9_sample_fusion --root "$root" --split "$split" \
       --batch-size "$batch" --workers 2 "${args[@]}" "${ref_args[@]}"
done
