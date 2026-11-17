#!/usr/bin/env bash
set -euo pipefail
phase=${1:?supply smoke or formal}
export CUDA_VISIBLE_DEVICES=${2:-0}
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=${PYTHON:-python}
root=output/d9_protected_gates/seed1_20261117
logs=logs/d9_protected_gates/seed1_20261117
reference=../multi-lane-main-d9-protected-fixed-20261117/output/d9_protected_fixed/seed1_20261117
batch=64
extra=()
case "$phase" in
    smoke) root="$root/smoke"; reference="$reference/smoke"; logs="$logs/smoke"; batch=4; extra=(--smoke);;
    formal) ;;
    *) exit 2;;
esac
mkdir -p "$logs"
exec > >(tee -a "$logs/multi.log") 2>&1
echo "Phase=$phase, GPU=$CUDA_VISIBLE_DEVICES, git=$(git rev-parse HEAD)"
"$py" -u -m scripts.emotic.run_d9_relation_training --mode multi --protect-branches \
    --output "$root/multi" --micro-batch-size "$batch" --workers 2 --precision bf16 "${extra[@]}"
"$py" -u -m scripts.emotic.verify_d9_protected_replay --root "$root/multi" \
    --reference-root "$reference/fixed" "${extra[@]}"
for split in val test; do
    "$py" -u -m scripts.emotic.evaluate_d9_protected_gates --root "$root/multi" \
        --reference-root "$reference/fixed" --split "$split" --batch-size "$batch" --workers 2 "${extra[@]}"
done
