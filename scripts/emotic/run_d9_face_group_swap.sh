#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
phase=${1:?expected smoke, val or test}
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
src=../multi-lane-main-d9-sample-fusion-20261118/output/d9_sample_fusion/seed1_20261118/sample_holdout
prior=../multi-lane-main-d9-paired-a-eight-20261119/output/d9_paired_a_eight/seed1_20261119/readout
out=output/d9_face_group_swap/seed1_20261119
logs=logs/d9_face_group_swap/seed1_20261119
mkdir -p "$out" "$logs"
exec 9>"$out/.phase.lock"
flock -n 9
exec > >(tee -a "$logs/$phase.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/$phase.exit.status"' EXIT
export CUDA_VISIBLE_DEVICES="" MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=1
echo "phase=$phase git=$(git rev-parse HEAD) start=$(date -Is)"
evaluate=("$py" -u -m scripts.emotic.d9_face_group_swap --source "$src" --prior "$prior" --out "$out")
case "$phase" in
  smoke) "$py" -m unittest discover -s tests -p 'test_d9_face_group_swap.py' -v ;;
  val)
    test "$(cat "$out/smoke.exit.status")" = 0
    "${evaluate[@]}" --split val ;;
  test)
    test "$(cat "$out/val.exit.status")" = 0
    "${evaluate[@]}" --split test ;;
  *) exit 2 ;;
esac
