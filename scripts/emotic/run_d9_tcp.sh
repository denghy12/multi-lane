#!/usr/bin/env bash
set -euo pipefail
phase=${1:?use diagnostic-val, diagnostic-test, smoke, fit-val, or test}
py=${PYTHON:-/opt/conda/envs/ddp/bin/python}
caran=${D9_CARAN:-../multi-lane-main-d9-caran-eight-20261119/output/d9_caran_eight/seed1_20261119}
prior=${D9_TASK10:-../multi-lane-main-d9-task10-probe-20261118/output/d9_task10_probe/seed1_20261118}
out=output/d9_tcp/seed1_20261119
logs=logs/d9_tcp/seed1_20261119
mkdir -p "$out" "$logs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=1
exec 9>"$out/.pipeline.lock"
flock -n 9 || exit 1
exec > >(tee -a "$logs/$phase.log") 2>&1
trap 'status=$?; echo "$status" > "$out/$phase.exit.status"' EXIT
echo "phase=$phase git=$(git rev-parse HEAD) start=$(date -Is)"
case "$phase" in
    diagnostic-val|diagnostic-test)
        "$py" -u -m scripts.emotic.d9_tcp_diagnostic --caran "$caran" --out "$out/diagnostic" --split "${phase#diagnostic-}" ;;
    smoke)
        "$py" -m unittest discover -s tests -p 'test_d9_tcp*.py' -v
        "$py" -u -m scripts.emotic.d9_tcp_probe --prior "$prior" --out "$out/smoke" --phase fit-val --tasks 6 --steps 3 --permutations 2 ;;
    fit-val|test)
        test "$(cat "$out/diagnostic-val.exit.status")" = 0
        test "$(cat "$out/smoke.exit.status")" = 0
        if [[ "$phase" == test ]]; then test "$(cat "$out/fit-val.exit.status")" = 0; fi
        if [[ "$phase" == fit-val ]]; then splits=(train val); else splits=(test); fi
        "$py" -u -m scripts.emotic.d9_tcp_preflight --prior "$prior" --caran "$caran" --splits "${splits[@]}"
        "$py" -u -m scripts.emotic.d9_tcp_probe --prior "$prior" --out "$out/probe" --phase "$phase" ;;
    *) exit 2 ;;
esac
