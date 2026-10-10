#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BATCH_ID="${BATCH_ID:-protected_parax_seed0_val_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
[[ "${GPU}" == 0 ]] || { echo 'Only GPU0 is authorized' >&2; exit 2; }
export GPU
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/emotic_protected_parax_val}"
LOG_ROOT="${LOG_ROOT:-./logs/emotic_protected_parax_val}"
export OUTPUT_ROOT LOG_ROOT
control="${OUTPUT_ROOT}/${BATCH_ID}/control"
[[ ! -e "${OUTPUT_ROOT}/${BATCH_ID}" ]] || { echo 'Batch already exists' >&2; exit 2; }
mkdir -p "${control}" "${LOG_ROOT}/${BATCH_ID}"
cat > "${control}/manifest.txt" <<EOF
branch=$(git branch --show-current)
commit=$(git rev-parse HEAD)
methods=baseline frozen_pool fresh_only reuse_old
dataset=EMOTIC Track-A; train and validation only; no test
seed=0; tasks=0-2; epochs=30/task; train/eval batch=64; workers=2
optimizer=Adam reset/task; base_lr=0.0125; ParaX_lr=0.0004; weight_decay=0; cosine min0 warmup0
AMP=initial_scale1024 growth_interval1000000000; TF32 unchanged
inputs=${DATA_ROOT:?}; ${CLIP_CHECKPOINT:?}; ${FACE_MANIFEST_ROOT:?}
position=after normalized final Task Forward CLS per lane, before fixed feature fusion; no Frozen Forward insertion
center=6 preallocated rank32 parameter-matrix pairs; task0 trains slots0,1; fresh/reuse allocate2/task
policy=frozen_pool only0,1; fresh_only current2; reuse_old old+current2; old masks/experts/router/projection frozen
router=task-local image-only hidden16; shared across Full/Person/Face; no level embedding
output=task-local zero-initialized rank32 projection; fixed scale1; differentiable actual residual norm bound0.02
fusion=fixed features reliableFace[0.64,0.16,0.20] otherwise[0.8,0.2,0]; current fused BCE+0.1 reliable-view BCE
other=CLIP frozen; Image-token Adapter off; Projector off; distillation off
artifacts=paired hashes and sampler IDs; immutable protected state audit; fixed16 validation images/features/logits/gates; per-view scores; task metrics; training history; compact checkpoints
output=${OUTPUT_ROOT}/${BATCH_ID}; logs=${LOG_ROOT}/${BATCH_ID}
EOF
# Finite resource queue; never stop or modify another user's process.
start=${SECONDS}
while true; do
  free_mib="$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if (( free_mib >= ${MIN_FREE_MIB:-18000} )); then break; fi
  (( SECONDS - start < ${MAX_WAIT_SECONDS:-21600} )) || { echo 'GPU memory wait expired' >&2; exit 2; }
  printf '%s waiting_gpu0_free_mib=%s\n' "$(date -Is)" "${free_mib}" >> "${control}/queue.log"
  sleep 60
done
date -Is > "${control}/started.txt"
methods=(baseline frozen_pool fresh_only reuse_old)
pids=()
for method in "${methods[@]}"; do
  (
    METHOD="${method}" RUN_ID="${BATCH_ID}" EPOCHS=30 UPDATES_PER_TASK=0 \
      bash scripts/emotic/run_protected_parax_validation.sh \
      > "${LOG_ROOT}/${BATCH_ID}/${method}.launcher.log" 2>&1
  ) &
  pids+=("$!")
done
failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    printf '0\n' > "${control}/${methods[index]}.exit_code"
  else
    printf '1\n' > "${control}/${methods[index]}.exit_code"
    failed=1
  fi
done
(( failed == 0 )) || { echo 'One or more runs failed; inspect logs' >&2; exit 1; }
"${PYTHON:-python}" -m multi_lane.track_a.protected_parax_compare "${OUTPUT_ROOT}/${BATCH_ID}" \
  > "${control}/comparison.log" 2>&1
date -Is > "${control}/completed.txt"
