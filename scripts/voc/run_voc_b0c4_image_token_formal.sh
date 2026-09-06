#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VARIANT="${VARIANT:?Set VARIANT=control or adapter}"
GPU_ID="${GPU_ID:?Set GPU_ID to one CUDA device index}"
BATCH_ID="${BATCH_ID:?Set BATCH_ID for the paired run}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/datasets}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [[ "${VARIANT}" != "control" && "${VARIANT}" != "adapter" ]]; then
  echo "VARIANT must be control or adapter" >&2
  exit 2
fi

OUTPUT_DIR="${ROOT_DIR}/output/voc_b0c4_image_token/${BATCH_ID}/${VARIANT}"
LOG_DIR="${ROOT_DIR}/logs/voc_b0c4_image_token/${BATCH_ID}"
RUN_NAME="voc_b0c4_seed0_${VARIANT}_${BATCH_ID}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

COMMON_ARGS=(
  voc
  --name "${RUN_NAME}"
  --notes "VOC2007 B0-C4 seed0 paired transfer check"
  --dataset Split-VOC
  --data_path "${DATA_ROOT}"
  --model vit_base_patch16_224
  --pretrained true
  --num_tasks 5
  --base_classes 4
  --batch_size 256
  --epochs 30
  --opt adam
  --lr 0.05
  --weight_decay 0
  --sched cosine
  --temperature 1.0
  --num_selectors 10
  --num_prompts 10
  --num_prompt_layers 5
  --prompt_init orthogonal
  --normalize pre-head
  --head_mode concat
  --min_scale 0.05
  --seed 0
  --num_workers "${NUM_WORKERS}"
  --output_dir "${OUTPUT_DIR}"
  --print_freq 10
  --adapter_bottleneck_dim 32
  --adapter_layer_indices 1
  --adapter_residual_scale 0.1
  --adapter_activation relu
  --adapter_learning_rate 0.0004
  --adapter_weight_decay 0
  --adapter_task_init independent
  --asl_gamma_neg 9.8
  --asl_gamma_pos 0
  --asl_clip 0.05
  --asl_eps 1e-8
)

if [[ "${VARIANT}" == "control" ]]; then
  ROUTING_ARGS=(--adapter_mode disabled --loss_routing joint_bce)
else
  ROUTING_ARGS=(--adapter_mode image_token --loss_routing adapter_asl)
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "${ROOT_DIR}"
python main.py \
  "${COMMON_ARGS[@]}" "${ROUTING_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/${VARIANT}.log"
