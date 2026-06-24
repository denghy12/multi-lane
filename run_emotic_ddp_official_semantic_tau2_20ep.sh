#!/bin/bash

set -euo pipefail

set +u
source ~/.bashrc
set -u
eval "$(conda shell.bash hook)"
conda activate multilane

mkdir -p ./logs ./output

export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

WANDB_API_KEY="${WANDB_API_KEY:-ADD_YOUR_API_KEY_HERE}" \
torchrun --nproc_per_node=1 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=localhost:0 \
  --nnodes=1 \
  main.py emotic \
  --name emotic_ddp_official_semantic_tau2_g07_b5c3_20ep \
  --notes "EMOTIC B5-C3 with official DDP semantic recipe, emotion prompts, PCD tau2 gamma0.7, 20ep" \
  --data_path ./datasets \
  --dataset Split-EMOTIC \
  --emotic_input_mode full \
  --output_dir ./output/emotic_ddp_official_semantic_tau2_g07_b5c3_20ep \
  --num_tasks 8 \
  --base_classes 5 \
  --epochs 20 \
  --batch_size 8 \
  --drop_last \
  --backbone clip_vit_b16_patch \
  --head_mode clip_ddp \
  --ddp_prompt_length 16 \
  --ddp_prompt_layers 5 \
  --ddp_pcd true \
  --ddp_tau_max 2.0 \
  --ddp_gamma 0.7 \
  --ddp_similarity_aggregation paper_attention \
  --ddp_paper_attention_scale 20.0 \
  --ddp_prompt_norm_mode prompted \
  --ddp_logit_scale_mode paper \
  --ddp_train_logit_scale_mode paper \
  --ddp_eval_logit_scale_mode paper \
  --ddp_paper_logit_scale 100.0 \
  --ddp_loss_mode paper_sum \
  --ddp_loss_weight 0.03 \
  --ddp_text_init semantic \
  --ddp_positive_text_template "a photo of a person clearly feeling {}." \
  --ddp_negative_text_template "a photo of a person not feeling {}." \
  --ddp_train_text_prompts true \
  --ddp_train_visual_prompts true \
  --ddp_prompt_polarity both \
  --ddp_class_chunk_size 4 \
  --ddp_optimizer_scope continual \
  --ddp_clear_frozen_optimizer_state true \
  --ddp_optimizer_lr 0.0059 \
  --ddp_scheduler_mode paper_multistep \
  --ddp_scheduler_milestones 0 20 \
  --ddp_scheduler_gamma 0.1 \
  --ddp_train_transform paper \
  --ddp_diagnostics true \
  --ddp_diagnostic_thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --ddp_eval_score_mode logits \
  --ddp_eval_threshold 0.8 \
  --ddp_score_dump true \
  --store_model \
  "$@" \
  2>&1 | tee ./logs/emotic_ddp_official_semantic_tau2_g07_b5c3_20ep.log
