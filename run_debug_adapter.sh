#!/bin/bash

# Activate target conda env so accelerate is available
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate xiaowen
else
  echo "conda not found; please load conda before running." >&2
  exit 1
fi

MODEL="facebook/opt-1.3b"
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_PROJECT="offsite_tuning"
export WANDB_NAME="eval_${MODEL}_wikitext_debug_lr2e-5"

num_student_layers=8
bs=1 # Conservative batch size for debugging
pad=2
lr=2e-5 # Reduced from 1e-4 to fix divergence

# Ensure emulator exists or is created (assuming user has it)
# We run the adapter training
echo "Starting adapter training debug with LR=${lr}..."

CUDA_VISIBLE_DEVICES="0" accelerate launch \
  --mixed_precision=bf16 \
  offsite_tuning/run_clm.py \
  --model_name_or_path $MODEL \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_train_batch_size $bs \
  --per_device_eval_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs 20 \
  --num_warmup_steps 100 \
  --lr_scheduler_type cosine \
  --lm_weight 1.0 \
  --kd_weight 0.0 \
  --no_save_model \
  --seed 42 \
  --block_size 512 \
  --eval_steps 50 \
  --num_student_layers $num_student_layers \
  --student_l_pad ${pad} \
  --student_r_pad ${pad} \
  --train_module adapter \
  --restart_training \
  --load_student emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
  --output_dir logs/debug_adapter/${MODEL}/lr_${lr} \
  --report_to wandb
