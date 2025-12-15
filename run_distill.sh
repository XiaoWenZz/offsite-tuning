#!/bin/bash

# Activate target conda env
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate xiaowen
else
    echo "conda not found; please load conda before running." >&2
    exit 1
fi

# 1. 设置模型和参数
MODEL="facebook/opt-1.3b"
num_student_layers=8
pad=2

# 2. 显存与Batch Size 策略 (针对 32GB 显存优化)
# 原脚本总 BS = 18 * 8卡 = 144
# 现设置: 单卡 BS=8, 梯度累积=18 -> 等效总 BS ≈ 144（对齐 opt.sh 全局批但适配单卡）
bs=8
grad_accum=18

# 3. WandB 设置 (如果不需要记录，可以将 report_to 改为 none)
export WANDB_PROJECT="offsite_tuning"
export WANDB_NAME="${MODEL}_emulator_${num_student_layers}_${pad}_${pad}_single_gpu"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0


# 4. 启动命令
# 注意：这里去掉了 --multi_gpu，指定只使用第0号卡
MODEL_PATH="/dataset/opt/$MODEL"
TRAIN_TOKENIZED="/dataset/pile/opt_tokenized/00"
VAL_TOKENIZED="/dataset/opt_tokenized/wikitext-2-raw-v1"

# Prefer local cached model/tokenized data when present; fall back to HF downloads otherwise.
if [ ! -d "$MODEL_PATH" ]; then
    MODEL_PATH="$MODEL"  # use HF hub repo id
fi

DATA_ARGS=( )
if [ -d "$TRAIN_TOKENIZED" ] && [ -d "$VAL_TOKENIZED" ]; then
    DATA_ARGS+=(--train_tokenized_dataset "$TRAIN_TOKENIZED")
    DATA_ARGS+=(--val_tokenized_dataset "$VAL_TOKENIZED")
    DATA_ARGS+=(--preprocessing_num_workers 8)
else
    DATA_ARGS+=(--dataset_name wikitext)
    DATA_ARGS+=(--dataset_config_name wikitext-2-raw-v1)
fi

CUDA_VISIBLE_DEVICES="0" accelerate launch \
    --mixed_precision=bf16 \
    offsite_tuning/run_clm.py \
    --model_name_or_path "$MODEL_PATH" \
    "${DATA_ARGS[@]}" \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --gradient_accumulation_steps $grad_accum \
    --learning_rate 1e-4 \
    --num_warmup_steps 2000 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --lm_weight 1.0 \
    --kd_weight 30.0 \
    --seed 42 \
    --block_size 512 \
    --eval_steps 10 \
    --num_student_layers $num_student_layers \
    --student_l_pad $pad \
    --student_r_pad $pad \
    --output_dir emulators/${MODEL}/${num_student_layers}_${pad}_${pad} \
    --report_to wandb