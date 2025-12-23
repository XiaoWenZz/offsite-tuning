#!/bin/bash

# === 0. 环境配置 (User Config) ===
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

# conda 环境激活
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate xiaowen
else
    echo "conda not found; please load conda before running." >&2
    exit 1
fi

# === 1. 实验变量定义 ===
MODEL="facebook/opt-125m"
DATASET="ag_news"
LR="2e-4"

# WandB 配置 (使用你指定的格式，但让 dataset 动态化以匹配实际运行)
export WANDB_PROJECT="offsite_tuning"
# ${MODEL##*/} 会把 "facebook/opt-125m" 截取为 "opt-125m"，让名字更短更干净
export WANDB_NAME="eval_${MODEL##*/}_${DATASET}_baseline_lr${LR}"
export WANDB_WATCH="false"

echo "=================================================="
echo "Running Baseline OT"
echo "Model: $MODEL | Dataset: $DATASET"
echo "WandB: $WANDB_PROJECT / $WANDB_NAME"
echo "HF Mirror: $HF_ENDPOINT"
echo "=================================================="

# === 2. 运行 Python 脚本 ===
python offsite_tuning/run_clm_noniid_fedOT.py \
    --model_name $MODEL \
    --dataset_name $DATASET \
    --keep_layers 2 \
    --lr $LR \
    --epochs 3 \
    --batch_size 4 \
    --seed 42 \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_NAME

echo "Done."