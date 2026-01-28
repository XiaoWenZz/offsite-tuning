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
# MODEL="facebook/opt-125m"
MODEL="Qwen/Qwen2.5-7B"
DATASET="yelp_review_full"
TEXT_COL="text"      # Yelp 的文本列名
LABEL_COL="label"    # Yelp 的标签列名
CACHE_DIR="/data/xiaowen"
NUM_CLUSTERS=5
LR="1e-4"
ALPHA=0.1          # 异构程度

# WandB 配置
export WANDB_PROJECT="offsite_tuning"
# export HF_TOKEN="MY_TOKEN"
# 命名格式：eval_模型_数据集_Cluster实验_Alpha值_K值_学习率
export WANDB_NAME="fedOT_eval_${MODEL##*/}_${DATASET}_a${ALPHA}_lr${LR}"
export WANDB_WATCH="false"

echo "=================================================="
echo "Running Cluster FedOT"
echo "Model: $MODEL | Dataset: $DATASET"
echo "Alpha: $ALPHA | Clusters: $NUM_CLUSTERS"
echo "WandB: $WANDB_PROJECT / $WANDB_NAME"
echo "HF Mirror: $HF_ENDPOINT"
echo "=================================================="

cd ..

# === 2. 运行 Python 脚本 ===
python offsite_tuning/run_clm_noniid_fedOT_qwen.py \
    --model_name $MODEL \
    --dataset_name $DATASET \
    --text_col $TEXT_COL \
    --label_col $LABEL_COL \
    --num_clients 10 \
    --alpha $ALPHA \
    --keep_layers 2 \
    --rounds 50 \
    --local_steps 5 \
    --lr $LR \
    --batch_size 4 \
    --seed 42 \
    --wandb_project $WANDB_PROJECT \
    --cache-dir $CACHE_DIR \
    --wandb_run_name $WANDB_NAME

echo "Done. Check WandB for divergence metrics."