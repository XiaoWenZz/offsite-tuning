#!/bin/bash

# =========================================================
# 0. 环境配置 (Environment Setup)
# =========================================================
cd ~/offsite-tuning/new_scripts/fedRole

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="${HF_TOKEN}"
export WANDB_WATCH="false"

# 激活 Conda 环境
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate xiaowen
else
    echo "conda not found" >&2
    exit 1
fi

# =========================================================
# 1. 全局超参数配置 (Global Hyperparameters) - 修改为 3 Cluster 极限测试
# =========================================================
MODEL="Qwen/Qwen2.5-1.5B" 
DATASET="mixed_piqa_hellaswag_sciq" 
CACHE_DIR="/data/xiaowen"

NUM_CLIENTS=12      # 修改为12，均分给3个任务
NUM_CLUSTERS=3      # 修改为3个聚类
LAYER_BUDGET=10      
ROUNDS=20           
LOCAL_STEPS=10      
LR="1e-4"           

ALPHA=0.25  
BETA=0.8    

export WANDB_PROJECT="fedrole_vs_scaleot_v3"

cd ../..
echo "=================================================="
echo "🚀 Starting Ultimate Benchmark: 3 Clusters Extreme Setup"
echo "Model: $MODEL | Clients: $NUM_CLIENTS | Clusters: $NUM_CLUSTERS | Budget: $LAYER_BUDGET"
echo "=================================================="

# =========================================================
# 2. 实验 A: ScaleOT Baseline
# =========================================================
SCALEOT_SCRIPT="offsite_tuning/run_scaleot_wo_src_v3.py" 
EXP_NAME_SCALEOT="ScaleOT_B${LAYER_BUDGET}_C${NUM_CLUSTERS}"

echo ">>> [1/2] Running Baseline: ScaleOT (RL + DL without SRC)"
export WANDB_NAME="$EXP_NAME_SCALEOT"

python $SCALEOT_SCRIPT \
    --model_name $MODEL \
    --dataset_name $DATASET \
    --num_clients $NUM_CLIENTS \
    --num_clusters $NUM_CLUSTERS \
    --layer_budget $LAYER_BUDGET \
    --rounds $ROUNDS \
    --local_steps $LOCAL_STEPS \
    --lr $LR \
    --batch_size 4 \
    --seed 42 \
    --wandb_project $WANDB_PROJECT \
    --cache-dir $CACHE_DIR \
    --wandb_run_name $WANDB_NAME

echo "--------------------------------------------------"

# =========================================================
# 3. 实验 B: Ours
# =========================================================
OURS_SCRIPT="offsite_tuning/run_fedrole_v3.py"
EXP_NAME_OURS="Ours_FedRole_B${LAYER_BUDGET}_C${NUM_CLUSTERS}"

echo ">>> [2/2] Running Ours: FedRole (Task-Aware Sensing)"
export WANDB_NAME="$EXP_NAME_OURS"

python $OURS_SCRIPT \
    --model_name $MODEL \
    --dataset_name $DATASET \
    --num_clients $NUM_CLIENTS \
    --num_clusters $NUM_CLUSTERS \
    --alpha $ALPHA \
    --layer_budget $LAYER_BUDGET \
    --rounds $ROUNDS \
    --local_steps $LOCAL_STEPS \
    --lr $LR \
    --batch_size 4 \
    --seed 42 \
    --wandb_project $WANDB_PROJECT \
    --cache-dir $CACHE_DIR \
    --wandb_run_name $WANDB_NAME

echo "=================================================="
echo "🎉 All Done. Please check WandB for the Final Table Metrics."
echo "=================================================="