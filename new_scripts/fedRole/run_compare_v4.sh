#!/bin/bash
# =========================================================
# Ultimate Benchmark v4: Global Top-K Extreme Setup
# Model: Qwen/Qwen2.5-1.5B | Clients: 12 | Clusters: 3 | Budget: 10
# =========================================================
cd ~/offsite-tuning/new_scripts/fedRole
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_WATCH="false"

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xiaowen

MODEL="Qwen/Qwen2.5-1.5B" 
NUM_CLIENTS=12      
NUM_CLUSTERS=3      
LAYER_BUDGET=12   
ROUNDS=20           
LOCAL_STEPS=10      
LR="1e-4"           

export WANDB_PROJECT="fedrole_vs_scaleot_v4"

cd ../..
echo "=================================================="
echo "🚀 Starting Ultimate Benchmark v4 (Global Top-K)"
echo "Model: $MODEL | Clients: $NUM_CLIENTS | Clusters: $NUM_CLUSTERS | Budget: $LAYER_BUDGET"
echo "=================================================="

# --- [1/2] Baseline ---
export WANDB_NAME="ScaleOT_B10_C3_GlobalTopK"
# python offsite_tuning/run_scaleot_wo_src_v4.py \
#     --model_name $MODEL \
#     --num_clients $NUM_CLIENTS \
#     --num_clusters $NUM_CLUSTERS \
#     --layer_budget $LAYER_BUDGET \
#     --rounds $ROUNDS \
#     --local_steps $LOCAL_STEPS \
#     --lr $LR \
#     --batch_size 4 \
#     --seed 42 \
#     --wandb_project $WANDB_PROJECT \
#     --wandb_run_name $WANDB_NAME

# --- [2/2] Ours ---
export WANDB_NAME="Ours_FedRole_B10_C3_GlobalTopK"
python offsite_tuning/run_fedrole_v4.py \
    --model_name $MODEL \
    --num_clients $NUM_CLIENTS \
    --num_clusters $NUM_CLUSTERS \
    --layer_budget $LAYER_BUDGET \
    --rounds $ROUNDS \
    --local_steps $LOCAL_STEPS \
    --lr $LR \
    --batch_size 4 \
    --seed 42 \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_NAME

echo "=================================================="
echo "🎉 All Done. Check WandB for the Global Top-K Battle."
echo "=================================================="