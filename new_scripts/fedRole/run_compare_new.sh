#!/bin/bash

cd ~/offsite-tuning/new_scripts/fedRole
# =========================================================
# 0. 环境配置
# =========================================================
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="${HF_TOKEN}"
export WANDB_WATCH="false"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate xiaowen
else
    echo "conda not found" >&2
    exit 1
fi

# =========================================================
# 1. 实验变量配置
# =========================================================
MODEL="Qwen/Qwen2.5-1.5B" 
DATASET="mixed_piqa_hellaswag" 
CACHE_DIR="/data/xiaowen"

NUM_CLIENTS=10      
NUM_CLUSTERS=2      
ALPHA=0.1           
ROUNDS=20           
LOCAL_STEPS=10      

# [修改] 降低学习率，防止在不平滑的特征空间内发散震荡
LR="1e-4"           

LAYER_BUDGET=6
export WANDB_PROJECT="fedrole_scaleot_reproduction"

cd ../..
echo "=================================================="
echo "Starting ScaleOT Reproduction (PIQA + HellaSwag)"
echo "Model: $MODEL | Clients: $NUM_CLIENTS"
echo "Strategy: Baseline vs FedRole (Global Harmonizer Pool)"
echo "Layer Budget: $LAYER_BUDGET | LR: $LR"
echo "=================================================="

# =========================================================
# 2. 实验 A: Baseline (Uniform + LoRA + Alignment)
# =========================================================
BASELINE_SCRIPT="offsite_tuning/run_cluster_clm_noniid_qwen_new.py" 
EXP_NAME_BASELINE="Baseline_ScaleOT_C${NUM_CLUSTERS}_B${LAYER_BUDGET}"

echo ">>> [1/2] Running Baseline: Uniform Stride"
export WANDB_NAME="$EXP_NAME_BASELINE"

if [ -f "$BASELINE_SCRIPT" ]; then
    python $BASELINE_SCRIPT \
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
else
    echo "Error: Baseline script not found!"
fi

echo ">>> Baseline Finished."
echo "--------------------------------------------------"

# =========================================================
# 3. 实验 B: Ours (FedRole + LoRA + Alignment)
# =========================================================
OURS_SCRIPT="offsite_tuning/run_fedrole_new.py"
EXP_NAME_OURS="Ours_FedRole_ScaleOT_C${NUM_CLUSTERS}_B${LAYER_BUDGET}"

echo ">>> [2/2] Running Ours: FedRole (Dynamic)"
export WANDB_NAME="$EXP_NAME_OURS"

if [ -f "$OURS_SCRIPT" ]; then
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
else
    echo "Error: FedRole script not found!"
fi

echo "=================================================="
echo "All Done."
echo "=================================================="