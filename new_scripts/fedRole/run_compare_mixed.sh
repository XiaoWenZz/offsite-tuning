#!/bin/bash

# =========================================================
# 0. 环境配置
# =========================================================
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
# 读取环境变量中的 HuggingFace Token，确保有访问权限
export HF_TOKEN="${HF_TOKEN}"
export WANDB_WATCH="false"

# 显式设置 offline 防止网络阻塞 (调试通过后可注释掉)
# export WANDB_MODE=offline 

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate xiaowen
else
    echo "conda not found" >&2
    exit 1
fi

# =========================================================
# 1. 实验变量配置 (针对 ScaleOT 复现优化)
# =========================================================
MODEL="Qwen/Qwen2.5-1.5B" 

# [修改] 数据集改为 ScaleOT 标准 benchmark
DATASET="mixed_piqa_hellaswag" 

CACHE_DIR="/data/xiaowen"

NUM_CLIENTS=10      
NUM_CLUSTERS=2      
ALPHA=0.1           

# [修改] 增加轮数，因为 LoRA 收敛比全量微调慢
ROUNDS=20           

# [修改] 增加本地步数，让 Client 学得更充分
LOCAL_STEPS=10      

# [修改] LoRA 学习率通常比全量微调大 (3e-4 或 5e-4)
LR="3e-4"           

LAYER_BUDGET=6

# [修改] 项目名称区分
export WANDB_PROJECT="fedrole_scaleot_reproduction"

cd ../..
echo "=================================================="
echo "Starting ScaleOT Reproduction (PIQA + HellaSwag)"
echo "Model: $MODEL | Clients: $NUM_CLIENTS"
echo "Strategy: Baseline vs FedRole (with Alignment & LoRA)"
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