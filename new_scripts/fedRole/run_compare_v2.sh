#!/bin/bash

# =========================================================
# 0. 环境配置 (Environment Setup)
# =========================================================
cd ~/offsite-tuning/new_scripts/fedRole

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="${HF_TOKEN}" # 确保您的环境变量中有此 Token
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
# 1. 全局超参数配置 (Global Hyperparameters)
# =========================================================
MODEL="Qwen/Qwen2.5-1.5B" 
DATASET="mixed_piqa_hellaswag" 
CACHE_DIR="/data/xiaowen"

NUM_CLIENTS=10      
NUM_CLUSTERS=2      
LAYER_BUDGET=10
ROUNDS=20           
LOCAL_STEPS=10      
LR="1e-4"           

# --- ScaleOT 专属超参数 ---
ALPHA=0.25  # (ScaleOT 论文默认) 被丢弃并替换为 Harmonizer 的比例
BETA=0.8    # (ScaleOT 论文默认) SRC 降秩保留比例 (保留 80% 的秩)

# 设置新的 WandB 项目名称以区分之前的实验
export WANDB_PROJECT="fedrole_vs_scaleot_full"

cd ../..
echo "=================================================="
echo "🚀 Starting Ultimate Benchmark: FedRole vs ScaleOT (Full)"
echo "Model: $MODEL | Clients: $NUM_CLIENTS | Clusters: $NUM_CLUSTERS"
echo "Layer Budget: $LAYER_BUDGET | LR: $LR | Public Data: Aligned"
echo "ScaleOT Params -> Alpha (Harmonizer ratio): $ALPHA, Beta (SRC rank): $BETA"
echo "=================================================="

# =========================================================
# 2. 实验 A: ScaleOT (Full: RL + DL + SRC)
# =========================================================
SCALEOT_SCRIPT="offsite_tuning/run_scaleot_wo_src_v2.py" 
EXP_NAME_SCALEOT="ScaleOT_Full_C${NUM_CLUSTERS}_B${LAYER_BUDGET}"

echo ">>> [1/2] Running Baseline: ScaleOT (Full Architecture)"
export WANDB_NAME="$EXP_NAME_SCALEOT"

if [ -f "$SCALEOT_SCRIPT" ]; then
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
else
    echo "❌ Error: ScaleOT Full script not found at $SCALEOT_SCRIPT !"
fi

echo ">>> ScaleOT (Full) Finished."
echo "--------------------------------------------------"

# =========================================================
# 3. 实验 B: Ours (FedRole: Block-wise + Task-Aware)
# =========================================================
OURS_SCRIPT="offsite_tuning/run_fedrole_v2.py"
EXP_NAME_OURS="Ours_FedRole_C${NUM_CLUSTERS}_B${LAYER_BUDGET}"

echo ">>> [2/2] Running Ours: FedRole (Task-Aware & Block-wise Selection)"
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
    echo "❌ Error: FedRole script not found at $OURS_SCRIPT !"
fi

echo "=================================================="
echo "🎉 All Done. Please check WandB for the Final Table Metrics."
echo "=================================================="