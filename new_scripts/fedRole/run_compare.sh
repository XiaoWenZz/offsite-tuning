#!/bin/bash

# =========================================================
# 0. 环境配置 (Environment Setup)
# =========================================================
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

# 显式关闭 WandB 的 git 追踪
export WANDB_WATCH="false"
# 请确认你的 Token 是否有效
# export HF_TOKEN="MY_TOKEN"

# 激活 Conda 环境
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate xiaowen
else
    echo "conda not found; please load conda before running." >&2
    exit 1
fi

# =========================================================
# 1. 实验变量升级 (Upgraded Config)
# =========================================================

# === [升级 1] 模型变大 ===
# 1.5B 具备基础推理能力，且单卡显存占用仅约 4-5G，完全可跑
MODEL="Qwen/Qwen2.5-1.5B" 
# 如果显存非常充裕(24G+)，也可以尝试 "Qwen/Qwen2.5-3B"

DATASET="yelp_review_full"
CACHE_DIR="/data/xiaowen"

# === [升级 2] 规模变大 ===
NUM_CLIENTS=10      # 增加到 10 个客户端
NUM_CLUSTERS=4      # 增加到 4 个聚类 (寻找更细微的异构)
ALPHA=0.1           # 保持高异构性 (越小数据越偏)

# === [升级 3] 训练加长 ===
ROUNDS=50           # 增加到 50 轮，保证收敛
LOCAL_STEPS=10      # 增加本地步数，加快收敛
LR="2e-4"           # 稍微调大一点学习率 (对于 Offsite Tuning 这种冻结大部分层的微调，2e-4 通常更有效)
KEEP_LAYERS=2       # 保留头尾各2层 (Total Budget = 4层)

# 更新 WandB 项目名称，与之前的小实验区分开
export WANDB_PROJECT="fedrole_benchmark_large"

# 回退到项目根目录
cd ../..
echo "Current Directory: $(pwd)"

echo "=================================================="
echo "Starting Large-Scale Benchmark"
echo "Model: $MODEL | Clients: $NUM_CLIENTS | Clusters: $NUM_CLUSTERS"
echo "Rounds: $ROUNDS | Local Steps: $LOCAL_STEPS"
echo "=================================================="

# =========================================================
# 2. 实验 A: Baseline (Vanilla ClusterOT)
# =========================================================
BASELINE_SCRIPT="offsite_tuning/run_cluster_clm_noniid_qwen.py" 
EXP_NAME_BASELINE="Baseline_Fixed_C${NUM_CLUSTERS}_R${ROUNDS}_1.5B"

echo ">>> [1/2] Running Baseline: Vanilla ClusterOT"
export WANDB_NAME="$EXP_NAME_BASELINE"

if [ -f "$BASELINE_SCRIPT" ]; then
    python $BASELINE_SCRIPT \
        --model_name $MODEL \
        --dataset_name $DATASET \
        --num_clients $NUM_CLIENTS \
        --num_clusters $NUM_CLUSTERS \
        --alpha $ALPHA \
        --keep_layers $KEEP_LAYERS \
        --rounds $ROUNDS \
        --local_steps $LOCAL_STEPS \
        --lr $LR \
        --batch_size 4 \
        --seed 42 \
        --wandb_project $WANDB_PROJECT \
        --cache-dir $CACHE_DIR \
        --wandb_run_name $WANDB_NAME
else
    echo "Error: Baseline script '$BASELINE_SCRIPT' not found!"
fi

echo ">>> Baseline Finished."
echo "--------------------------------------------------"

# =========================================================
# 3. 实验 B: Ours (FedRole)
# =========================================================
OURS_SCRIPT="offsite_tuning/run_fedrole.py"
EXP_NAME_OURS="Ours_FedRole_C${NUM_CLUSTERS}_R${ROUNDS}_1.5B"

echo ">>> [2/2] Running Ours: FedRole (Dynamic Routing)"
export WANDB_NAME="$EXP_NAME_OURS"

if [ -f "$OURS_SCRIPT" ]; then
    python $OURS_SCRIPT \
        --model_name $MODEL \
        --dataset_name $DATASET \
        --num_clients $NUM_CLIENTS \
        --num_clusters $NUM_CLUSTERS \
        --alpha $ALPHA \
        --keep_layers $KEEP_LAYERS \
        --rounds $ROUNDS \
        --local_steps $LOCAL_STEPS \
        --lr $LR \
        --batch_size 4 \
        --seed 42 \
        --wandb_project $WANDB_PROJECT \
        --cache-dir $CACHE_DIR \
        --wandb_run_name $WANDB_NAME
else
    echo "Error: FedRole script '$OURS_SCRIPT' not found!"
fi

echo "=================================================="
echo "Benchmark Finished. Check results at WandB."
echo "=================================================="