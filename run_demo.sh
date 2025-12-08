#!/bin/bash

# 尝试初始化 conda
# 这里列出了一些常见的 conda 安装路径，脚本会依次尝试
CONDA_PATHS=(
    "/home/conda/etc/profile.d/conda.sh"
)

CONDA_INITIALIZED=false
for path in "${CONDA_PATHS[@]}"; do
    if [ -f "$path" ]; then
        source "$path"
        CONDA_INITIALIZED=true
        break
    fi
done

# 如果找不到 conda.sh，尝试直接激活（假设已经 source 了 .bashrc/.zshrc）
if [ "$CONDA_INITIALIZED" = false ]; then
    echo "Warning: Could not source conda.sh from common paths. Assuming conda is in PATH."
    eval "$(conda shell.bash hook)" 2>/dev/null || true
fi

# 激活指定环境
conda activate xiaowen

# 检查是否激活成功
if [ "$CONDA_DEFAULT_ENV" != "xiaowen" ]; then
    echo "Error: Failed to activate conda environment 'xiaowen'."
    echo "Current environment: $CONDA_DEFAULT_ENV"
    exit 1
fi

echo "Successfully activated conda environment: $CONDA_DEFAULT_ENV"

# 切换到脚本所在目录（确保在项目根目录运行）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 设置单卡可见
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

echo "Starting training..."

# 运行全参数微调
accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    offsite_tuning/run_clm.py \
    --model_name_or_path facebook/opt-1.3b \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --block_size 512 \
    --output_dir output/opt-1.3b-ft

echo "Training finished."
