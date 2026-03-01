#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="scaleot_complete_reproduction"


CACHE_DIR="/data/xiaowen"

export HF_ENDPOINT=https://hf-mirror.com
# 读取环境变量中的 HuggingFace Token，确保有访问权限

export HF_TOKEN="${HF_TOKEN}"

# 使用 3e-4 的学习率，给 LoRA 足够的动力
# 使用 6 层 (从 28 层压缩到 6 层，压缩率 ~4.6x)

cd ~/offsite-tuning

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate xiaowen
else
    echo "conda not found" >&2
    exit 1
fi


# python scale_ot/run_scaleot_rl_benchmark.py \
#     --model_name "Qwen/Qwen2.5-1.5B" \
#     --layer_budget 6 \
#     --rounds 20 \
#     --local_steps 15 \
#     --lr 3e-4 \
#     --seed 42

python scale_ot/scaleot_exact.py