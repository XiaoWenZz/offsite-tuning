# 设置单卡可见
export CUDA_VISIBLE_DEVICES=0

# 运行全参数微调 (基于 scripts/table1/opt-1.3b/wikitext.sh 修改)
accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    run_clm.py \
    --model_name_or_path facebook/opt-1.3b \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --block_size 512 \
    --output_dir output/opt-1.3b-ft