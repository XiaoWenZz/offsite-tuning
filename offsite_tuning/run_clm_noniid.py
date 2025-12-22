import argparse
import logging
import sys
import copy
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from accelerate import Accelerator
from accelerate.logging import get_logger
import wandb  # === [新增] ===

# 设置日志
logger = get_logger(__name__)

# ==========================================
# 工具函数 (Emulator生成 & 回填)
# ==========================================
def create_offsite_emulator(full_model, keep_k=2):
    logger.info(f"Creating Baseline Emulator (Keeping top {keep_k} & bottom {keep_k} layers)...")
    emulator = copy.deepcopy(full_model)
    
    if hasattr(emulator, "model"):
        layers = emulator.model.decoder.layers
    else:
        layers = emulator.decoder.layers
        
    total_layers = len(layers)
    indices_to_keep = list(range(keep_k)) + list(range(total_layers - keep_k, total_layers))
    
    new_layers = nn.ModuleList([layers[i] for i in indices_to_keep])
    
    if hasattr(emulator, "model"):
        emulator.model.decoder.layers = new_layers
        emulator.config.num_hidden_layers = len(new_layers)
    else:
        emulator.decoder.layers = new_layers
        emulator.config.num_hidden_layers = len(new_layers)
    
    # 必须禁用 use_cache，否则被剪枝的模型在前向传播时会因为缓存索引不匹配报错
    emulator.config.use_cache = False
    
    return emulator, indices_to_keep

def plugin_adapter_to_full_model(full_model, adapted_emulator, layer_mapping):
    logger.info("Plugging adapted weights back into Full Model...")
    if hasattr(full_model, "model"):
        full_layers = full_model.model.decoder.layers
    else:
        full_layers = full_model.decoder.layers
        
    if hasattr(adapted_emulator, "model"):
        adapted_layers = adapted_emulator.model.decoder.layers
    else:
        adapted_layers = adapted_emulator.decoder.layers
        
    for emulator_idx, full_model_idx in enumerate(layer_mapping):
        full_layers[full_model_idx].load_state_dict(adapted_layers[emulator_idx].state_dict())
    return full_model

def evaluate_model(model, dataloader, device="cpu"):
    model.eval()
    model.to(device)
    total_loss = 0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            total_loss += outputs.loss.item()
            steps += 1
            if steps >= 20: break 
    return total_loss / steps

# ==========================================
# 参数解析
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run Baseline Offsite-Tuning")
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
    parser.add_argument("--dataset_name", type=str, default="ag_news")
    parser.add_argument("--keep_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="fed_offsite_tuning")
    parser.add_argument("--wandb_run_name", type=str, default="baseline_ot")
    args = parser.parse_args()
    return args

# ==========================================
# 主程序
# ==========================================
def main():
    args = parse_args()
    accelerator = Accelerator()
    set_seed(args.seed)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # === [WandB 初始化] ===
    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )

    logger.info("=== [Baseline OT] Step 1: Initialization ===")
    
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    full_model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)
    
    # 评估基准数据 (Target Label = 1, Sports)
    dataset = datasets.load_dataset(args.dataset_name, split="train[:1000]")
    client_dataset = dataset.filter(lambda x: x['label'] == 1) 
    
    def tokenize_fn(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)
    
    tokenized_ds = client_dataset.map(tokenize_fn, batched=True, remove_columns=['text', 'label'])
    client_loader = DataLoader(tokenized_ds, batch_size=args.batch_size, collate_fn=default_data_collator)
    
    logger.info("Evaluating Original Full Model (Zero-shot)...")
    original_loss = evaluate_model(full_model, client_loader, accelerator.device)
    logger.info(f"Original Full Model Loss: {original_loss:.4f}")
    
    if accelerator.is_main_process:
        wandb.log({"original_loss": original_loss})

    # Step 2: Generate Emulator
    emulator, layer_mapping = create_offsite_emulator(full_model, keep_k=args.keep_layers)
    
    # Step 3: Client Fine-tuning
    logger.info("=== [Baseline OT] Step 3: Client Fine-tuning ===")
    emulator = accelerator.prepare(emulator)
    optimizer = torch.optim.AdamW(emulator.parameters(), lr=args.lr)
    client_loader = accelerator.prepare(client_loader)
    
    emulator.train()
    global_step = 0
    for epoch in range(args.epochs):
        for step, batch in enumerate(client_loader):
            optimizer.zero_grad()
            outputs = emulator(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            
            global_step += 1
            if step % 10 == 0:
                logger.info(f"Epoch {epoch} Step {step}: Loss = {loss.item():.4f}")
                if accelerator.is_main_process:
                    wandb.log({"train_loss": loss.item(), "epoch": epoch, "step": global_step})
    
    # Step 4: Plug-in & Final Eval
    logger.info("=== [Baseline OT] Step 4: Plug-in & Final Eval ===")
    trained_emulator_unwrapped = accelerator.unwrap_model(emulator)
    full_model.to(accelerator.device)
    adapted_full_model = plugin_adapter_to_full_model(full_model, trained_emulator_unwrapped, layer_mapping)
    
    final_loss = evaluate_model(adapted_full_model, client_loader, accelerator.device)
    improvement = original_loss - final_loss
    
    logger.info(f"Final Loss: {final_loss:.4f}, Improvement: {improvement:.4f}")
    
    if accelerator.is_main_process:
        wandb.log({
            "final_loss": final_loss,
            "improvement": improvement
        })
        wandb.finish()

if __name__ == "__main__":
    main()