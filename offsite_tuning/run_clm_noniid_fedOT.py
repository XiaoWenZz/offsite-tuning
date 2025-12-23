import argparse
import logging
import sys
import copy
import random
import numpy as np
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
import wandb

logger = get_logger(__name__)

# ==========================================
# 工具函数 (Emulator与数据划分)
# ==========================================
def create_offsite_emulator(full_model, keep_k=2):
    """创建仅保留首尾 keep_k 层的压缩 Emulator"""
    logger.info(f"Creating Global Emulator (Keeping top {keep_k} & bottom {keep_k} layers)...")
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
    
    # 必须禁用 cache
    emulator.config.use_cache = False
    return emulator, indices_to_keep

def partition_data_dirichlet(dataset, num_clients, alpha=0.1, seed=42):
    """模拟 Non-IID 数据分布"""
    np.random.seed(seed)
    # 兼容处理
    try:
        labels = np.array(dataset['label'])
    except:
        labels = np.array(dataset['train']['label'])

    num_classes = len(np.unique(labels))
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    class_indices = [np.argwhere(labels == y).flatten() for y in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    for c_idx, (c_data_indices, fracs) in enumerate(zip(class_indices, label_distribution)):
        fracs = fracs / fracs.sum()
        split_points = (np.cumsum(fracs) * len(c_data_indices)).astype(int)[:-1]
        client_data_split = np.split(c_data_indices, split_points)
        for i, split in enumerate(client_data_split):
            client_indices[i].append(split)
    return [np.concatenate(idcs) for idcs in client_indices]

def evaluate_model(model, dataloader, accelerator):
    model.eval()
    total_loss = 0
    steps = 0
    MAX_EVAL_STEPS = 10 
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= MAX_EVAL_STEPS: break
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            total_loss += loss.item()
            steps += 1
            
    return total_loss / steps if steps > 0 else 0.0

class VirtualClient:
    def __init__(self, client_id, train_loader, test_loader):
        self.id = client_id
        self.train_dataloader = train_loader
        self.test_dataloader = test_loader

# ==========================================
# 参数解析
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run Vanilla Federated Offsite-Tuning (FedOT)")
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
    parser.add_argument("--dataset_name", type=str, default="ag_news")
    parser.add_argument("--num_clients", type=int, default=4)
    # alpha 越小，Non-IID 越严重，Vanilla FedOT 的效果通常越差
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet alpha for Non-IID")
    parser.add_argument("--keep_layers", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="fed_offsite_tuning")
    parser.add_argument("--wandb_run_name", type=str, default="vanilla_fedot")
    return parser.parse_args()

# ==========================================
# 主程序
# ==========================================
def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator()
    
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    logger.info("=== [Step 1] Initialization ===")
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    full_model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)
    
    # 1. 创建唯一的全局 Emulator
    emulator, _ = create_offsite_emulator(full_model, keep_k=args.keep_layers)
    model = accelerator.prepare(emulator)

    # 2. 准备数据与客户端
    dataset = datasets.load_dataset(args.dataset_name, split="train[:2000]")
    client_indices = partition_data_dirichlet(dataset, args.num_clients, alpha=args.alpha)
    
    clients = []
    logger.info("Creating Virtual Clients...")
    for cid, indices in enumerate(client_indices):
        if len(indices) == 0: continue
        sub_ds = dataset.select(indices)
        
        # 简单的 Train/Test 切分
        if len(sub_ds) > 10:
            split_ds = sub_ds.train_test_split(test_size=0.1)
        else:
            split_ds = {'train': sub_ds, 'test': sub_ds}

        def tokenize_fn(examples):
            return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)
        
        tokenized_train = split_ds['train'].map(tokenize_fn, batched=True, remove_columns=['text', 'label'])
        tokenized_test = split_ds['test'].map(tokenize_fn, batched=True, remove_columns=['text', 'label'])
        
        train_loader = DataLoader(tokenized_train, batch_size=args.batch_size, collate_fn=default_data_collator, shuffle=True)
        test_loader = DataLoader(tokenized_test, batch_size=args.batch_size, collate_fn=default_data_collator)
        
        train_loader = accelerator.prepare(train_loader)
        test_loader = accelerator.prepare(test_loader)
        
        clients.append(VirtualClient(cid, train_loader, test_loader))

    # 3. 初始化全局状态 (Global State)
    # 区别点：这里只有一份 Global Weights，而不是 Clustered 版本的字典
    global_emulator_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

    logger.info(f"=== [Step 2] Start Federated Training ({args.rounds} Rounds) ===")
    
    for round_idx in range(args.rounds):
        logger.info(f"--- Round {round_idx + 1} ---")
        
        # FedAvg 累加器
        global_update_accumulator = {}
        participating_clients_count = 0
        
        # --- Client Training Phase ---
        for client in clients:
            # A. 下载：从全局状态加载参数
            model.load_state_dict(global_emulator_state)
            model.train()
            
            # B. 训练：本地 SGD
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            for step, batch in enumerate(client.train_dataloader):
                if step >= args.local_steps: break 
                optimizer.zero_grad()
                outputs = model(**batch, labels=batch["input_ids"])
                accelerator.backward(outputs.loss)
                optimizer.step()
            
            # C. 计算 Delta：本地参数 - 全局参数
            client_final_state = model.state_dict()
            for key, value in client_final_state.items():
                # 必须移到 cpu 计算，避免显存占用
                delta = value.detach().cpu() - global_emulator_state[key]
                if key not in global_update_accumulator:
                    global_update_accumulator[key] = delta
                else:
                    global_update_accumulator[key] += delta
            
            participating_clients_count += 1
            
        # --- Server Aggregation Phase (FedAvg) ---
        if participating_clients_count > 0:
            logger.info("Aggregating updates (FedAvg)...")
            for key in global_emulator_state:
                if key in global_update_accumulator:
                    # 公式：W_global = W_global + (Sum(Deltas) / N)
                    avg_update = global_update_accumulator[key] / participating_clients_count
                    global_emulator_state[key] += avg_update
        
        # --- Evaluation Phase ---
        # 区别点：使用这一份全局模型评估所有 Client
        model.load_state_dict(global_emulator_state)
        test_losses = []
        for client in clients:
            # 这里的 Loss 通常比 Clustered 版本高，因为一个模型要拟合所有人的分布
            loss = evaluate_model(model, client.test_dataloader, accelerator)
            test_losses.append(loss)
        
        avg_loss = np.mean(test_losses)
        logger.info(f"Round {round_idx+1} Global Avg Test Loss: {avg_loss:.4f}")
        
        if accelerator.is_main_process:
            wandb.log({
                "round": round_idx + 1,
                "global_test_loss": avg_loss
            })

    if accelerator.is_main_process:
        wandb.finish()
    logger.info("=== Training Finished ===")

if __name__ == "__main__":
    main()