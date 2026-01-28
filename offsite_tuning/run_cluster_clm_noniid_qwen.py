import argparse
import logging
import sys
import copy
import random
import numpy as np
from sklearn.cluster import KMeans
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
import os

logger = get_logger(__name__)

# ==========================================
# [核心组件] Sacrificial Bridge (最终尝试版)
# ==========================================
class SacrificialBridge(nn.Module):
    def __init__(self, config, target_attention_type="full"):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.attention_type = target_attention_type
        self.layer_idx = 0 
        self.print_once = True 

    def forward(
        self,
        hidden_states,
        *args,
        **kwargs,
    ):
        # [防御] 强力解包输入
        while isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
            
        # 线性变换
        output = self.linear(hidden_states)
        
        # [Log] 记录我们在做什么
        if self.print_once:
            print(f"🔥 [Bridge] Returning RAW TENSOR (Type: {type(output)}) to bypass tuple unpacking issues.")
            self.print_once = False

        # [激进策略] 直接返回 Tensor，不包 Tuple
        # 某些版本的 Transformers 会检查 isinstance(output, tuple)，如果不是则直接使用
        # 如果报错 'Tensor' object is not subscriptable，说明必须包 Tuple，但目前的 Tuple 被双重包了
        return output
    
# ==========================================
# 工具函数
# ==========================================

def get_module_layers(model):
    if hasattr(model, "model"):
        if hasattr(model.model, "layers"): return model.model.layers
        elif hasattr(model.model, "decoder"): return model.model.decoder.layers
    if hasattr(model, "decoder"): return model.decoder.layers
    raise ValueError(f"Unknown model architecture: {type(model)}")

def set_module_layers(model, new_layers):
    if hasattr(model, "model"):
        if hasattr(model.model, "layers"): 
            model.model.layers = new_layers
            model.config.num_hidden_layers = len(new_layers)
            return
        elif hasattr(model.model, "decoder"): 
            model.model.decoder.layers = new_layers
            model.config.num_hidden_layers = len(new_layers)
            return
    if hasattr(model, "decoder"):
        model.decoder.layers = new_layers
        model.config.num_hidden_layers = len(new_layers)
        return
    raise ValueError(f"Unknown model architecture: {type(model)}")

def create_emulator(full_model, budget=4):
    """
    [Baseline] Uniform Stride + Sacrificial Bridge (Auto-Type Detect)
    """
    emulator = copy.deepcopy(full_model)
    layers = get_module_layers(emulator)
    total_layers = len(layers)
    
    if budget >= total_layers: return emulator, list(range(total_layers))
    if budget < 2: budget = 2

    # 1. 选层 (Uniform)
    indices = {0, total_layers - 1}
    remaining_budget = budget - 2
    if remaining_budget > 0:
        if remaining_budget == 1: middle_indices = [total_layers // 2]
        else: middle_indices = np.linspace(1, total_layers - 2, remaining_budget, dtype=int)
        indices.update(middle_indices)
    
    indices_to_keep = sorted(list(indices))
    logger.info(f"✅ [Baseline Strategy] Budget={budget} | Layers: {indices_to_keep}")
    
    # 2. 构建 Layer List
    new_layers_list = [layers[i] for i in indices_to_keep]
    
    # 3. [关键修复] 自动侦测真实层的 attention_type
    ref_layer = layers[0]
    # 尝试获取属性，如果没有则默认 "full"
    detected_type = getattr(ref_layer, "attention_type", "full")
    logger.info(f"🕵️ [Bridge] Detected attention_type from Layer 0: '{detected_type}'")

    # 4. 插入 Bridge
    bridge = SacrificialBridge(full_model.config, target_attention_type=detected_type)
    insert_pos = len(new_layers_list) // 2
    new_layers_list.insert(insert_pos, bridge)
    logger.info(f"🔧 [Bridge] Inserted SacrificialBridge at index {insert_pos}")
    
    set_module_layers(emulator, nn.ModuleList(new_layers_list))
    emulator.config.use_cache = False
    
    return emulator, indices_to_keep

def compute_layer_sensitivity(model, accelerator):
    layers = get_module_layers(model)
    num_layers = len(layers)
    sensitivity_vector = np.zeros(num_layers)
    for i, layer in enumerate(layers):
        grad_sum = 0.0
        for param in layer.parameters():
            if param.grad is not None:
                grad_sum += param.grad.detach().float().norm(2).item()
        sensitivity_vector[i] = grad_sum
    norm = np.linalg.norm(sensitivity_vector)
    if norm > 0: sensitivity_vector = sensitivity_vector / norm
    return sensitivity_vector

def evaluate_model(model, dataloader, accelerator, tokenizer=None):
    model.eval()
    total_loss = 0; steps = 0; MAX_EVAL_STEPS = 10 
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= MAX_EVAL_STEPS: break
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            if tokenizer and tokenizer.pad_token_id is not None: labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(**batch, labels=labels)
            total_loss += outputs.loss.item(); steps += 1
    return total_loss / steps if steps > 0 else 0.0

# ==========================================
# [核心组件] Plug-back Evaluation (Auto-Skip Bridge)
# ==========================================
def evaluate_full_model_plugback(full_model, emulator_state_dict, selected_indices, dataloader, accelerator, tokenizer):
    """
    将 Emulator 参数插回 Full Model。
    自动识别并跳过 Sacrificial Bridge。
    """
    full_layers = get_module_layers(full_model)
    original_weights = {}
    
    # Bridge 所在的 Emulator 索引位置
    bridge_idx_in_emulator = len(selected_indices) // 2
    
    with torch.no_grad():
        current_real_idx_pointer = 0
        
        # Emulator 层数 = 选中的层数 + 1 (Bridge)
        num_emulator_layers = len(selected_indices) + 1
        
        for i in range(num_emulator_layers):
            # [关键] 如果是 Bridge 层，跳过，不回填
            if i == bridge_idx_in_emulator:
                continue
                
            # 找到对应的 Full Model 真实层索引
            real_idx = selected_indices[current_real_idx_pointer]
            current_real_idx_pointer += 1
            
            target_layer = full_layers[real_idx]
            
            # 备份原始参数
            original_weights[real_idx] = {k: v.clone().cpu() for k, v in target_layer.state_dict().items()}
            
            # 提取参数
            prefix_candidates = [f"model.layers.{i}.", f"model.decoder.layers.{i}.", f"decoder.layers.{i}."]
            layer_new_state = {}
            for key, value in emulator_state_dict.items():
                for prefix in prefix_candidates:
                    if key.startswith(prefix):
                        sub_key = key[len(prefix):]
                        layer_new_state[sub_key] = value
                        break
            if layer_new_state: 
                target_layer.load_state_dict(layer_new_state, strict=True)

    # 评估
    full_model.to(accelerator.device); full_model.eval()
    total_loss = 0; steps = 0; MAX_EVAL = 10
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= MAX_EVAL: break
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            if tokenizer and tokenizer.pad_token_id is not None: labels[labels == tokenizer.pad_token_id] = -100
            outputs = full_model(**batch, labels=labels)
            total_loss += outputs.loss.item(); steps += 1
    avg_loss = total_loss / steps if steps > 0 else 0.0
    
    # 还原
    full_model.cpu()
    with torch.no_grad():
        for real_idx, weights in original_weights.items():
            full_layers[real_idx].load_state_dict(weights)
    torch.cuda.empty_cache()
    return avg_loss

class VirtualClient:
    def __init__(self, client_id, train_loader, test_loader, label_dist_str):
        self.id = client_id; self.train_dataloader = train_loader; self.test_dataloader = test_loader; self.label_info = label_dist_str

# ==========================================
# 主程序
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Baseline: Vanilla Clustered FedOT")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset_name", type=str, default="mixed_yelp_gsm8k")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_clusters", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--layer_budget", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="fedrole_benchmark")
    parser.add_argument("--wandb_run_name", type=str, default="baseline_run")
    parser.add_argument("--cache-dir", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args(); set_seed(args.seed)
    accelerator = Accelerator(log_with="wandb")
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    if accelerator.is_main_process:
        accelerator.init_trackers(project_name=args.wandb_project, config=vars(args), init_kwargs={"wandb": {"name": args.wandb_run_name}})

    logger.info(f"=== [Step 1] Loading Full Model: {args.model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    full_model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config, trust_remote_code=True)
    
    # [Baseline] 创建 Uniform Emulator + Bridge
    base_emulator, uniform_indices = create_emulator(full_model, budget=args.layer_budget)
    full_model.cpu() 
    
    logger.info("=== [Baseline] Loading Mixed Datasets (Yelp + GSM8K) ===")
    clients = []
    half_clients = args.num_clients // 2
    
    logger.info("Loading Task A: Yelp Review Full...")
    ds_yelp = datasets.load_dataset("yelp_review_full", cache_dir=args.cache_dir)
    if "train" in ds_yelp: ds_yelp = ds_yelp["train"]
    ds_yelp = ds_yelp.select(range(min(len(ds_yelp), 5000))) 
    
    logger.info("Loading Task B: GSM8K...")
    ds_gsm = datasets.load_dataset("gsm8k", "main", cache_dir=args.cache_dir)
    if "train" in ds_gsm: ds_gsm = ds_gsm["train"]
    ds_gsm = ds_gsm.select(range(min(len(ds_gsm), 5000))) 
    
    def tok_fn(text): return tokenizer(text, padding="max_length", truncation=True, max_length=64)

    logger.info("Creating Heterogeneous Clients...")
    shards_yelp = np.array_split(range(len(ds_yelp)), half_clients)
    for i in range(half_clients):
        sub_ds = ds_yelp.select(shards_yelp[i])
        split_ds = sub_ds.train_test_split(test_size=0.1)
        cols = [c for c in sub_ds.column_names if c not in ['input_ids', 'attention_mask']]
        train_ds = split_ds['train'].map(lambda x: tok_fn(x['text']), batched=True, remove_columns=cols)
        test_ds = split_ds['test'].map(lambda x: tok_fn(x['text']), batched=True, remove_columns=cols)
        clients.append(VirtualClient(i, DataLoader(train_ds, batch_size=4, collate_fn=default_data_collator, shuffle=True), DataLoader(test_ds, batch_size=4, collate_fn=default_data_collator), "Task: Yelp"))

    shards_gsm = np.array_split(range(len(ds_gsm)), args.num_clients - half_clients)
    for i in range(args.num_clients - half_clients):
        cid = half_clients + i
        sub_ds = ds_gsm.select(shards_gsm[i])
        split_ds = sub_ds.train_test_split(test_size=0.1)
        cols = [c for c in sub_ds.column_names if c not in ['input_ids', 'attention_mask']]
        train_ds = split_ds['train'].map(lambda x: tok_fn(x['question']), batched=True, remove_columns=cols)
        test_ds = split_ds['test'].map(lambda x: tok_fn(x['question']), batched=True, remove_columns=cols)
        clients.append(VirtualClient(cid, DataLoader(train_ds, batch_size=4, collate_fn=default_data_collator, shuffle=True), DataLoader(test_ds, batch_size=4, collate_fn=default_data_collator), "Task: GSM8K"))

    logger.info("=== [Step 2] Pilot Round: Sensitivity Profiling ===")
    sensitivity_vectors = []
    full_model.to(accelerator.device); full_model.train()
    initial_full_state_dict = {k: v.clone().detach().cpu() for k, v in full_model.state_dict().items()}
    
    # [优化] Multi-Batch Profiling
    PROFILING_BATCHES = 5
    for client in clients:
        full_model.load_state_dict(initial_full_state_dict)
        full_model.zero_grad()
        
        avg_grad_norm = np.zeros(len(get_module_layers(full_model)))
        valid_batches = 0
        iter_loader = iter(client.train_dataloader)
        
        for _ in range(PROFILING_BATCHES):
            try: batch = next(iter_loader)
            except StopIteration: break
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            if tokenizer.pad_token_id is not None: labels[labels == tokenizer.pad_token_id] = -100
            outputs = full_model(**batch, labels=labels)
            outputs.loss.backward()
            
            grad_norm = compute_layer_sensitivity(full_model, accelerator)
            avg_grad_norm += grad_norm
            valid_batches += 1
            full_model.zero_grad()
        
        if valid_batches > 0: avg_grad_norm /= valid_batches
        sensitivity_vectors.append(avg_grad_norm)
        
    full_model.cpu(); torch.cuda.empty_cache()

    logger.info("Running K-Means...")
    if len(sensitivity_vectors) > 0:
        labels = KMeans(n_clusters=min(args.num_clusters, len(sensitivity_vectors)), random_state=42, n_init=10).fit_predict(np.stack(sensitivity_vectors))
    else: labels = []
    clusters = {i: [] for i in range(args.num_clusters)}
    for i, label in enumerate(labels): clusters[label].append(clients[i])
    accelerator.log({f"cluster_{k}_size": len(v) for k, v in clusters.items()}, step=0)

    logger.info("=== [Step 3] Clustered Federated Training (Fixed Emulator) ===")
    # Baseline: 所有 Cluster 使用相同的 Uniform Emulator
    initial_emu_state = {k: v.clone().detach().cpu() for k, v in base_emulator.state_dict().items()}
    cluster_global_states = {k: copy.deepcopy(initial_emu_state) for k in range(args.num_clusters)}
    training_model = copy.deepcopy(base_emulator)
    
    for round_idx in range(args.rounds):
        logger.info(f"--- Round {round_idx + 1} ---")
        round_metrics = {}
        for cluster_id, cluster_clients in clusters.items():
            if not cluster_clients: continue
            
            training_model.load_state_dict(cluster_global_states[cluster_id])
            training_model.to(accelerator.device); training_model.train()
            optimizer = torch.optim.AdamW(training_model.parameters(), lr=args.lr)
            global_update = {}
            current_round_cpu = {k: v.clone().detach().cpu() for k, v in training_model.state_dict().items()}
            
            # Training
            for client in cluster_clients:
                training_model.load_state_dict(current_round_cpu)
                optimizer = torch.optim.AdamW(training_model.parameters(), lr=args.lr); optimizer.zero_grad()
                for step, batch in enumerate(client.train_dataloader):
                    if step >= args.local_steps: break
                    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                    labels = batch["input_ids"].clone()
                    if tokenizer.pad_token_id is not None: labels[labels == tokenizer.pad_token_id] = -100
                    outputs = training_model(**batch, labels=labels); outputs.loss.backward()
                    optimizer.step(); optimizer.zero_grad()
                
                client_state = training_model.state_dict()
                for key, value in client_state.items():
                    delta = value.detach().cpu() - current_round_cpu[key]
                    global_update[key] = global_update.get(key, 0) + delta

            # Aggregation
            if global_update:
                for key in cluster_global_states[cluster_id]:
                    if key in global_update: cluster_global_states[cluster_id][key] += global_update[key] / len(cluster_clients)
            
            # Plug-back Evaluation
            eval_client = cluster_clients[0] 
            full_loss = evaluate_full_model_plugback(
                full_model, 
                cluster_global_states[cluster_id], 
                uniform_indices, 
                eval_client.test_dataloader, 
                accelerator, 
                tokenizer
            )
            logger.info(f"Cluster {cluster_id} Full Model Loss: {full_loss:.4f}")
            round_metrics[f"cluster_{cluster_id}_full_loss"] = full_loss

        round_metrics["round"] = round_idx + 1
        accelerator.log(round_metrics, step=round_idx + 1)
    accelerator.end_training()

if __name__ == "__main__": main()