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
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

logger = get_logger(__name__)

# [Harmonizer Identical to FedRole]
class Harmonizer(nn.Module):
    def __init__(self, config, rank=128, target_attention_type="full"):
        super().__init__()
        self.input_dim = config.hidden_size
        self.rank = rank
        self.down_proj = nn.Linear(self.input_dim, self.rank)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(self.rank, self.input_dim)
        nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.zeros_(self.down_proj.bias)
        self.attention_type = target_attention_type
        self.layer_idx = 0 
        self.original_layer_idx = None

    def forward(self, hidden_states, *args, **kwargs):
        x = hidden_states
        while isinstance(x, tuple): x = x[0]
        if not isinstance(x, torch.Tensor): return x
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return x + residual 

# [Helpers Identical to FedRole]
def get_module_layers(model):
    obj = model
    if hasattr(obj, "base_model"): obj = obj.base_model
    if hasattr(obj, "model"): obj = obj.model
    if hasattr(obj, "model") and hasattr(obj.model, "layers"): return obj.model.layers
    if hasattr(obj, "layers"): return obj.layers
    if hasattr(obj, "decoder") and hasattr(obj.decoder, "layers"): return obj.decoder.layers
    raise ValueError(f"Could not find layers in {type(model)}")

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

def get_trainable_keys(model):
    keys = []
    for name, param in model.named_parameters():
        if param.requires_grad: keys.append(name)
    return set(keys)

def create_emulator(full_model, budget=4):
    """Baseline Emulator Construction"""
    emulator = copy.deepcopy(full_model)
    layers = get_module_layers(emulator)
    total = len(layers)
    if budget >= total: return emulator
    indices = {0, total-1}
    rem = budget - 2
    if rem > 0:
        if rem == 1: indices.add(total//2)
        else: indices.update(np.linspace(1, total-2, rem, dtype=int))
    adapter_indices = sorted(list(indices))
    
    new_layers = []
    curr = 0
    ref_type = getattr(layers[0], "attention_type", "full")
    while curr < total:
        if curr in adapter_indices:
            l = layers[curr]
            l.original_layer_idx = curr
            new_layers.append(l)
            curr += 1
        else:
            gap = 0
            while (curr + gap < total) and (curr + gap not in adapter_indices): gap += 1
            new_layers.append(Harmonizer(full_model.config, 128, ref_type))
            curr += gap
            
    set_module_layers(emulator, nn.ModuleList(new_layers))
    emulator.config.use_cache = False
    
    # [Config Map]
    layer_map = {}
    for i, layer in enumerate(new_layers):
        if hasattr(layer, "original_layer_idx") and layer.original_layer_idx is not None:
            layer_map[i] = layer.original_layer_idx
    emulator.config.layer_map = layer_map
    
    # [LoRA]
    for param in emulator.parameters(): param.requires_grad = False
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    emulator = get_peft_model(emulator, peft_config)
    if hasattr(emulator, "config"): emulator.config.layer_map = layer_map
    
    for name, param in emulator.named_parameters():
        if "Harmonizer" in name or "lora_" in name: param.requires_grad = True
        else: param.requires_grad = False
    return emulator

def evaluate_full_model_plugback(full_model, emulator_model, dataloader, accelerator, tokenizer):
    """
    Safe LoRA-aware Plug-back Evaluation.
    Uses a COPY of full_model to prevent contaminating the base model with adapters.
    """
    # 1. 提取 Emulator 的 LoRA 权重 & 建立映射
    emu_state = emulator_model.state_dict()
    adapter_lora_state = {}
    
    # 获取层号映射表
    idx_map = {}
    # Check wrapper config first
    if hasattr(emulator_model, "config") and hasattr(emulator_model.config, "layer_map"):
        idx_map = emulator_model.config.layer_map
    elif hasattr(emulator_model, "base_model") and hasattr(emulator_model.base_model.model, "config"):
        if hasattr(emulator_model.base_model.model.config, "layer_map"):
            idx_map = emulator_model.base_model.model.config.layer_map
            
    # 重映射 LoRA Key
    mapped_count = 0
    for k, v in emu_state.items():
        if "lora_" in k:
            parts = k.split('.')
            try:
                if 'layers' in parts:
                    layer_kw_idx = parts.index('layers')
                    emu_layer_idx = int(parts[layer_kw_idx + 1])
                    
                    if emu_layer_idx in idx_map:
                        real_idx = idx_map[emu_layer_idx]
                        parts[layer_kw_idx + 1] = str(real_idx)
                        new_key = ".".join(parts)
                        adapter_lora_state[new_key] = v.cpu() # 确保在 CPU
                        mapped_count += 1
            except: continue

    # [Debug Print]
    if not hasattr(evaluate_full_model_plugback, "debug_printed"):
        print(f"🔥 Plug-back: Mapped {mapped_count} keys to Full Model.")
        evaluate_full_model_plugback.debug_printed = True

    # 2. [CRITICAL FIX] 创建 Full Model 的临时副本
    # 我们不能直接修改 full_model，因为 get_peft_model 是 In-place 操作
    # Qwen-1.5B 复制一份约为 3GB 内存，这比污染基座模型要安全得多
    temp_full_model = copy.deepcopy(full_model)
    
    # 3. 挂载 LoRA 到副本上
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=8, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Wrap copy
    peft_full_model = get_peft_model(temp_full_model, peft_config)
        
    # 加载重映射后的权重 (Strict=False 是必须的，因为我们只有部分层的 LoRA)
    peft_full_model.load_state_dict(adapter_lora_state, strict=False)
    
    # 移动到 GPU 进行评估
    peft_full_model.to(accelerator.device)
    peft_full_model.eval()
    
    total_loss = 0; steps = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10: break
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            # 清理冲突键
            batch.pop("labels", None); batch.pop("label", None)
            
            labels = batch["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            outputs = peft_full_model(**batch, labels=labels)
            total_loss += outputs.loss.item(); steps += 1
            
    # 4. 销毁副本，释放显存
    del peft_full_model
    del temp_full_model
    torch.cuda.empty_cache()
    
    return total_loss / steps if steps > 0 else 0.0

def compute_layer_sensitivity(model, accelerator):
    layers = get_module_layers(model)
    num_layers = len(layers)
    sensitivity_vector = np.zeros(num_layers)
    for i, layer in enumerate(layers):
        grad_sum = 0.0
        for param in layer.parameters():
            if param.grad is not None: grad_sum += param.grad.detach().float().norm(2).item()
        sensitivity_vector[i] = grad_sum
    norm = np.linalg.norm(sensitivity_vector)
    if norm > 0: sensitivity_vector = sensitivity_vector / norm
    return sensitivity_vector

class VirtualClient:
    def __init__(self, client_id, train_loader, test_loader, label_dist_str):
        self.id = client_id; self.train_dataloader = train_loader; self.test_dataloader = test_loader; self.label_info = label_dist_str

def parse_args():
    parser = argparse.ArgumentParser()
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    full_model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config, trust_remote_code=True)
    
    base_emulator = create_emulator(full_model, budget=args.layer_budget)
    full_model.cpu() 
    
    clients = []
    half_clients = args.num_clients // 2
    ds_yelp = datasets.load_dataset("yelp_review_full", cache_dir=args.cache_dir)
    if "train" in ds_yelp: ds_yelp = ds_yelp["train"]
    ds_yelp = ds_yelp.select(range(min(len(ds_yelp), 5000))) 
    ds_gsm = datasets.load_dataset("gsm8k", "main", cache_dir=args.cache_dir)
    if "train" in ds_gsm: ds_gsm = ds_gsm["train"]
    ds_gsm = ds_gsm.select(range(min(len(ds_gsm), 5000))) 
    def tok_fn(text): return tokenizer(text, padding="max_length", truncation=True, max_length=64)
    for i in range(half_clients):
        sub = ds_yelp.select(np.array_split(range(len(ds_yelp)), half_clients)[i])
        split = sub.train_test_split(test_size=0.1)
        train_ds = split['train'].map(lambda x: tok_fn(x['text']), batched=True)
        test_ds = split['test'].map(lambda x: tok_fn(x['text']), batched=True)
        clients.append(VirtualClient(i, DataLoader(train_ds, batch_size=4, collate_fn=default_data_collator, shuffle=True), DataLoader(test_ds, batch_size=4, collate_fn=default_data_collator), "Yelp"))
    for i in range(args.num_clients - half_clients):
        cid = half_clients + i
        sub = ds_gsm.select(np.array_split(range(len(ds_gsm)), args.num_clients - half_clients)[i])
        split = sub.train_test_split(test_size=0.1)
        train_ds = split['train'].map(lambda x: tok_fn(x['question']), batched=True)
        test_ds = split['test'].map(lambda x: tok_fn(x['question']), batched=True)
        clients.append(VirtualClient(cid, DataLoader(train_ds, batch_size=4, collate_fn=default_data_collator, shuffle=True), DataLoader(test_ds, batch_size=4, collate_fn=default_data_collator), "GSM8K"))

    sensitivity_vectors = []
    full_model.to(accelerator.device); full_model.train()
    init_state = {k: v.clone().cpu() for k, v in full_model.state_dict().items()}
    for client in clients:
        full_model.load_state_dict(init_state); full_model.zero_grad()
        avg_grad = np.zeros(len(get_module_layers(full_model)))
        valid = 0; iter_loader = iter(client.train_dataloader)
        for _ in range(3):
            try: batch = next(iter_loader); batch = {k: v.to(accelerator.device) for k, v in batch.items()}; batch.pop("labels",None); batch.pop("label",None)
            except: break
            full_model(**batch, labels=batch["input_ids"]).loss.backward()
            avg_grad += compute_layer_sensitivity(full_model, accelerator)
            valid += 1; full_model.zero_grad()
        if valid>0: avg_grad/=valid
        sensitivity_vectors.append(avg_grad)
    full_model.cpu()
    if len(sensitivity_vectors)>0: labels = KMeans(n_clusters=min(args.num_clusters,len(sensitivity_vectors)), random_state=42).fit_predict(np.stack(sensitivity_vectors))
    else: labels = []
    clusters = {i: [] for i in range(args.num_clusters)}
    for i, l in enumerate(labels): clusters[l].append(clients[i])

    logger.info("=== [Step 3] Baseline Training (LoRA + Harmonizer) ===")
    trainable_keys = get_trainable_keys(base_emulator)
    init_emu_state = {k: v.clone().detach().cpu() for k, v in base_emulator.state_dict().items() if k in trainable_keys}
    cluster_global_states = {k: copy.deepcopy(init_emu_state) for k in range(args.num_clusters)}
    
    for round_idx in range(args.rounds):
        logger.info(f"--- Round {round_idx + 1} ---")
        round_metrics = {}
        for cid, c_clients in clusters.items():
            if not c_clients: continue
            
            training_model = create_emulator(full_model, budget=args.layer_budget)
            training_model.load_state_dict(cluster_global_states[cid], strict=False)
            training_model.to(accelerator.device); training_model.train()
            
            trainable_params = [p for p in training_model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
            global_update = {}
            current_cpu = {k: v.clone().cpu() for k, v in training_model.state_dict().items() if k in cluster_global_states[cid]}
            
            emu_loss = 0; emu_steps = 0
            
            for client in c_clients:
                training_model.load_state_dict(cluster_global_states[cid], strict=False)
                for step, batch in enumerate(client.train_dataloader):
                    if step >= args.local_steps: break
                    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                    batch.pop("labels", None); batch.pop("label", None)
                    labels = batch["input_ids"].clone(); labels[labels==tokenizer.pad_token_id] = -100
                    
                    outputs = training_model(**batch, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step(); optimizer.zero_grad()
                    emu_loss += loss.item(); emu_steps += 1
                
                client_state = training_model.state_dict()
                for key in current_cpu:
                    delta = client_state[key].cpu() - current_cpu[key]
                    global_update[key] = global_update.get(key, 0) + delta

            if global_update:
                for key in global_update:
                    cluster_global_states[cid][key] += global_update[key] / len(c_clients)
            
            training_model.load_state_dict(cluster_global_states[cid], strict=False)
            plug_loss = evaluate_full_model_plugback(full_model, training_model, c_clients[0].test_dataloader, accelerator, tokenizer)
            avg_emu = emu_loss / emu_steps if emu_steps > 0 else 0
            
            logger.info(f"Cluster {cid} | Emu: {avg_emu:.4f} | Plug: {plug_loss:.4f}")
            round_metrics[f"c{cid}_emu"] = avg_emu; round_metrics[f"c{cid}_plug"] = plug_loss
            del training_model; del optimizer; torch.cuda.empty_cache()

        round_metrics["round"] = round_idx + 1
        accelerator.log(round_metrics, step=round_idx + 1)
    accelerator.end_training()

if __name__ == "__main__": main()