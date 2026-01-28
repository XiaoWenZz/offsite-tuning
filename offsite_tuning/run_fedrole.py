import argparse
import logging
import sys
import copy
import random
import numpy as np
import gc
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
# [组件] Harmonizer (ScaleOT)
# ==========================================
class Harmonizer(nn.Module):
    def __init__(self, config, rank=128, target_attention_type="full"):
        super().__init__()
        self.input_dim = config.hidden_size
        self.rank = rank
        self.down_proj = nn.Linear(self.input_dim, self.rank)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(self.rank, self.input_dim)
        
        # [FIX] Zero Initialization for Identity Start
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
        output = x + residual  # Initial state: output = x + 0 = x
        return output

class SRCLayer(nn.Module):
    def __init__(self, original_layer, idx, rank_ratio=0.6):
        super().__init__()
        self.layer = copy.deepcopy(original_layer)
        self.original_layer_idx = idx 
        # [FIX] Copy attributes
        self.attention_type = getattr(original_layer, "attention_type", "full")
        for param in self.layer.parameters():
            param.requires_grad = False
        self._compress_attention(rank_ratio)

    def _compress_attention(self, rank_ratio):
        if hasattr(self.layer, "self_attn"):
            attn = self.layer.self_attn
            for module_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                if hasattr(attn, module_name):
                    self._apply_svd(getattr(attn, module_name), rank_ratio)

    def _apply_svd(self, linear_layer, rank_ratio):
        try:
            W = linear_layer.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            target_rank = max(1, int(min(W.shape) * rank_ratio))
            U_r = U[:, :target_rank]; S_r = torch.diag(S[:target_rank]); Vt_r = Vt[:target_rank, :]
            W_approx = U_r @ S_r @ Vt_r
            linear_layer.weight.data = W_approx.to(linear_layer.weight.dtype)
        except Exception: pass 

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)

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

def create_custom_emulator(full_model, sensitivity_vector, budget_adapter=4, budget_src=4):
    emulator = copy.deepcopy(full_model)
    layers = get_module_layers(emulator)
    total_layers = len(layers)
    if isinstance(sensitivity_vector, list): sensitivity_vector = np.array(sensitivity_vector)
    ranked_layers = sorted(enumerate(sensitivity_vector), key=lambda x: x[1], reverse=True)
    mandatory_indices = {0, total_layers - 1}
    adapter_indices = set(mandatory_indices)
    for idx, score in ranked_layers:
        if len(adapter_indices) >= budget_adapter: break
        adapter_indices.add(idx)
    src_indices = set()
    for idx, score in ranked_layers:
        if idx not in adapter_indices:
            if len(src_indices) < budget_src: src_indices.add(idx)
    
    new_layers_list = []
    current_layer_idx = 0
    ref_type = getattr(layers[0], "attention_type", "full")

    while current_layer_idx < total_layers:
        if current_layer_idx in adapter_indices:
            layer = layers[current_layer_idx]
            layer.original_layer_idx = current_layer_idx
            for param in layer.parameters(): param.requires_grad = True
            new_layers_list.append(layer)
            current_layer_idx += 1
        elif current_layer_idx in src_indices:
            src_layer = SRCLayer(layers[current_layer_idx], current_layer_idx, rank_ratio=0.6)
            new_layers_list.append(src_layer)
            current_layer_idx += 1
        else:
            gap_size = 0
            while (current_layer_idx + gap_size < total_layers) and \
                  (current_layer_idx + gap_size not in adapter_indices) and \
                  (current_layer_idx + gap_size not in src_indices):
                gap_size += 1
            harmonizer = Harmonizer(full_model.config, rank=128, target_attention_type=ref_type)
            new_layers_list.append(harmonizer)
            current_layer_idx += gap_size
            
    set_module_layers(emulator, nn.ModuleList(new_layers_list))
    emulator.config.use_cache = False
    
    # Freeze Anchors (Embedding/Head)
    for name, param in emulator.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = False
            
    return emulator

def get_trainable_keys(model):
    """[FIX] 获取所有需要更新的参数名称"""
    keys = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            keys.append(name)
    return set(keys)

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

def evaluate_full_model_plugback(full_model, emulator_model, dataloader, accelerator, tokenizer):
    full_layers = get_module_layers(full_model)
    emulator_layers = get_module_layers(emulator_model)
    original_weights = {}
    
    with torch.no_grad():
        for emu_layer in emulator_layers:
            if hasattr(emu_layer, "original_layer_idx") and emu_layer.original_layer_idx is not None:
                real_idx = emu_layer.original_layer_idx
                has_grad = any(p.requires_grad for p in emu_layer.parameters())
                if not has_grad: continue
                target_layer = full_layers[real_idx]
                original_weights[real_idx] = {k: v.clone().cpu() for k, v in target_layer.state_dict().items()}
                target_layer.load_state_dict(emu_layer.state_dict())

    full_model.to(accelerator.device); full_model.eval()
    total_loss = 0; steps = 0; MAX_EVAL = 10
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= MAX_EVAL: break
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            # [FIX] Conflict
            batch.pop("labels", None); batch.pop("label", None)
            labels = batch["input_ids"].clone()
            if tokenizer and tokenizer.pad_token_id is not None: labels[labels == tokenizer.pad_token_id] = -100
            outputs = full_model(**batch, labels=labels)
            total_loss += outputs.loss.item(); steps += 1
    avg_loss = total_loss / steps if steps > 0 else 0.0
    
    full_model.cpu()
    with torch.no_grad():
        for real_idx, weights in original_weights.items(): full_layers[real_idx].load_state_dict(weights)
    torch.cuda.empty_cache()
    return avg_loss

class VirtualClient:
    def __init__(self, client_id, train_loader, test_loader, label_dist_str):
        self.id = client_id; self.train_dataloader = train_loader; self.test_dataloader = test_loader; self.label_info = label_dist_str

def parse_args():
    parser = argparse.ArgumentParser(description="FedRole")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset_name", type=str, default="mixed_yelp_gsm8k")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_clusters", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--layer_budget", type=int, default=4)
    parser.add_argument("--src_budget", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="fedrole_benchmark")
    parser.add_argument("--wandb_run_name", type=str, default="fedrole_run")
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
    full_model.cpu() 
    
    # ... (Data Loading Omitted for Brevity - Same as before) ...
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

    # Sensitivity Profiling
    sensitivity_vectors = []
    full_model.to(accelerator.device); full_model.train()
    init_state = {k: v.clone().cpu() for k, v in full_model.state_dict().items()}
    for client in clients:
        full_model.load_state_dict(init_state); full_model.zero_grad()
        avg_grad = np.zeros(len(get_module_layers(full_model)))
        valid = 0; iter_loader = iter(client.train_dataloader)
        for _ in range(3):
            try: batch = next(iter_loader)
            except: break
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            batch.pop("labels", None); batch.pop("label", None)
            full_model(**batch, labels=batch["input_ids"]).loss.backward()
            avg_grad += compute_layer_sensitivity(full_model, accelerator)
            valid += 1; full_model.zero_grad()
        if valid > 0: avg_grad /= valid
        sensitivity_vectors.append(avg_grad)
    full_model.cpu()

    # Clustering
    if len(sensitivity_vectors) > 0:
        labels = KMeans(n_clusters=min(args.num_clusters, len(sensitivity_vectors)), random_state=42, n_init=10).fit_predict(np.stack(sensitivity_vectors))
    else: labels = []
    clusters = {i: [] for i in range(args.num_clusters)}
    for i, label in enumerate(labels): clusters[label].append(clients[i])

    logger.info("=== [Step 3] Initializing Emulators ===")
    cluster_initial_states = {}; cluster_layer_sens = {}
    for cid in range(args.num_clusters):
        c_idxs = [i for i, l in enumerate(labels) if l == cid]
        if not c_idxs: continue
        sens = np.mean(np.stack(sensitivity_vectors)[c_idxs], axis=0)
        cluster_layer_sens[cid] = sens
        emu = create_custom_emulator(full_model, sens, args.layer_budget, args.src_budget)
        cluster_initial_states[cid] = {k: v.clone().cpu() for k, v in emu.state_dict().items()}
        del emu

    logger.info("=== [Step 4] Federated Training ===")
    cluster_global_states = copy.deepcopy(cluster_initial_states)
    
    for round_idx in range(args.rounds):
        logger.info(f"--- Round {round_idx + 1} ---")
        round_metrics = {}
        for cid, c_clients in clusters.items():
            if not c_clients: continue
            
            temp_model = create_custom_emulator(full_model, cluster_layer_sens[cid], args.layer_budget, args.src_budget)
            temp_model.load_state_dict(cluster_global_states[cid])
            temp_model.to(accelerator.device); temp_model.train()
            
            # [FIX] Get keys that REALLY require grad
            trainable_keys = get_trainable_keys(temp_model)
            if round_idx == 0: logger.info(f"Cluster {cid} Trainable Params: {len(trainable_keys)}")

            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, temp_model.parameters()), lr=args.lr)
            global_update = {}
            cluster_train_loss = 0.0 # Track Emulator Loss
            total_train_steps = 0
            
            current_cpu = {k: v.clone().cpu() for k, v in temp_model.state_dict().items()}
            
            for client in c_clients:
                temp_model.load_state_dict(current_cpu)
                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, temp_model.parameters()), lr=args.lr)
                optimizer.zero_grad()
                
                for step, batch in enumerate(client.train_dataloader):
                    if step >= args.local_steps: break
                    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                    batch.pop("labels", None); batch.pop("label", None)
                    labels = batch["input_ids"].clone()
                    if tokenizer.pad_token_id is not None: labels[labels == tokenizer.pad_token_id] = -100
                    
                    outputs = temp_model(**batch, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step(); optimizer.zero_grad()
                    
                    cluster_train_loss += loss.item()
                    total_train_steps += 1
                
                # [FIX] Aggregation Logic
                client_state = temp_model.state_dict()
                for key in trainable_keys:
                    if key in client_state:
                        delta = client_state[key].cpu() - current_cpu[key]
                        global_update[key] = global_update.get(key, 0) + delta
            
            # Apply Update
            if global_update:
                for key in global_update:
                    if key in cluster_global_states[cid]:
                        cluster_global_states[cid][key] += global_update[key] / len(c_clients)
            
            # Eval (Smart Plug-back)
            # Load updated state into temp_model to pass to eval function (which checks original_layer_idx)
            temp_model.load_state_dict(cluster_global_states[cid])
            plug_loss = evaluate_full_model_plugback(full_model, temp_model, c_clients[0].test_dataloader, accelerator, tokenizer)
            
            avg_emu_loss = cluster_train_loss / total_train_steps if total_train_steps > 0 else 0.0
            
            logger.info(f"Cluster {cid} | Emulator Train Loss: {avg_emu_loss:.4f} | Plug-in Eval Loss: {plug_loss:.4f}")
            round_metrics[f"cluster_{cid}_emu_loss"] = avg_emu_loss
            round_metrics[f"cluster_{cid}_plugin_loss"] = plug_loss
            
            del temp_model; torch.cuda.empty_cache()

        round_metrics["round"] = round_idx + 1
        accelerator.log(round_metrics, step=round_idx + 1)
    accelerator.end_training()

if __name__ == "__main__": main()