import argparse
import logging
import sys
import copy
import random
import math
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

# ==========================================
# [组件] Harmonizer & SRC
# ==========================================
class Harmonizer(nn.Module):
    def __init__(self, config, rank=128, target_attention_type="full"):
        super().__init__()
        self.input_dim = config.hidden_size
        self.rank = rank
        self.down_proj = nn.Linear(self.input_dim, self.rank)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(self.rank, self.input_dim)
        
        # Zero Init -> Identity Start
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

class SRCLayer(nn.Module):
    def __init__(self, original_layer, idx, rank_ratio=0.6):
        super().__init__()
        self.layer = copy.deepcopy(original_layer)
        self.original_layer_idx = idx 
        self.attention_type = getattr(original_layer, "attention_type", "full")
        for param in self.layer.parameters(): param.requires_grad = False
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

def get_trainable_keys(model):
    keys = []
    for name, param in model.named_parameters():
        if param.requires_grad: keys.append(name)
    return set(keys)

# ==========================================
# [关键] Local Benchmark Loader
# ==========================================
def load_benchmark_data(dataset_name, tokenizer, cache_dir=None, num_samples=1000):
    logger.info(f"📚 Loading Benchmark: {dataset_name}...")
    
    # 定义本地路径
    BASE_DIR = "/data/xiaowen/piqa_local/physicaliqa-train-dev"
    DATA_PATH = os.path.join(BASE_DIR, "train.jsonl")
    LABEL_PATH = os.path.join(BASE_DIR, "train-labels.lst")
    
    ds = None
    data_source_type = "unknown"

    # Case A: PIQA / Hellaswag -> 优先读本地 PIQA
    if dataset_name == "piqa" or dataset_name == "hellaswag":
        if os.path.exists(DATA_PATH) and os.path.exists(LABEL_PATH):
            logger.info(f"   🏠 Found local PIQA files at {BASE_DIR}")
            
            # 1. 加载文本
            ds_text = datasets.load_dataset("json", data_files={"train": DATA_PATH}, split="train")
            
            # 2. 加载标签
            with open(LABEL_PATH, "r") as f:
                labels = [int(line.strip()) for line in f.readlines()]
            
            # 3. 合并
            min_len = min(len(ds_text), len(labels))
            ds_text = ds_text.select(range(min_len))
            labels = labels[:min_len]
            ds = ds_text.add_column("label", labels)
            
            data_source_type = "piqa_local"
        else:
            logger.warning(f"   ⚠️ Local file not found at {DATA_PATH}, trying network...")
            try:
                ds = datasets.load_dataset(dataset_name, split="train", cache_dir=cache_dir)
                data_source_type = "network"
            except: pass

    # Case B: 终极保底
    if ds is None:
        logger.warning("🚨 ALL LOADS FAILED. Generating SYNTHETIC data.")
        ds = datasets.Dataset.from_list([{"text": "The quick brown fox jumps over the lazy dog."}] * num_samples)
        data_source_type = "synthetic"

    # 截取
    ds = ds.select(range(min(len(ds), num_samples)))
    logger.info(f"   ✅ Loaded source: {data_source_type} | Size: {len(ds)}")

    # 格式化
    def format_fn(ex):
        if data_source_type == "piqa_local":
            label = ex['label']
            correct_sol = ex['sol1'] if label == 0 else ex['sol2']
            text = f"Goal: {ex['goal']}\nSolution: {correct_sol}"
        elif data_source_type == "network" and dataset_name == "hellaswag":
            correct_ending = ex['endings'][int(ex['label'])]
            text = f"Context: {ex['ctx']}\nEnding: {correct_ending}"
        else:
            content = ex.get('text', "")
            text = content if len(content) > 10 else "Padding text."
        return tokenizer(text, truncation=True, max_length=128, padding="max_length")

    tokenized_ds = ds.map(format_fn, remove_columns=ds.column_names)
    return tokenized_ds

# ==========================================
# Core Logic
# ==========================================
def create_custom_emulator(full_model, sensitivity_vector, budget_adapter=4, budget_src=4):
    emulator = copy.deepcopy(full_model)
    layers = get_module_layers(emulator)
    total_layers = len(layers)
    
    if isinstance(sensitivity_vector, list): sensitivity_vector = np.array(sensitivity_vector)
    ranked = sorted(enumerate(sensitivity_vector), key=lambda x: x[1], reverse=True)
    
    adapter_indices = set({0, total_layers-1})
    for idx, _ in ranked:
        if len(adapter_indices) >= budget_adapter: break
        adapter_indices.add(idx)
    
    src_indices = set()
    for idx, _ in ranked:
        if idx not in adapter_indices:
            if len(src_indices) < budget_src: src_indices.add(idx)
            
    new_layers = []
    curr = 0
    ref_type = getattr(layers[0], "attention_type", "full")
    
    while curr < total_layers:
        if curr in adapter_indices:
            l = layers[curr]
            l.original_layer_idx = curr
            new_layers.append(l)
            curr += 1
        elif curr in src_indices:
            new_layers.append(SRCLayer(layers[curr], curr, 0.6))
            curr += 1
        else:
            gap = 0
            while (curr + gap < total_layers) and (curr+gap not in adapter_indices) and (curr+gap not in src_indices): gap+=1
            new_layers.append(Harmonizer(full_model.config, 128, ref_type))
            curr += gap
            
    set_module_layers(emulator, nn.ModuleList(new_layers))
    emulator.config.use_cache = False
    
    # Save Map
    layer_map = {}
    for i, layer in enumerate(new_layers):
        if hasattr(layer, "original_layer_idx") and layer.original_layer_idx is not None:
            layer_map[i] = layer.original_layer_idx
    emulator.config.layer_map = layer_map 
    
    # LoRA
    for param in emulator.parameters(): param.requires_grad = False
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    emulator = get_peft_model(emulator, peft_config)
    if hasattr(emulator, "config"): emulator.config.layer_map = layer_map
    
    for name, param in emulator.named_parameters():
        if "Harmonizer" in name or "lora_" in name: param.requires_grad = True
        else: param.requires_grad = False
    return emulator

def server_side_alignment(full_model, emulator, tokenizer, accelerator, num_steps=100):
    """Alignment using Synthetic/WikiText to warm up Harmonizer"""
    logger.info("⚡ Server-Side Alignment...")
    # Generate simple synthetic data for alignment (just to learn identity mapping)
    raw_ds = [{"text": "The quick brown fox jumps over the lazy dog. " * 20} for _ in range(200)]
    
    def tok(ex): return tokenizer(ex["text"], truncation=True, max_length=128, padding="max_length")
    tokenized_ds = [tok(x) for x in raw_ds]
    input_ids = torch.tensor([t['input_ids'] for t in tokenized_ds])
    attention_mask = torch.tensor([t['attention_mask'] for t in tokenized_ds])
    
    align_loader = DataLoader(
        list(zip(input_ids, attention_mask)), 
        batch_size=4, shuffle=True, 
        collate_fn=lambda x: {"input_ids": torch.stack([i[0] for i in x]), "attention_mask": torch.stack([i[1] for i in x])}
    )

    full_model.eval(); full_model.to(accelerator.device)
    emulator.train(); emulator.to(accelerator.device)
    
    params = [p for n, p in emulator.named_parameters() if "Harmonizer" in n]
    if not params: return
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    loss_fct = nn.KLDivLoss(reduction="batchmean")

    iterator = iter(align_loader)
    for step in range(num_steps):
        try: batch = next(iterator)
        except: iterator = iter(align_loader); batch = next(iterator)
        
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        with torch.no_grad(): teacher_logits = full_model(**batch).logits
        student_logits = emulator(**batch).logits

        loss = loss_fct(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.softmax(teacher_logits, dim=-1)
        )
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        if step % 20 == 0: logger.info(f"   Align Step {step} | Loss: {loss.item():.4f}")

    for name, param in emulator.named_parameters():
        if "lora_" in name: param.requires_grad = True
    torch.cuda.empty_cache()

def evaluate_full_model_plugback(full_model, emulator_model, dataloader, accelerator, tokenizer):
    """Safe LoRA Plug-back using COPY"""
    full_model.eval()
    emu_state = emulator_model.state_dict()
    adapter_lora_state = {}
    
    idx_map = {}
    if hasattr(emulator_model, "config") and hasattr(emulator_model.config, "layer_map"):
        idx_map = emulator_model.config.layer_map
    elif hasattr(emulator_model, "base_model") and hasattr(emulator_model.base_model.model, "config"):
        if hasattr(emulator_model.base_model.model.config, "layer_map"):
            idx_map = emulator_model.base_model.model.config.layer_map
            
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
                        adapter_lora_state[new_key] = v.cpu()
                        mapped_count += 1
            except: continue
            
    if not hasattr(evaluate_full_model_plugback, "debug_printed"):
        print(f"🔥 Plug-back: Mapped {mapped_count} keys.")
        evaluate_full_model_plugback.debug_printed = True

    # COPY FULL MODEL
    temp_full = copy.deepcopy(full_model)
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    peft_full = get_peft_model(temp_full, peft_config)
    peft_full.load_state_dict(adapter_lora_state, strict=False)
    peft_full.to(accelerator.device); peft_full.eval()
    
    total_loss = 0; steps = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10: break
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            batch.pop("labels", None); batch.pop("label", None)
            labels = batch["input_ids"].clone(); labels[labels==tokenizer.pad_token_id] = -100
            outputs = peft_full(**batch, labels=labels)
            total_loss += outputs.loss.item(); steps += 1
            
    del peft_full; del temp_full; torch.cuda.empty_cache()
    return total_loss / steps if steps > 0 else 0.0

class VirtualClient:
    def __init__(self, client_id, train_loader, test_loader, label_dist_str):
        self.id = client_id; self.train_dataloader = train_loader; self.test_dataloader = test_loader; self.label_info = label_dist_str

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset_name", type=str, default="mixed_piqa_hellaswag")
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
    
    # [NEW] Load Benchmarks (PIQA + HellaSwag)
    clients = []
    half_clients = args.num_clients // 2
    
    # Task A: PIQA (Local)
    ds_a = load_benchmark_data("piqa", tokenizer, cache_dir=args.cache_dir, num_samples=2000)
    # Task B: HellaSwag (Fallback to PIQA Local if net fails)
    ds_b = load_benchmark_data("hellaswag", tokenizer, cache_dir=args.cache_dir, num_samples=2000)
    
    for i in range(half_clients):
        sub = ds_a.select(np.array_split(range(len(ds_a)), half_clients)[i])
        split = sub.train_test_split(test_size=0.1)
        clients.append(VirtualClient(i, DataLoader(split['train'], batch_size=4, collate_fn=default_data_collator, shuffle=True), DataLoader(split['test'], batch_size=4, collate_fn=default_data_collator), "PIQA"))
            
    for i in range(args.num_clients - half_clients):
        cid = half_clients + i
        sub = ds_b.select(np.array_split(range(len(ds_b)), args.num_clients - half_clients)[i])
        split = sub.train_test_split(test_size=0.1)
        clients.append(VirtualClient(cid, DataLoader(split['train'], batch_size=4, collate_fn=default_data_collator, shuffle=True), DataLoader(split['test'], batch_size=4, collate_fn=default_data_collator), "HellaSwag"))

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

    logger.info("=== [Step 3] Initializing Emulators with LoRA & Alignment ===")
    cluster_initial_states = {}; cluster_sens = {}
    for cid in range(args.num_clusters):
        c_idxs = [i for i, l in enumerate(labels) if l == cid]
        if not c_idxs: continue
        sens = np.mean(np.stack(sensitivity_vectors)[c_idxs], axis=0)
        cluster_sens[cid] = sens
        
        emu = create_custom_emulator(full_model, sens, args.layer_budget, args.src_budget)
        
        server_side_alignment(full_model, emu, tokenizer, accelerator, num_steps=100)
        
        trainable_keys = get_trainable_keys(emu)
        cluster_initial_states[cid] = {k: v.clone().cpu() for k, v in emu.state_dict().items() if k in trainable_keys}
        del emu

    logger.info("=== [Step 4] Federated Training ===")
    cluster_global_states = copy.deepcopy(cluster_initial_states)
    
    for round_idx in range(args.rounds):
        logger.info(f"--- Round {round_idx + 1} ---")
        round_metrics = {}
        for cid, c_clients in clusters.items():
            if not c_clients: continue
            
            temp_model = create_custom_emulator(full_model, cluster_sens[cid], args.layer_budget, args.src_budget)
            temp_model.load_state_dict(cluster_global_states[cid], strict=False)
            temp_model.to(accelerator.device); temp_model.train()
            
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, temp_model.parameters()), lr=args.lr)
            global_update = {}
            current_cpu = {k: v.clone().cpu() for k, v in temp_model.state_dict().items() if k in cluster_global_states[cid]}
            emu_loss = 0; emu_steps = 0
            
            for client in c_clients:
                temp_model.load_state_dict(cluster_global_states[cid], strict=False)
                for step, batch in enumerate(client.train_dataloader):
                    if step >= args.local_steps: break
                    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                    batch.pop("labels", None); batch.pop("label", None)
                    labels = batch["input_ids"].clone(); labels[labels==tokenizer.pad_token_id] = -100
                    
                    outputs = temp_model(**batch, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step(); optimizer.zero_grad()
                    emu_loss += loss.item(); emu_steps += 1
                
                client_state = temp_model.state_dict()
                for key in current_cpu:
                    delta = client_state[key].cpu() - current_cpu[key]
                    global_update[key] = global_update.get(key, 0) + delta
            
            if global_update:
                for key in global_update:
                    cluster_global_states[cid][key] += global_update[key] / len(c_clients)
            
            temp_model.load_state_dict(cluster_global_states[cid], strict=False)
            plug_loss = evaluate_full_model_plugback(full_model, temp_model, c_clients[0].test_dataloader, accelerator, tokenizer)
            
            avg_emu = emu_loss / emu_steps if emu_steps > 0 else 0.0
            
            # [NEW] PPL Calc
            try: emu_ppl = math.exp(avg_emu)
            except: emu_ppl = float('inf')
            try: plug_ppl = math.exp(plug_loss)
            except: plug_ppl = float('inf')
            
            logger.info(f"Cluster {cid} | Emu: {avg_emu:.4f} (PPL: {emu_ppl:.2f}) | Plug: {plug_loss:.4f} (PPL: {plug_ppl:.2f})")
            round_metrics[f"c{cid}_emu"] = avg_emu
            round_metrics[f"c{cid}_plug"] = plug_loss
            round_metrics[f"c{cid}_plug_ppl"] = plug_ppl
            del temp_model; torch.cuda.empty_cache()

        round_metrics["round"] = round_idx + 1
        accelerator.log(round_metrics, step=round_idx + 1)
    accelerator.end_training()

if __name__ == "__main__": main()