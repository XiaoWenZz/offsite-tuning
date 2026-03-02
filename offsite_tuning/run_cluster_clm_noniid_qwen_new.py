import argparse
import logging
import sys
import copy
import math
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datasets
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, default_data_collator, set_seed
from accelerate import Accelerator
from accelerate.logging import get_logger
import wandb
import os
from peft import get_peft_model, LoraConfig, TaskType

logger = get_logger(__name__)

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
        self.original_layer_idx = None

    def forward(self, hidden_states, *args, **kwargs):
        x = hidden_states
        while isinstance(x, tuple): x = x[0]
        if not isinstance(x, torch.Tensor): return x
        return x + self.up_proj(self.activation(self.down_proj(x)))

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
            model.model.layers = new_layers; model.config.num_hidden_layers = len(new_layers); return
    if hasattr(model, "layers"):
        model.layers = new_layers; model.config.num_hidden_layers = len(new_layers); return
    raise ValueError(f"Unknown architecture for setting layers")

def get_trainable_keys(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}

def load_benchmark_data(dataset_name, tokenizer, cache_dir=None, num_samples=1000):
    BASE_DIR = "/data/xiaowen/piqa_local/physicaliqa-train-dev"
    DATA_PATH = os.path.join(BASE_DIR, "train.jsonl")
    LABEL_PATH = os.path.join(BASE_DIR, "train-labels.lst")
    ds = None; data_source_type = "unknown"

    if dataset_name == "piqa" or dataset_name == "hellaswag":
        if os.path.exists(DATA_PATH) and os.path.exists(LABEL_PATH):
            ds_text = datasets.load_dataset("json", data_files={"train": DATA_PATH}, split="train")
            with open(LABEL_PATH, "r") as f: labels = [int(line.strip()) for line in f.readlines()]
            min_len = min(len(ds_text), len(labels))
            ds = ds_text.select(range(min_len)).add_column("label", labels[:min_len])
            data_source_type = "piqa_local"
        else:
            try: ds = datasets.load_dataset(dataset_name, split="train", cache_dir=cache_dir); data_source_type = "network"
            except: pass

    if ds is None:
        ds = datasets.Dataset.from_list([{"text": "The quick brown fox."}] * num_samples); data_source_type = "synthetic"

    ds = ds.select(range(min(len(ds), num_samples)))
    def format_fn(ex):
        if data_source_type == "piqa_local": return tokenizer(f"Goal: {ex['goal']}\nSolution: {ex['sol1'] if ex['label']==0 else ex['sol2']}", truncation=True, max_length=128, padding="max_length")
        elif data_source_type == "network" and dataset_name == "hellaswag": return tokenizer(f"Context: {ex['ctx']}\nEnding: {ex['endings'][int(ex['label'])]}", truncation=True, max_length=128, padding="max_length")
        else: return tokenizer(ex.get('text', "Pad"), truncation=True, max_length=128, padding="max_length")
    return ds.map(format_fn, remove_columns=ds.column_names)

def train_global_harmonizers(full_model, tokenizer, accelerator, num_steps=200):
    logger.info("⚡ [Server] Pre-training Global Harmonizer Pool...")
    
    if hasattr(full_model, "model") and hasattr(full_model.model, "layers"):
        original_layers = full_model.model.layers
    elif hasattr(full_model, "layers"):
        original_layers = full_model.layers
    else:
        raise ValueError("Unsupported architecture")
        
    num_layers = len(original_layers)
    ref_type = getattr(original_layers[0], "attention_type", "full")
    
    global_harmonizers = nn.ModuleList([Harmonizer(full_model.config, 128, ref_type) for _ in range(num_layers)])
    global_harmonizers.to(accelerator.device); global_harmonizers.train()
    optimizer = torch.optim.AdamW(global_harmonizers.parameters(), lr=1e-3)
    loss_fct = nn.KLDivLoss(reduction="batchmean")

    raw_ds = [{"text": "The quick brown fox jumps over the lazy dog. " * 20} for _ in range(400)]
    def tok(ex): return tokenizer(ex["text"], truncation=True, max_length=128, padding="max_length")
    tokenized_ds = [tok(x) for x in raw_ds]
    align_loader = DataLoader(
        list(zip([torch.tensor(t['input_ids']) for t in tokenized_ds], [torch.tensor(t['attention_mask']) for t in tokenized_ds])), 
        batch_size=4, shuffle=True
    )

    full_model.to(accelerator.device)
    iterator = iter(align_loader)
    
    for param in full_model.parameters():
        param.requires_grad = False
    
    for step in range(num_steps):
        try: batch_raw = next(iterator)
        except: iterator = iter(align_loader); batch_raw = next(iterator)
        
        input_ids = batch_raw[0].to(accelerator.device)
        attention_mask = batch_raw[1].to(accelerator.device)

        with torch.no_grad():
            full_model.eval()
            teacher_logits = full_model(input_ids=input_ids, attention_mask=attention_mask).logits

        mask = torch.rand(num_layers) > 0.5
        mask[0] = True; mask[-1] = True
        
        mixed_layers = nn.ModuleList()
        for idx in range(num_layers):
            if mask[idx]: mixed_layers.append(original_layers[idx])
            else: mixed_layers.append(global_harmonizers[idx])
            
        if hasattr(full_model, "model") and hasattr(full_model.model, "layers"):
            full_model.model.layers = mixed_layers
        else: full_model.layers = mixed_layers
            
        full_model.train()
        student_logits = full_model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        if hasattr(full_model, "model") and hasattr(full_model.model, "layers"):
            full_model.model.layers = original_layers
        else: full_model.layers = original_layers

        loss = loss_fct(torch.nn.functional.log_softmax(student_logits, dim=-1), torch.nn.functional.softmax(teacher_logits, dim=-1))
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        if step % 50 == 0: logger.info(f"   Pool Align Step {step} | Loss: {loss.item():.4f}")

    global_harmonizers.cpu()
    logger.info("✅ Global Harmonizer Pool Ready.")
    return global_harmonizers

def create_emulator(full_model, global_harmonizers, budget=6):
    emulator = copy.deepcopy(full_model)
    layers = get_module_layers(emulator)
    total = len(layers)
    
    indices = {0, total-1}
    rem = budget - 2
    if rem > 0:
        if rem == 1: indices.add(total//2)
        else: indices.update(np.linspace(1, total-2, rem, dtype=int))
    adapter_indices = set(indices)
    
    new_layers = []; layer_map = {}
    for curr in range(total):
        if curr in adapter_indices:
            l = layers[curr]
            l.original_layer_idx = curr
            new_layers.append(l)
            layer_map[curr] = curr
        else:
            h = copy.deepcopy(global_harmonizers[curr])
            h.original_layer_idx = None
            new_layers.append(h)
            
    set_module_layers(emulator, nn.ModuleList(new_layers))
    emulator.config.use_cache = False
    emulator.config.layer_map = layer_map
    
    for param in emulator.parameters(): param.requires_grad = False
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    emulator = get_peft_model(emulator, peft_config)
    if hasattr(emulator, "config"): emulator.config.layer_map = layer_map
    for name, param in emulator.named_parameters():
        if "Harmonizer" in name or "lora_" in name: param.requires_grad = True
        else: param.requires_grad = False
    return emulator

def evaluate_full_model_plugback(full_model, emulator_model, dataloader, accelerator, tokenizer):
    full_model.eval()
    emu_state = emulator_model.state_dict()
    adapter_lora_state = {}
    
    idx_map = {}
    if hasattr(emulator_model, "config") and hasattr(emulator_model.config, "layer_map"):
        idx_map = emulator_model.config.layer_map
    elif hasattr(emulator_model, "base_model") and hasattr(emulator_model.base_model.model, "config"):
        idx_map = emulator_model.base_model.model.config.layer_map
            
    for k, v in emu_state.items():
        if "lora_" in k:
            parts = k.split('.')
            try:
                if 'layers' in parts:
                    emu_layer_idx = int(parts[parts.index('layers') + 1])
                    if emu_layer_idx in idx_map:
                        parts[parts.index('layers') + 1] = str(idx_map[emu_layer_idx])
                        adapter_lora_state[".".join(parts)] = v.cpu()
            except: continue

    temp_full = copy.deepcopy(full_model)
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    peft_full = get_peft_model(temp_full, peft_config)
    peft_full.load_state_dict(adapter_lora_state, strict=False)
    peft_full.to(accelerator.device); peft_full.eval()
    
    total_loss = 0; steps = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10: break
            batch = {k: v.to(accelerator.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            labels = batch["input_ids"].clone(); labels[labels==tokenizer.pad_token_id] = -100
            outputs = peft_full(**batch, labels=labels)
            total_loss += outputs.loss.item(); steps += 1
            
    del peft_full; del temp_full; torch.cuda.empty_cache()
    return total_loss / steps if steps > 0 else 0.0

class VirtualClient:
    def __init__(self, client_id, train_loader, test_loader):
        self.id = client_id; self.train_dataloader = train_loader; self.test_dataloader = test_loader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset_name", type=str, default="mixed_piqa_hellaswag")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_clusters", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--layer_budget", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_steps", type=int, default=10)
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
    full_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    full_model.cpu() 
    
    clients = []
    ds_a = load_benchmark_data("piqa", tokenizer, cache_dir=args.cache_dir, num_samples=2000)
    for i in range(args.num_clients):
        sub = ds_a.select(np.array_split(range(len(ds_a)), args.num_clients)[i])
        split = sub.train_test_split(test_size=0.1)
        clients.append(VirtualClient(i, DataLoader(split['train'], batch_size=4, collate_fn=default_data_collator, shuffle=True), DataLoader(split['test'], batch_size=4, collate_fn=default_data_collator)))

    # Dummy clustering for baseline to match structure
    clusters = {i: clients[i*(args.num_clients//args.num_clusters):(i+1)*(args.num_clients//args.num_clusters)] for i in range(args.num_clusters)}

    logger.info("=== [Step 3] Baseline Training Global Harmonizers ===")
    global_harmonizers = train_global_harmonizers(full_model, tokenizer, accelerator, num_steps=200)

    cluster_global_states = {}
    for cid in range(args.num_clusters):
        emu = create_emulator(full_model, global_harmonizers, args.layer_budget)
        trainable_keys = get_trainable_keys(emu)
        cluster_global_states[cid] = {k: v.clone().cpu() for k, v in emu.state_dict().items() if k in trainable_keys}
        del emu

    logger.info("=== [Step 5] Federated Training ===")
    for round_idx in range(args.rounds):
        logger.info(f"--- Round {round_idx + 1} ---")
        round_metrics = {}
        for cid, c_clients in clusters.items():
            if not c_clients: continue
            
            training_model = create_emulator(full_model, global_harmonizers, args.layer_budget)
            training_model.load_state_dict(cluster_global_states[cid], strict=False)
            training_model.to(accelerator.device); training_model.train()
            
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, training_model.parameters()), lr=args.lr)
            global_update = {}
            current_cpu = {k: v.clone().cpu() for k, v in training_model.state_dict().items() if k in cluster_global_states[cid]}
            emu_loss = 0; emu_steps = 0
            
            for client in c_clients:
                training_model.load_state_dict(cluster_global_states[cid], strict=False)
                for step, batch in enumerate(client.train_dataloader):
                    if step >= args.local_steps: break
                    batch = {k: v.to(accelerator.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
                    labels = batch["input_ids"].clone(); labels[labels==tokenizer.pad_token_id] = -100
                    loss = training_model(**batch, labels=labels).loss
                    loss.backward(); optimizer.step(); optimizer.zero_grad()
                    emu_loss += loss.item(); emu_steps += 1
                
                for key in current_cpu:
                    global_update[key] = global_update.get(key, 0) + (training_model.state_dict()[key].cpu() - current_cpu[key])

            if global_update:
                for key in global_update: cluster_global_states[cid][key] += global_update[key] / len(c_clients)
            
            training_model.load_state_dict(cluster_global_states[cid], strict=False)
            plug_loss = evaluate_full_model_plugback(full_model, training_model, c_clients[0].test_dataloader, accelerator, tokenizer)
            avg_emu = emu_loss / emu_steps if emu_steps > 0 else 0.0
            
            try: emu_ppl = math.exp(avg_emu)
            except: emu_ppl = float('inf')
            try: plug_ppl = math.exp(plug_loss)
            except: plug_ppl = float('inf')
            
            logger.info(f"Cluster {cid} | Emu PPL: {emu_ppl:.2f} | Plug PPL: {plug_ppl:.2f}")
            round_metrics[f"c{cid}_emu"] = avg_emu; round_metrics[f"c{cid}_plug"] = plug_loss; round_metrics[f"c{cid}_plug_ppl"] = plug_ppl
            del training_model; del optimizer; torch.cuda.empty_cache()

        round_metrics["round"] = round_idx + 1
        accelerator.log(round_metrics, step=round_idx + 1)
    accelerator.end_training()

if __name__ == "__main__": main()