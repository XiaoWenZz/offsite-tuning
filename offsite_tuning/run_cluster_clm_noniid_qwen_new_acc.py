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
    raise ValueError("Architecture not found")

def set_module_layers(model, new_layers):
    if hasattr(model, "model"):
        if hasattr(model.model, "layers"): 
            model.model.layers = new_layers; model.config.num_hidden_layers = len(new_layers); return
    if hasattr(model, "layers"):
        model.layers = new_layers; model.config.num_hidden_layers = len(new_layers); return
    raise ValueError("Unknown architecture")

def get_trainable_keys(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}

def load_raw_benchmark_data(dataset_name, cache_dir=None, num_samples=1000):
    BASE_DIR = "/data/xiaowen/piqa_local/physicaliqa-train-dev"
    DATA_PATH = os.path.join(BASE_DIR, "train.jsonl")
    LABEL_PATH = os.path.join(BASE_DIR, "train-labels.lst")
    ds = None; data_source_type = "unknown"

    if dataset_name == "piqa" or dataset_name == "hellaswag":
        if os.path.exists(DATA_PATH) and os.path.exists(LABEL_PATH) and dataset_name == "piqa":
            ds_text = datasets.load_dataset("json", data_files={"train": DATA_PATH}, split="train")
            with open(LABEL_PATH, "r") as f: labels = [int(line.strip()) for line in f.readlines()]
            min_len = min(len(ds_text), len(labels))
            ds = ds_text.select(range(min_len)).add_column("label", labels[:min_len])
            data_source_type = "piqa_local"
        else:
            try: ds = datasets.load_dataset(dataset_name, split="train", cache_dir=cache_dir); data_source_type = "network"
            except: pass

    if ds is None:
        if dataset_name == "piqa": ds = datasets.Dataset.from_list([{"goal": "test", "sol1": "a", "sol2": "b", "label": 0}] * num_samples)
        else: ds = datasets.Dataset.from_list([{"ctx": "test", "endings": ["a", "b", "c", "d"], "label": 0}] * num_samples)
        data_source_type = "synthetic"

    return ds.select(range(min(len(ds), num_samples))), data_source_type

def format_for_training(ex, data_source_type, dataset_name, tokenizer):
    if data_source_type == "piqa_local" or (data_source_type == "synthetic" and dataset_name == "piqa"): 
        text = f"Goal: {ex['goal']}\nSolution: {ex['sol1'] if ex['label']==0 else ex['sol2']}"
    elif data_source_type == "network" and dataset_name == "hellaswag": 
        text = f"Context: {ex['ctx']}\nEnding: {ex['endings'][int(ex['label'])]}"
    else: text = ex.get('text', "Pad")
    return tokenizer(text, truncation=True, max_length=128, padding="max_length")

def eval_mc_accuracy(model, test_dataset, task_type, accelerator, tokenizer):
    model.eval()
    total_loss = 0; correct = 0; total = 0
    eval_samples = test_dataset.select(range(min(len(test_dataset), 100))) 
    
    with torch.no_grad():
        for ex in eval_samples:
            if task_type == "piqa":
                ctx = f"Goal: {ex['goal']}\nSolution: "
                choices = [str(ex['sol1']), str(ex['sol2'])]
                label = int(ex['label'])
            elif task_type == "hellaswag":
                ctx = f"Context: {ex['ctx']}\nEnding: "
                choices = [str(c) for c in ex['endings']]
                label = int(ex['label'])
            else: return 0.0, 0.0
                
            losses = []
            valid_tokens_list = []
            
            for choice in choices:
                text = ctx + choice
                enc = tokenizer(text, return_tensors="pt").to(accelerator.device)
                ctx_enc = tokenizer(ctx, return_tensors="pt").to(accelerator.device)
                
                logits = model(**enc).logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = enc.input_ids[..., 1:].contiguous()
                
                ctx_len = ctx_enc.input_ids.shape[1]
                if ctx_len - 1 < shift_labels.shape[1]:
                    shift_labels[0, :ctx_len-1] = -100
                
                loss_fct = nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                losses.append(loss.item())
                valid_tokens_list.append((shift_labels[0] != -100).sum().item())
            
            pred = np.argmin(losses)
            if pred == label: correct += 1
            total += 1
            
            if valid_tokens_list[label] > 0:
                total_loss += (losses[label] / valid_tokens_list[label])
                
    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return avg_loss, acc

def train_global_harmonizers(full_model, tokenizer, accelerator, num_steps=200):
    logger.info("⚡ [Server] Pre-training Global Harmonizer Pool...")
    if hasattr(full_model, "model") and hasattr(full_model.model, "layers"): original_layers = full_model.model.layers
    elif hasattr(full_model, "layers"): original_layers = full_model.layers
    else: raise ValueError("Unsupported architecture")
        
    num_layers = len(original_layers)
    ref_type = getattr(original_layers[0], "attention_type", "full")
    global_harmonizers = nn.ModuleList([Harmonizer(full_model.config, 128, ref_type) for _ in range(num_layers)])
    global_harmonizers.to(accelerator.device); global_harmonizers.train()
    optimizer = torch.optim.AdamW(global_harmonizers.parameters(), lr=1e-4)
    loss_fct = nn.KLDivLoss(reduction="batchmean")

    raw_ds = [{"text": "The quick brown fox jumps over the lazy dog. " * 20} for _ in range(400)]
    def tok(ex): return tokenizer(ex["text"], truncation=True, max_length=128, padding="max_length")
    tokenized_ds = [tok(x) for x in raw_ds]
    align_loader = DataLoader(list(zip([torch.tensor(t['input_ids']) for t in tokenized_ds], [torch.tensor(t['attention_mask']) for t in tokenized_ds])), batch_size=4, shuffle=True)

    full_model.to(accelerator.device); iterator = iter(align_loader)
    for param in full_model.parameters(): param.requires_grad = False
    
    for step in range(num_steps):
        try: batch_raw = next(iterator)
        except: iterator = iter(align_loader); batch_raw = next(iterator)
        input_ids = batch_raw[0].to(accelerator.device); attention_mask = batch_raw[1].to(accelerator.device)

        with torch.no_grad():
            full_model.eval()
            teacher_logits = full_model(input_ids=input_ids, attention_mask=attention_mask).logits

        mask = torch.rand(num_layers) > 0.5
        mask[0] = True; mask[-1] = True
        
        mixed_layers = nn.ModuleList()
        for idx in range(num_layers):
            if mask[idx]: mixed_layers.append(original_layers[idx])
            else: mixed_layers.append(global_harmonizers[idx])
            
        if hasattr(full_model, "model") and hasattr(full_model.model, "layers"): full_model.model.layers = mixed_layers
        else: full_model.layers = mixed_layers
            
        full_model.train()
        student_logits = full_model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        if hasattr(full_model, "model") and hasattr(full_model.model, "layers"): full_model.model.layers = original_layers
        else: full_model.layers = original_layers

        loss = loss_fct(torch.nn.functional.log_softmax(student_logits, dim=-1), torch.nn.functional.softmax(teacher_logits, dim=-1))
        loss.backward(); optimizer.step(); optimizer.zero_grad()

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

def evaluate_full_model_plugback(full_model, emulator_model, test_dataset, task_type, accelerator, tokenizer):
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
    peft_full.to(accelerator.device)
    
    avg_loss, acc = eval_mc_accuracy(peft_full, test_dataset, task_type, accelerator, tokenizer)
            
    del peft_full; del temp_full; torch.cuda.empty_cache()
    return avg_loss, acc

class VirtualClient:
    def __init__(self, client_id, train_loader, test_dataset, task_type):
        self.id = client_id; self.train_dataloader = train_loader; self.test_dataset = test_dataset; self.task_type = task_type

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
    raw_ds_a, type_a = load_raw_benchmark_data("piqa", cache_dir=args.cache_dir, num_samples=2000)
    raw_ds_b, type_b = load_raw_benchmark_data("hellaswag", cache_dir=args.cache_dir, num_samples=2000)
    half_clients = args.num_clients // 2
    
    global_test_piqa = raw_ds_a.train_test_split(test_size=0.1)['test']
    global_test_hellaswag = raw_ds_b.train_test_split(test_size=0.1)['test']
    
    for i in range(half_clients):
        sub = raw_ds_a.select(np.array_split(range(len(raw_ds_a)), half_clients)[i])
        train_mapped = sub.map(lambda x: format_for_training(x, type_a, "piqa", tokenizer), remove_columns=sub.column_names)
        clients.append(VirtualClient(i, DataLoader(train_mapped, batch_size=4, collate_fn=default_data_collator, shuffle=True), global_test_piqa, "piqa"))

    for i in range(args.num_clients - half_clients):
        cid = i + half_clients
        sub = raw_ds_b.select(np.array_split(range(len(raw_ds_b)), args.num_clients - half_clients)[i])
        train_mapped = sub.map(lambda x: format_for_training(x, type_b, "hellaswag", tokenizer), remove_columns=sub.column_names)
        clients.append(VirtualClient(cid, DataLoader(train_mapped, batch_size=4, collate_fn=default_data_collator, shuffle=True), global_test_hellaswag, "hellaswag"))

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
            task_type = c_clients[0].task_type
            
            training_model = create_emulator(full_model, global_harmonizers, args.layer_budget)
            training_model.load_state_dict(cluster_global_states[cid], strict=False)
            training_model.to(accelerator.device); training_model.train()
            
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, training_model.parameters()), lr=args.lr)
            global_update = {}
            current_cpu = {k: v.clone().cpu() for k, v in training_model.state_dict().items() if k in cluster_global_states[cid]}
            
            train_loss_sum = 0; train_steps = 0
            
            for client in c_clients:
                training_model.load_state_dict(cluster_global_states[cid], strict=False)
                for step, batch in enumerate(client.train_dataloader):
                    if step >= args.local_steps: break
                    batch = {k: v.to(accelerator.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
                    labels = batch["input_ids"].clone(); labels[labels==tokenizer.pad_token_id] = -100
                    loss = training_model(**batch, labels=labels).loss
                    loss.backward(); optimizer.step(); optimizer.zero_grad()
                    train_loss_sum += loss.item(); train_steps += 1
                
                for key in current_cpu:
                    global_update[key] = global_update.get(key, 0) + (training_model.state_dict()[key].cpu() - current_cpu[key])

            if global_update:
                for key in global_update: cluster_global_states[cid][key] += global_update[key] / len(c_clients)
            
            training_model.load_state_dict(cluster_global_states[cid], strict=False)
            
            train_loss = train_loss_sum / train_steps if train_steps > 0 else 0.0
            plug_mc_loss, plug_acc = evaluate_full_model_plugback(full_model, training_model, c_clients[0].test_dataset, task_type, accelerator, tokenizer)
            
            try: emu_ppl = math.exp(train_loss)
            except: emu_ppl = float('inf')
            try: plug_mc_ppl = math.exp(plug_mc_loss)
            except: plug_mc_ppl = float('inf')
            
            logger.info(f"Cluster {cid} ({task_type}) | Emu Train Loss: {train_loss:.4f} | Plug ACC: {plug_acc:.2%}")
            round_metrics[f"c{cid}_emu_train_loss"] = train_loss
            round_metrics[f"c{cid}_emu_train_ppl"] = emu_ppl
            round_metrics[f"c{cid}_plug_acc"] = plug_acc
            round_metrics[f"c{cid}_plug_mc_ppl"] = plug_mc_ppl
            del training_model; del optimizer; torch.cuda.empty_cache()

        round_metrics["round"] = round_idx + 1
        accelerator.log(round_metrics, step=round_idx + 1)
    accelerator.end_training()

if __name__ == "__main__": main()