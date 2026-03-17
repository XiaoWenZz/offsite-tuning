import argparse
import logging
import sys
import copy
import math
import numpy as np
import time
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

# ==========================================
# Components
# ==========================================
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

# ==========================================
# Helpers & Data
# ==========================================
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

    if dataset_name in ["piqa", "hellaswag", "sciq"]:
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
        elif dataset_name == "hellaswag": ds = datasets.Dataset.from_list([{"ctx": "test", "endings": ["a", "b", "c", "d"], "label": 0}] * num_samples)
        elif dataset_name == "sciq": ds = datasets.Dataset.from_list([{"question": "test", "correct_answer": "a", "distractor1": "b", "distractor2": "c", "distractor3": "d", "label": 0}] * num_samples)
        data_source_type = "synthetic"

    return ds.select(range(min(len(ds), num_samples))), data_source_type

def format_for_training(ex, data_source_type, dataset_name, tokenizer):
    if dataset_name == "piqa": 
        text = f"Goal: {ex.get('goal', 'test')}\nSolution: {ex.get('sol1', 'a') if ex.get('label', 0)==0 else ex.get('sol2', 'b')}"
    elif dataset_name == "hellaswag": 
        endings = ex.get('endings', ["a", "b", "c", "d"])
        text = f"Context: {ex.get('ctx', 'test')}\nEnding: {endings[int(ex.get('label', 0))]}"
    elif dataset_name == "sciq": 
        text = f"Question: {ex.get('question', 'test')}\nAnswer: {ex.get('correct_answer', 'a')}"
    else: text = "Pad"
    return tokenizer(text, truncation=True, max_length=128, padding="max_length")

def eval_mc_accuracy(model, test_dataset, task_type, accelerator, tokenizer, full_eval=False):
    model.eval()
    total_loss = 0; correct = 0; total = 0
    if full_eval:
        eval_samples = test_dataset
        logger.info(f"    🌟 Running FULL EVALUATION on {len(eval_samples)} samples...")
    else:
        eval_samples = test_dataset.select(range(min(len(test_dataset), 100))) 
        
    with torch.no_grad():
        for ex in eval_samples:
            if task_type == "piqa":
                ctx = f"Goal: {ex.get('goal', 'test')}\nSolution: "
                choices = [str(ex.get('sol1', 'a')), str(ex.get('sol2', 'b'))]
                label = int(ex.get('label', 0))
            elif task_type == "hellaswag":
                ctx = f"Context: {ex.get('ctx', 'test')}\nEnding: "
                choices = [str(c) for c in ex.get('endings', ["a", "b", "c", "d"])]
                label = int(ex.get('label', 0))
            elif task_type == "sciq":
                ctx = f"Question: {ex.get('question', 'test')}\nAnswer: "
                choices = [str(ex.get('correct_answer', 'a')), str(ex.get('distractor1', 'b')), str(ex.get('distractor2', 'c')), str(ex.get('distractor3', 'd'))]
                label = 0
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

# ==========================================
# [CORE] ScaleOT: RL + DL Dynamic LayerReplace
# ==========================================
def train_dynamic_layer_replace(full_model, tokenizer, accelerator, cache_dir=None, num_steps=1000):
    logger.info("⚡ [ScaleOT Server] Starting Dynamic LayerReplace (RL + DL) on WikiText...")
    
    if hasattr(full_model, "model") and hasattr(full_model.model, "layers"): original_layers = full_model.model.layers
    elif hasattr(full_model, "layers"): original_layers = full_model.layers
    else: raise ValueError("Unsupported architecture")
        
    num_layers = len(original_layers)
    ref_type = getattr(original_layers[0], "attention_type", "full")
    
    harmonizers = nn.ModuleList([Harmonizer(full_model.config, 128, ref_type) for _ in range(num_layers)])
    harmonizers.to(accelerator.device); harmonizers.train()
    dl_optimizer = torch.optim.AdamW(harmonizers.parameters(), lr=1e-4)
    loss_fct = nn.CrossEntropyLoss()
    
    importance_scores = torch.zeros(num_layers, device=accelerator.device, requires_grad=False)

    try:
        wiki_ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cache_dir)
        wiki_ds = wiki_ds.filter(lambda x: len(x["text"].strip()) > 20)
        raw_ds = [{"text": x["text"]} for x in wiki_ds.select(range(min(len(wiki_ds), 1000)))]
        logger.info(f"✅ Successfully loaded {len(raw_ds)} samples from WikiText-2 for RL Search.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load WikiText: {e}. Falling back to dummy text.")
        raw_ds = [{"text": "The knowledge of science and common sense is fundamental to AI. " * 10} for _ in range(400)]

    def tok(ex): return tokenizer(ex["text"], truncation=True, max_length=128, padding="max_length")
    tokenized_ds = [tok(x) for x in raw_ds]
    
    split_idx = int(len(tokenized_ds) * 0.75)
    train_dl = DataLoader([{"input_ids": torch.tensor(t['input_ids'])} for t in tokenized_ds[:split_idx]], batch_size=4, shuffle=True)
    val_dl = DataLoader([{"input_ids": torch.tensor(t['input_ids'])} for t in tokenized_ds[split_idx:]], batch_size=4, shuffle=True)

    full_model.to(accelerator.device)
    train_iter = iter(train_dl)
    val_iter = iter(val_dl)
    
    for param in full_model.parameters(): param.requires_grad = False
    
    for step in range(num_steps):
        try: batch_t = next(train_iter)
        except: train_iter = iter(train_dl); batch_t = next(train_iter)
        input_ids = batch_t["input_ids"].to(accelerator.device)

        current_probs = torch.sigmoid(importance_scores)
        rand_u = torch.rand(num_layers, device=accelerator.device)
        sampled_probs = rand_u * current_probs
        
        mask_dl = torch.zeros(num_layers, dtype=torch.bool, device=accelerator.device)
        for i in range(0, num_layers, 4):
            group_p = sampled_probs[i:i+4]
            if len(group_p) > 0: mask_dl[i:i+4] = group_p >= torch.median(group_p)
        mask_dl[0] = True; mask_dl[-1] = True 
        
        mixed_layers = nn.ModuleList([original_layers[idx] if mask_dl[idx] else harmonizers[idx] for idx in range(num_layers)])
        if hasattr(full_model, "model") and hasattr(full_model.model, "layers"): full_model.model.layers = mixed_layers
        else: full_model.layers = mixed_layers
            
        full_model.train()
        outputs = full_model(input_ids=input_ids)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_dl_val = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_dl_val.backward(); dl_optimizer.step(); dl_optimizer.zero_grad()

        try: batch_v = next(val_iter)
        except: val_iter = iter(val_dl); batch_v = next(val_iter)
        val_input_ids = batch_v["input_ids"].to(accelerator.device)
        
        full_model.eval()
        losses_v = []
        masks_v = []
        Nc = 3 
        with torch.no_grad():
            for _ in range(Nc):
                rand_u_v = torch.rand(num_layers, device=accelerator.device)
                sampled_probs_v = rand_u_v * current_probs
                mask_j = torch.zeros(num_layers, dtype=torch.bool, device=accelerator.device)
                for i in range(0, num_layers, 4):
                    group_p = sampled_probs_v[i:i+4]
                    if len(group_p) > 0: mask_j[i:i+4] = group_p >= torch.median(group_p)
                mask_j[0] = True; mask_j[-1] = True
                
                mixed_j = nn.ModuleList([original_layers[idx] if mask_j[idx] else harmonizers[idx] for idx in range(num_layers)])
                if hasattr(full_model, "model") and hasattr(full_model.model, "layers"): full_model.model.layers = mixed_j
                else: full_model.layers = mixed_j
                
                out_v = full_model(input_ids=val_input_ids)
                shift_logits_v = out_v.logits[..., :-1, :].contiguous()
                shift_labels_v = val_input_ids[..., 1:].contiguous()
                loss_j = loss_fct(shift_logits_v.view(-1, shift_logits_v.size(-1)), shift_labels_v.view(-1)).item()
                losses_v.append(loss_j)
                masks_v.append(mask_j)
                
        exp_losses = [math.exp(-l) for l in losses_v]
        baseline = sum(exp_losses) / Nc
        
        for j in range(Nc):
            r_j = exp_losses[j] - baseline
            for i in range(num_layers):
                if masks_v[j][i]:
                    importance_scores[i] += 0.1 * r_j * current_probs[i] * (1 - current_probs[i])
        
        if hasattr(full_model, "model") and hasattr(full_model.model, "layers"): full_model.model.layers = original_layers
        else: full_model.layers = original_layers
        
        if step % 50 == 0: logger.info(f"   RL+DL Step {step} | DL Loss: {loss_dl_val.item():.4f} | RL Reward Base: {baseline:.4f}")

    harmonizers.cpu()
    logger.info("✅ ScaleOT Important Scores and Harmonizers Ready.")
    return importance_scores.cpu().numpy(), harmonizers

# ==========================================
# [CORE] ScaleOT Emulator Creation (NO SRC)
# ==========================================
def create_scaleot_emulator_no_src(full_model, importance_scores, harmonizers, budget_adapter=6, verbose=False):
    emulator = copy.deepcopy(full_model)
    layers = get_module_layers(emulator)
    num_layers = len(layers)
    
    harmonizer_indices = set()
    
    # 🔥 三明治约束：强制锁定 0, 1 层和 倒数两层
    adapter_indices = {0, 1, num_layers - 2, num_layers - 1} 
    rem_adapters = budget_adapter - 4
    
    if rem_adapters > 0:
        middle_layers = list(range(2, num_layers - 2))
        chunks = np.array_split(middle_layers, rem_adapters)
        for chunk in chunks:
            if len(chunk) == 0: continue
            chunk_scores = [(idx, importance_scores[idx]) for idx in chunk]
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            adapter_indices.add(int(chunk_scores[0][0]))
            
    for i in range(num_layers):
        if i not in adapter_indices:
            harmonizer_indices.add(i)
                
    if verbose:
        logger.info(f"    - Adapters (RL Sandwich Block-wise): {sorted(list(adapter_indices))}")
        logger.info(f"    - Harmonizers (All Others): {sorted(list(harmonizer_indices))}")
                
    new_layers = []
    layer_map = {}
    
    for curr in range(num_layers):
        if curr in adapter_indices:
            l = layers[curr]
            l.original_layer_idx = curr
            new_layers.append(l)
            layer_map[curr] = curr
        else:
            h = copy.deepcopy(harmonizers[curr])
            h.original_layer_idx = None
            new_layers.append(h)
            
    set_module_layers(emulator, nn.ModuleList(new_layers))
    emulator.config.use_cache = False
    emulator.config.layer_map = layer_map 
    
    for param in emulator.parameters(): param.requires_grad = False
    # 🔥 全量 LoRA
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    emulator = get_peft_model(emulator, peft_config)
    if hasattr(emulator, "config"): emulator.config.layer_map = layer_map
    
    for name, param in emulator.named_parameters():
        if "Harmonizer" in name or "lora_" in name: param.requires_grad = True
        else: param.requires_grad = False
        
    return emulator

def evaluate_full_model_plugback(full_model, emulator_model, test_dataset, task_type, accelerator, tokenizer, full_eval=False):
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
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    peft_full = get_peft_model(temp_full, peft_config)
    peft_full.load_state_dict(adapter_lora_state, strict=False)
    peft_full.to(accelerator.device)
    
    avg_loss, acc = eval_mc_accuracy(peft_full, test_dataset, task_type, accelerator, tokenizer, full_eval)
            
    del peft_full; del temp_full; torch.cuda.empty_cache()
    return avg_loss, acc

class VirtualClient:
    def __init__(self, client_id, train_loader, test_dataset, task_type):
        self.id = client_id; self.train_dataloader = train_loader; self.test_dataset = test_dataset; self.task_type = task_type

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num_clients", type=int, default=12)
    parser.add_argument("--num_clusters", type=int, default=3)
    parser.add_argument("--layer_budget", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="fedrole_benchmark")
    parser.add_argument("--wandb_run_name", type=str, default="scaleot_nosrc_run")
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
    clients_per_task = args.num_clients // 3
    
    raw_ds_a, type_a = load_raw_benchmark_data("piqa", cache_dir=args.cache_dir, num_samples=2000)
    raw_ds_b, type_b = load_raw_benchmark_data("hellaswag", cache_dir=args.cache_dir, num_samples=2000)
    raw_ds_c, type_c = load_raw_benchmark_data("sciq", cache_dir=args.cache_dir, num_samples=2000)
    
    global_test_piqa = raw_ds_a.train_test_split(test_size=0.1)['test']
    global_test_hellaswag = raw_ds_b.train_test_split(test_size=0.1)['test']
    global_test_sciq = raw_ds_c.train_test_split(test_size=0.1)['test']
    
    for i in range(clients_per_task):
        sub = raw_ds_a.select(np.array_split(range(len(raw_ds_a)), clients_per_task)[i])
        train_mapped = sub.map(lambda x: format_for_training(x, type_a, "piqa", tokenizer), remove_columns=sub.column_names)
        clients.append(VirtualClient(i, DataLoader(train_mapped, batch_size=4, collate_fn=default_data_collator, shuffle=True), global_test_piqa, "piqa"))

    for i in range(clients_per_task):
        sub = raw_ds_b.select(np.array_split(range(len(raw_ds_b)), clients_per_task)[i])
        train_mapped = sub.map(lambda x: format_for_training(x, type_b, "hellaswag", tokenizer), remove_columns=sub.column_names)
        clients.append(VirtualClient(i + clients_per_task, DataLoader(train_mapped, batch_size=4, collate_fn=default_data_collator, shuffle=True), global_test_hellaswag, "hellaswag"))

    for i in range(clients_per_task):
        sub = raw_ds_c.select(np.array_split(range(len(raw_ds_c)), clients_per_task)[i])
        train_mapped = sub.map(lambda x: format_for_training(x, type_c, "sciq", tokenizer), remove_columns=sub.column_names)
        clients.append(VirtualClient(i + 2 * clients_per_task, DataLoader(train_mapped, batch_size=4, collate_fn=default_data_collator, shuffle=True), global_test_sciq, "sciq"))

    clusters = {i: clients[i*(args.num_clients//args.num_clusters):(i+1)*(args.num_clients//args.num_clusters)] for i in range(args.num_clusters)}

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    start_time = time.time()

    full_model.to(accelerator.device)
    importance_scores, global_harmonizers = train_dynamic_layer_replace(full_model, tokenizer, accelerator, cache_dir=args.cache_dir, num_steps=1000)

    torch.cuda.synchronize()
    end_time = time.time()
    peak_vram_gb = torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 3)
    search_time = end_time - start_time
    
    logger.info("==================================================")
    logger.info(f"⏱️ [ScaleOT Profiling] RL Search Time: {search_time:.2f} seconds")
    logger.info(f"💾 [ScaleOT Profiling] Peak VRAM: {peak_vram_gb:.2f} GB")
    logger.info("==================================================")
    
    accelerator.log({"search_time_seconds": search_time, "search_peak_vram_gb": peak_vram_gb}, step=0)

    full_model.cpu()
    global_harmonizers.cpu()

    logger.info("=== [Step 3] Assembling ScaleOT Emulator (RL Sandwich Block-wise + NO SRC) ===")
    scaleot_emulator = create_scaleot_emulator_no_src(full_model, importance_scores, global_harmonizers, args.layer_budget, verbose=True)
    trainable_keys = get_trainable_keys(scaleot_emulator)
    
    cluster_global_states = {}
    for cid in range(args.num_clusters):
        cluster_global_states[cid] = {k: v.clone().cpu() for k, v in scaleot_emulator.state_dict().items() if k in trainable_keys}

    logger.info("=== [Step 4] Federated Training ===")
    for round_idx in range(args.rounds):
        logger.info(f"--- Round {round_idx + 1} ---")
        round_metrics = {}
        for cid, c_clients in clusters.items():
            if not c_clients: continue
            
            task_type = c_clients[0].task_type
            
            training_model = create_scaleot_emulator_no_src(full_model, importance_scores, global_harmonizers, args.layer_budget, verbose=False)
            training_model.load_state_dict(cluster_global_states[cid], strict=False)
            training_model.to(accelerator.device); training_model.train()
            
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, training_model.parameters()), lr=args.lr)
            global_update = {}
            current_cpu = {k: v.detach().clone().cpu() for k, v in training_model.state_dict().items() if k in cluster_global_states[cid]}
            
            train_loss_sum = 0; train_steps = 0
            
            for client in c_clients:
                training_model.load_state_dict(cluster_global_states[cid], strict=False)
                for step, batch in enumerate(client.train_dataloader):
                    if step >= args.local_steps: break
                    batch = {k: v.to(accelerator.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
                    labels_tensor = batch["input_ids"].clone(); labels_tensor[labels_tensor==tokenizer.pad_token_id] = -100
                    loss = training_model(**batch, labels=labels_tensor).loss
                    loss.backward(); optimizer.step(); optimizer.zero_grad()
                    train_loss_sum += loss.item(); train_steps += 1
                
                for key in current_cpu:
                    global_update[key] = global_update.get(key, 0) + (training_model.state_dict()[key].detach().cpu() - current_cpu[key])

            if global_update:
                for key in global_update: cluster_global_states[cid][key] += global_update[key] / len(c_clients)
            
            training_model.load_state_dict(cluster_global_states[cid], strict=False)
            
            train_loss = train_loss_sum / train_steps if train_steps > 0 else 0.0
            plug_mc_loss, plug_acc = evaluate_full_model_plugback(full_model, training_model, c_clients[0].test_dataset, task_type, accelerator, tokenizer, full_eval=False)
            
            try: emu_ppl = math.exp(train_loss)
            except: emu_ppl = float('inf')
            try: plug_mc_ppl = math.exp(plug_mc_loss)
            except: plug_mc_ppl = float('inf')
            
            logger.info(f"Task {task_type} (Cluster {cid}) | Emu Train Loss: {train_loss:.4f} | Plug ACC: {plug_acc:.2%}")
            
            round_metrics[f"{task_type}_emu_train_loss"] = train_loss
            round_metrics[f"{task_type}_emu_train_ppl"] = emu_ppl
            round_metrics[f"{task_type}_plug_acc"] = plug_acc
            round_metrics[f"{task_type}_plug_mc_ppl"] = plug_mc_ppl
            del training_model; del optimizer; torch.cuda.empty_cache()

        round_metrics["round"] = round_idx + 1
        accelerator.log(round_metrics, step=round_idx + 1)
        
    logger.info("==================================================")
    logger.info("🏆 Starting Final Full Evaluation...")
    final_metrics = {}
    for cid, c_clients in clusters.items():
        if not c_clients: continue
        task_type = c_clients[0].task_type
        
        final_model = create_scaleot_emulator_no_src(full_model, importance_scores, global_harmonizers, args.layer_budget, verbose=False)
        final_model.load_state_dict(cluster_global_states[cid], strict=False)
        final_model.to(accelerator.device)
        
        final_loss, final_acc = evaluate_full_model_plugback(full_model, final_model, c_clients[0].test_dataset, task_type, accelerator, tokenizer, full_eval=True)
        try: final_ppl = math.exp(final_loss)
        except: final_ppl = float('inf')
        
        logger.info(f"🏅 Task {task_type} FINAL FULL EVAL | Plug PPL: {final_ppl:.2f} | Plug ACC: {final_acc:.2%}")
        
        final_metrics[f"final_{task_type}_plug_ppl"] = final_ppl
        final_metrics[f"final_{task_type}_plug_acc"] = final_acc
        del final_model; torch.cuda.empty_cache()

    accelerator.log(final_metrics)
    accelerator.end_training()

if __name__ == "__main__": main()