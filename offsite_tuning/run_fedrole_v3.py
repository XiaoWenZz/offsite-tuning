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
        
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return x + residual

# ==========================================
# Helpers (基于 v2)
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

def compute_layer_sensitivity(model):
    layers = get_module_layers(model)
    sensitivity = np.zeros(len(layers))
    for i, layer in enumerate(layers):
        # 使用 Taylor-1 (权重绝对值 * 梯度绝对值) 替代纯梯度
        score = sum((p.data * p.grad.detach()).abs().sum().item() 
                    for p in layer.parameters() if p.grad is not None)
        sensitivity[i] = score
    # 归一化防溢出
    return sensitivity / (np.linalg.norm(sensitivity) + 1e-8)

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
# Global Pool & Emulator Builders
# ==========================================
def train_global_harmonizers(full_model, tokenizer, accelerator, cache_dir=None, num_steps=200):
    logger.info("⚡ [Server] Pre-training Global Harmonizer Pool on WikiText (Stochastic Swap)...")
    if hasattr(full_model, "model") and hasattr(full_model.model, "layers"): original_layers = full_model.model.layers
    elif hasattr(full_model, "layers"): original_layers = full_model.layers
    else: raise ValueError("Unsupported architecture")
        
    num_layers = len(original_layers)
    ref_type = getattr(original_layers[0], "attention_type", "full")
    global_harmonizers = nn.ModuleList([Harmonizer(full_model.config, 128, ref_type) for _ in range(num_layers)])
    global_harmonizers.to(accelerator.device); global_harmonizers.train()
    
    optimizer = torch.optim.AdamW(global_harmonizers.parameters(), lr=1e-4)
    loss_fct = nn.KLDivLoss(reduction="batchmean")

    # ==========================================
    # 📚 加载真正的 WikiText-2 作为公共预训练数据
    # ==========================================
    try:
        wiki_ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cache_dir)
        # 过滤掉太短的空白行或无意义行
        wiki_ds = wiki_ds.filter(lambda x: len(x["text"].strip()) > 20)
        # 选取 1000 条进行轻量级预热
        raw_ds = [{"text": x["text"]} for x in wiki_ds.select(range(min(len(wiki_ds), 1000)))]
        logger.info(f"✅ Successfully loaded {len(raw_ds)} samples from WikiText-2.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load WikiText: {e}. Falling back to dummy text.")
        raw_ds = [{"text": "The knowledge of science and common sense is fundamental to AI. " * 10} for _ in range(400)]

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

def create_custom_emulator(full_model, adapter_indices, global_harmonizers):
    emulator = copy.deepcopy(full_model)
    layers = get_module_layers(emulator)
    total_layers = len(layers)
    
    new_layers = []; layer_map = {}
    for curr in range(total_layers):
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
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
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
    parser.add_argument("--dataset_name", type=str, default="mixed_piqa_hellaswag_sciq")
    parser.add_argument("--num_clients", type=int, default=12)
    parser.add_argument("--num_clusters", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--layer_budget", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_steps", type=int, default=10)
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
    full_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    
    # 1. Load Data
    clients = []
    clients_per_task = args.num_clients // 3
    
    raw_ds_a, type_a = load_raw_benchmark_data("piqa", cache_dir=args.cache_dir, num_samples=2000)
    raw_ds_b, type_b = load_raw_benchmark_data("hellaswag", cache_dir=args.cache_dir, num_samples=2000)
    raw_ds_c, type_c = load_raw_benchmark_data("sciq", cache_dir=args.cache_dir, num_samples=2000)
    
    global_test_piqa = raw_ds_a.train_test_split(test_size=0.1)['test']
    global_test_hellaswag = raw_ds_b.train_test_split(test_size=0.1)['test']
    global_test_sciq = raw_ds_c.train_test_split(test_size=0.1)['test']
    
    for i in range(clients_per_task):
        cid = i
        sub = raw_ds_a.select(np.array_split(range(len(raw_ds_a)), clients_per_task)[i])
        train_mapped = sub.map(lambda x: format_for_training(x, type_a, "piqa", tokenizer), remove_columns=sub.column_names)
        clients.append(VirtualClient(cid, DataLoader(train_mapped, batch_size=4, collate_fn=default_data_collator, shuffle=True), global_test_piqa, "piqa"))

    for i in range(clients_per_task):
        cid = i + clients_per_task
        sub = raw_ds_b.select(np.array_split(range(len(raw_ds_b)), clients_per_task)[i])
        train_mapped = sub.map(lambda x: format_for_training(x, type_b, "hellaswag", tokenizer), remove_columns=sub.column_names)
        clients.append(VirtualClient(cid, DataLoader(train_mapped, batch_size=4, collate_fn=default_data_collator, shuffle=True), global_test_hellaswag, "hellaswag"))

    for i in range(clients_per_task):
        cid = i + 2 * clients_per_task
        sub = raw_ds_c.select(np.array_split(range(len(raw_ds_c)), clients_per_task)[i])
        train_mapped = sub.map(lambda x: format_for_training(x, type_c, "sciq", tokenizer), remove_columns=sub.column_names)
        clients.append(VirtualClient(cid, DataLoader(train_mapped, batch_size=4, collate_fn=default_data_collator, shuffle=True), global_test_sciq, "sciq"))

    # =================================================================
    # [探针开始：FedRole 梯度感知开销]
    # =================================================================
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    start_time = time.time()

    # 2. Get Sensitivities & Cluster
    logger.info("🔍 [Sensing] Calculating Task Sensitivities...")
    sensitivity_vectors = []
    full_model.to(accelerator.device); full_model.train()
    init_state = {k: v.clone().cpu() for k, v in full_model.state_dict().items()}
    for client in clients:
        full_model.load_state_dict(init_state); full_model.zero_grad()
        avg_grad = np.zeros(len(get_module_layers(full_model)))
        valid = 0; iter_loader = iter(client.train_dataloader)
        for _ in range(3):
            try: batch = next(iter_loader); batch = {k: v.to(accelerator.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            except: break
            full_model(**batch, labels=batch["input_ids"]).loss.backward()
            avg_grad += compute_layer_sensitivity(full_model)
            valid += 1; full_model.zero_grad()
        if valid>0: avg_grad/=valid
        sensitivity_vectors.append(avg_grad)
    full_model.cpu()
    labels = KMeans(n_clusters=args.num_clusters, random_state=42).fit_predict(np.stack(sensitivity_vectors))
    
    # =================================================================
    # [探针结束：FedRole 梯度感知开销]
    # =================================================================
    torch.cuda.synchronize()
    end_time = time.time()
    peak_vram_gb = torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 3)
    search_time = end_time - start_time
    
    logger.info("==================================================")
    logger.info(f"⏱️ [FedRole Profiling] Gradient Sensing Time: {search_time:.2f} seconds")
    logger.info(f"💾 [FedRole Profiling] Peak VRAM: {peak_vram_gb:.2f} GB")
    logger.info("==================================================")
    
    # 将探针数据记录到 WandB 步数 0
    accelerator.log({"search_time_seconds": search_time, "search_peak_vram_gb": peak_vram_gb}, step=0)

    clusters = {i: [clients[j] for j, l in enumerate(labels) if l == i] for i in range(args.num_clusters)}

    # 3. Pre-train Global Harmonizers
    global_harmonizers = train_global_harmonizers(full_model, tokenizer, accelerator, cache_dir=args.cache_dir, num_steps=200)

    # 4. Initialize Emulators
    logger.info("=== [Step 4] Assembling Task-Aware Custom Emulators ===")
    cluster_global_states = {}; cluster_selected_indices = {}
    total_layers = len(get_module_layers(full_model))
    
    for cid in range(args.num_clusters):
        c_idxs = [i for i, l in enumerate(labels) if l == cid]
        if not c_idxs: continue
        sens = np.mean(np.stack(sensitivity_vectors)[c_idxs], axis=0)
        
        adapter_indices = {0, total_layers - 1} 
        rem = args.layer_budget - 2
        if rem > 0:
            middle_layers = list(range(1, total_layers - 1))
            chunks = np.array_split(middle_layers, rem)
            for chunk in chunks:
                if len(chunk) == 0: continue
                best_idx = chunk[np.argmax(sens[chunk])]
                adapter_indices.add(best_idx)
                
        adapter_indices = sorted(list(adapter_indices))
        cluster_selected_indices[cid] = adapter_indices
        
        task_label = clusters[cid][0].task_type
        logger.info(f"👉 Cluster {cid} ({task_label}) Selected Layers: {adapter_indices}")
        
        emu = create_custom_emulator(full_model, adapter_indices, global_harmonizers)
        trainable_keys = get_trainable_keys(emu)
        cluster_global_states[cid] = {k: v.clone().cpu() for k, v in emu.state_dict().items() if k in trainable_keys}
        del emu

    # 5. FL Loop
    logger.info("=== [Step 5] Federated Training ===")
    for round_idx in range(args.rounds):
        logger.info(f"--- Round {round_idx + 1} ---")
        round_metrics = {}
        for cid, c_clients in clusters.items():
            if not c_clients: continue
            
            task_type = c_clients[0].task_type 
            
            temp_model = create_custom_emulator(full_model, cluster_selected_indices[cid], global_harmonizers)
            temp_model.load_state_dict(cluster_global_states[cid], strict=False)
            temp_model.to(accelerator.device); temp_model.train()
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, temp_model.parameters()), lr=args.lr)
            
            global_update = {}
            current_cpu = {k: v.clone().cpu() for k, v in temp_model.state_dict().items() if k in cluster_global_states[cid]}
            
            train_loss_sum = 0; train_steps = 0
            
            for client in c_clients:
                temp_model.load_state_dict(cluster_global_states[cid], strict=False)
                for step, batch in enumerate(client.train_dataloader):
                    if step >= args.local_steps: break
                    batch = {k: v.to(accelerator.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
                    labels_tensor = batch["input_ids"].clone(); labels_tensor[labels_tensor==tokenizer.pad_token_id] = -100
                    loss = temp_model(**batch, labels=labels_tensor).loss
                    loss.backward(); optimizer.step(); optimizer.zero_grad()
                    train_loss_sum += loss.item(); train_steps += 1
                
                for key in current_cpu:
                    global_update[key] = global_update.get(key, 0) + (temp_model.state_dict()[key].cpu() - current_cpu[key])
            
            if global_update:
                for key in global_update: cluster_global_states[cid][key] += global_update[key] / len(c_clients)
            
            temp_model.load_state_dict(cluster_global_states[cid], strict=False)
            
            train_loss = train_loss_sum / train_steps if train_steps > 0 else 0.0
            plug_mc_loss, plug_acc = evaluate_full_model_plugback(full_model, temp_model, c_clients[0].test_dataset, task_type, accelerator, tokenizer, full_eval=False)
            
            try: emu_ppl = math.exp(train_loss)
            except: emu_ppl = float('inf')
            try: plug_mc_ppl = math.exp(plug_mc_loss)
            except: plug_mc_ppl = float('inf')
            
            logger.info(f"Task {task_type} (Cluster {cid}) | Emu Train Loss: {train_loss:.4f} | Plug ACC: {plug_acc:.2%}")
            
            round_metrics[f"{task_type}_emu_train_loss"] = train_loss
            round_metrics[f"{task_type}_emu_train_ppl"] = emu_ppl
            round_metrics[f"{task_type}_plug_acc"] = plug_acc
            round_metrics[f"{task_type}_plug_mc_ppl"] = plug_mc_ppl
            del temp_model; del optimizer; torch.cuda.empty_cache()

        round_metrics["round"] = round_idx + 1
        accelerator.log(round_metrics, step=round_idx + 1)
        
    logger.info("==================================================")
    logger.info("🏆 Starting Final Full Evaluation...")
    final_metrics = {}
    
    for cid, c_clients in clusters.items():
        if not c_clients: continue
        task_type = c_clients[0].task_type
        
        final_model = create_custom_emulator(full_model, cluster_selected_indices[cid], global_harmonizers)
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