import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import datasets
from transformers import default_data_collator
from peft import get_peft_model, LoraConfig, TaskType
import logging
import os

logger = logging.getLogger(__name__)

# ==========================================
# 1. 核心组件: Harmonizer & SRC
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
        
        # [Critical Fix] for Qwen2 compatibility
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
            U, S, Vt = torch.linalg.svd(W.cpu(), full_matrices=False)
            target_rank = max(1, int(min(W.shape) * rank_ratio))
            U_r = U[:, :target_rank]; S_r = torch.diag(S[:target_rank]); Vt_r = Vt[:target_rank, :]
            W_approx = (U_r @ S_r @ Vt_r).to(linear_layer.weight.device).to(linear_layer.weight.dtype)
            linear_layer.weight.data = W_approx
        except Exception: pass 

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)

# ==========================================
# 2. 工具函数
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
            model.model.layers = new_layers; model.config.num_hidden_layers = len(new_layers); return
    if hasattr(model, "layers"):
        model.layers = new_layers; model.config.num_hidden_layers = len(new_layers); return
    raise ValueError(f"Unknown architecture for setting layers: {type(model)}")

def get_trainable_keys(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}

# ==========================================
# 3. Emulator 构建器
# ==========================================
def build_emulator(full_model, selected_indices, use_src=True, src_ratio=0.5):
    emulator = copy.deepcopy(full_model)
    layers = get_module_layers(emulator)
    total = len(layers)
    new_layers = []
    
    # 获取第一层的 attention_type 作为参考
    ref_attn_type = getattr(layers[0], "attention_type", "full")
    
    selected_set = set(selected_indices)
    curr = 0
    
    while curr < total:
        if curr in selected_set:
            l = layers[curr]
            l.original_layer_idx = curr
            new_layers.append(l)
            curr += 1
        else:
            if use_src and curr % 2 == 0: 
                new_layers.append(SRCLayer(layers[curr], curr, src_ratio))
                curr += 1
            else:
                gap = 0
                while (curr + gap < total) and (curr + gap not in selected_set): gap += 1
                # 传入 attention_type 防止 Qwen2 报错
                new_layers.append(Harmonizer(full_model.config, 128, target_attention_type=ref_attn_type))
                curr += gap
                
    set_module_layers(emulator, nn.ModuleList(new_layers))
    emulator.config.use_cache = False
    
    layer_map = {}
    for i, layer in enumerate(new_layers):
        if hasattr(layer, "original_layer_idx") and layer.original_layer_idx is not None:
            layer_map[i] = layer.original_layer_idx
    emulator.config.layer_map = layer_map
    
    for param in emulator.parameters(): param.requires_grad = False
    
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    emulator = get_peft_model(emulator, peft_config)
    
    for name, param in emulator.named_parameters():
        if "Harmonizer" in name or "lora_" in name: param.requires_grad = True
        
    if hasattr(emulator, "config"): emulator.config.layer_map = layer_map
    return emulator

# ==========================================
# 4. 服务端对齐
# ==========================================
def server_side_alignment(full_model, emulator, tokenizer, accelerator, num_steps=50):
    logger.info("⚡ [Server] Running Alignment (Warm-up)...")
    raw_ds = [{"text": "The quick brown fox jumps over the lazy dog. " * 20} for _ in range(100)]
    def tok(ex): return tokenizer(ex["text"], truncation=True, max_length=128, padding="max_length")
    tokenized_ds = [tok(x) for x in raw_ds]
    input_ids = torch.tensor([t['input_ids'] for t in tokenized_ds])
    attention_mask = torch.tensor([t['attention_mask'] for t in tokenized_ds])
    loader = DataLoader(list(zip(input_ids, attention_mask)), batch_size=4, shuffle=True, collate_fn=lambda x: {"input_ids": torch.stack([i[0] for i in x]), "attention_mask": torch.stack([i[1] for i in x])})

    full_model.eval(); full_model.to(accelerator.device)
    emulator.train(); emulator.to(accelerator.device)
    
    params = [p for n, p in emulator.named_parameters() if "Harmonizer" in n]
    if not params: return
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    loss_fct = nn.KLDivLoss(reduction="batchmean")

    iter_dl = iter(loader)
    for _ in range(num_steps):
        try: batch = next(iter_dl)
        except: iter_dl = iter(loader); batch = next(iter_dl)
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        with torch.no_grad(): t_logits = full_model(**batch).logits
        s_logits = emulator(**batch).logits
        loss = loss_fct(torch.nn.functional.log_softmax(s_logits, dim=-1), torch.nn.functional.softmax(t_logits, dim=-1))
        loss.backward(); optimizer.step(); optimizer.zero_grad()
    
    for name, param in emulator.named_parameters():
        if "lora_" in name: param.requires_grad = True
    torch.cuda.empty_cache()

# ==========================================
# 5. 安全回填评估
# ==========================================
def evaluate_plugback(full_model, emulator, dataloader, accelerator, tokenizer):
    full_model.eval()
    emu_state = emulator.state_dict()
    adapter_state = {}
    
    idx_map = {}
    if hasattr(emulator, "config") and hasattr(emulator.config, "layer_map"):
        idx_map = emulator.config.layer_map
    elif hasattr(emulator, "base_model") and hasattr(emulator.base_model.model, "config"):
        idx_map = emulator.base_model.model.config.layer_map
        
    for k, v in emu_state.items():
        if "lora_" in k:
            parts = k.split('.')
            try:
                if 'layers' in parts:
                    idx_pos = parts.index('layers') + 1
                    emu_idx = int(parts[idx_pos])
                    if emu_idx in idx_map:
                        parts[idx_pos] = str(idx_map[emu_idx])
                        adapter_state[".".join(parts)] = v.cpu()
            except: continue

    temp_full = copy.deepcopy(full_model)
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    peft_full = get_peft_model(temp_full, peft_config)
    peft_full.load_state_dict(adapter_state, strict=False)
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

# ==========================================
# 6. 本地 PIQA 加载器
# ==========================================
def load_local_piqa(tokenizer, num_samples=1000):
    BASE_DIR = "/data/xiaowen/piqa_local/physicaliqa-train-dev"
    DATA_PATH = os.path.join(BASE_DIR, "train.jsonl")
    LABEL_PATH = os.path.join(BASE_DIR, "train-labels.lst")
    
    if os.path.exists(DATA_PATH):
        logger.info(f"Using Local PIQA: {DATA_PATH}")
        ds = datasets.load_dataset("json", data_files={"train": DATA_PATH}, split="train")
        with open(LABEL_PATH, "r") as f: labels = [int(line.strip()) for line in f.readlines()]
        min_len = min(len(ds), len(labels))
        ds = ds.select(range(min_len))
        ds = ds.add_column("label", labels[:min_len])
        ds = ds.select(range(min(len(ds), num_samples)))
        
        def fmt(ex):
            correct = ex['sol1'] if ex['label'] == 0 else ex['sol2']
            return tokenizer(f"Goal: {ex['goal']}\nSolution: {correct}", truncation=True, max_length=128, padding="max_length")
        
        return ds.map(fmt, remove_columns=ds.column_names)
    else:
        logger.warning("Local PIQA not found, using Synthetic!")
        ds = datasets.Dataset.from_list([{"text": "Synthetic data."}] * num_samples)
        return ds.map(lambda x: tokenizer(x['text'], truncation=True, max_length=128, padding="max_length"))