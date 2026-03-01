import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, set_seed
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from accelerate.logging import get_logger
import logging
import os
import math

logger = get_logger(__name__)

# ==========================================
# 1. Components: Harmonizer & SRC
# ==========================================
class Harmonizer(nn.Module):
    def __init__(self, config, rank=64): # Paper uses rank 64 for medium models [cite: 274]
        super().__init__()
        self.input_dim = config.hidden_size
        self.rank = rank
        # "Simple low-rank FFN with ReLU" [cite: 274]
        self.down_proj = nn.Linear(self.input_dim, self.rank)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(self.rank, self.input_dim)
        
        # Zero Init to behave like Identity initially
        nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.zeros_(self.down_proj.bias)

    def forward(self, hidden_states, *args, **kwargs):
        x = hidden_states
        if isinstance(x, tuple): x = x[0]
        return x + self.up_proj(self.activation(self.down_proj(x)))

def apply_src_to_emulator(emulator, beta=0.8):
    """
    Selective Rank Compression (SRC).
    Paper: "Rank compress MHSA layers... when ratio > 0.6" [cite: 255, 256]
    """
    logger.info(f"✂️ Applying SRC (SVD) to MHSA layers with ratio {beta}...")
    
    # Identify layers
    if hasattr(emulator, "model"): layers = emulator.model.layers
    elif hasattr(emulator, "base_model"): layers = emulator.base_model.model.layers
    else: layers = emulator.layers

    for layer in layers:
        # Only compress MHSA (Self-Attention), keep FFN intact 
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            # Target Q, K, V, O projections
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                if hasattr(attn, name):
                    module = getattr(attn, name)
                    W = module.weight.data.float()
                    # SVD
                    try:
                        U, S, Vt = torch.linalg.svd(W.cpu(), full_matrices=False)
                        # Keep top (1-beta) * 100% of rank, or reduce rank by beta
                        # Paper says "rank reduction ratio beta", so we keep (1-beta)
                        target_rank = max(1, int(min(W.shape) * (1 - beta)))
                        
                        U_r = U[:, :target_rank]
                        S_r = torch.diag(S[:target_rank])
                        Vt_r = Vt[:target_rank, :]
                        
                        W_approx = (U_r @ S_r @ Vt_r).to(module.weight.device).to(module.weight.dtype)
                        module.weight.data = W_approx
                    except Exception as e:
                        logger.warning(f"SVD failed for {name}: {e}")

# ==========================================
# 2. The RL Engine (Importance Learner)
# ==========================================
class DynamicLayerReplaceEngine(nn.Module):
    def __init__(self, full_model, harmonizer_rank=64):
        super().__init__()
        self.config = full_model.config
        
        # 1. Extract Layers
        if hasattr(full_model, "model"): self.raw_layers = full_model.model.layers
        else: self.raw_layers = full_model.layers
        self.num_layers = len(self.raw_layers)
        
        # 2. Create Harmonizers (one for every layer) [cite: 139]
        self.harmonizers = nn.ModuleList([
            Harmonizer(self.config, rank=harmonizer_rank) for _ in range(self.num_layers)
        ])
        
        # 3. Learnable Importance Scores [cite: 139]
        # Initialize to 0.0 -> Sigmoid(0) = 0.5 (Neutral importance)
        self.importance_scores = nn.Parameter(torch.zeros(self.num_layers))
        
        # Freeze Raw Model (We only train Harmonizers and Scores) [cite: 141]
        for p in self.raw_layers.parameters():
            p.requires_grad = False

    def get_probabilities(self):
        return torch.sigmoid(self.importance_scores)

    def sample_mask(self, group_size=4):
        """
        Implements the Grouping Sampling Strategy [cite: 151-153].
        Regroup layers into Ng=4. Keep layer if p_i > median(group).
        """
        probs = self.get_probabilities()
        mask = torch.zeros_like(probs, dtype=torch.bool)
        
        # Iterate in groups
        for i in range(0, self.num_layers, group_size):
            group_probs = probs[i : i + group_size]
            if len(group_probs) == 0: continue
            
            # Calculate median of this group [cite: 155]
            median_p = torch.median(group_probs)
            
            # Keep if p > median [cite: 153]
            # Note: This enforces roughly 50% replacement
            group_mask = group_probs >= median_p
            mask[i : i + group_size] = group_mask
            
        return mask

    def forward_with_mask(self, batch, mask):
        """
        Constructs the candidate network F dynamically[cite: 157, 158].
        """
        hidden_states = batch["input_ids"]
        # Basic embedding forward (simplified for brevity, assumes standard causal LM)
        # Note: Ideally we grab embeddings from full_model, but for reproduction we assume
        # we are wrapping the layers.
        # FIX: We need the embeddings. Let's assume we pass full_model to this class
        # or we hack the forward pass.
        pass # See forward_pass_logic below

# ==========================================
# 3. Main Training Logic (RL + DL)
# ==========================================
# 将此函数完全覆盖到 offsite_tuning/scaleot_exact.py 中

def train_scaleot_exact(args, full_model, tokenizer, train_dl, val_dl, accelerator):
    logger.info("🚀 Starting Exact ScaleOT Training (RL + DL)")
    
    full_model.to(accelerator.device)
    
    # Initialize Engine
    engine = DynamicLayerReplaceEngine(full_model, harmonizer_rank=64).to(accelerator.device)
    
    # Optimizer
    dl_optimizer = torch.optim.AdamW(engine.harmonizers.parameters(), lr=1e-4)
    
    Nc = 3 
    full_model.eval() 
    
    # [FIXED] Helper to run forward pass
    def run_forward(batch, mask):
        # 1. Embeddings
        x = full_model.model.embed_tokens(batch["input_ids"])

        # 2. Prepare RoPE (Position Embeddings)
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # 获取 RoPE (cos, sin)
        # 原始形状通常是 (Batch, Seq_Len, Head_Dim)
        rotary_emb = full_model.model.rotary_emb(x, position_ids)
        cos, sin = rotary_emb
        
        # [CRITICAL FIX] 调整维度以支持广播
        # 从 (Batch, Seq_Len, Dim) -> (Batch, 1, Seq_Len, Dim)
        # 这样才能与 (Batch, Num_Heads, Seq_Len, Dim) 的 Query/Key 匹配
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        position_embeddings = (cos, sin)

        # 3. Iterate Layers
        for idx, (raw_layer, harmonizer) in enumerate(zip(engine.raw_layers, engine.harmonizers)):
            if mask[idx]: # Keep Original
                layer_outputs = raw_layer(
                    x, 
                    attention_mask=None, 
                    position_ids=position_ids,
                    position_embeddings=position_embeddings 
                )
                x = layer_outputs[0]
            else: # Use Harmonizer
                x = harmonizer(x)
        
        # 4. Norm & Head
        x = full_model.model.norm(x)
        logits = full_model.lm_head(x)
        return logits

    # Dataset Iterators
    train_iter = iter(train_dl)
    val_iter = iter(val_dl)
    
    for step in range(args.steps):
        # === A. DL Step: Train Harmonizers ===
        try: batch_t = next(train_iter)
        except: train_iter = iter(train_dl); batch_t = next(train_iter)
        batch_t = {k: v.to(accelerator.device) for k, v in batch_t.items() if k in ['input_ids']}
        
        mask_dl = engine.sample_mask(group_size=4)
        logits_dl = run_forward(batch_t, mask_dl)
        
        labels = batch_t["input_ids"].clone()
        shift_logits = logits_dl[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss_dl = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        dl_optimizer.zero_grad()
        loss_dl.backward()
        dl_optimizer.step()
        
        # === B. RL Step: Update Importance Scores ===
        try: batch_v = next(val_iter)
        except: val_iter = iter(val_dl); batch_v = next(val_iter)
        batch_v = {k: v.to(accelerator.device) for k, v in batch_v.items() if k in ['input_ids']}
        
        losses_v = []
        masks_stored = []
        
        with torch.no_grad():
            for _ in range(Nc):
                m = engine.sample_mask(group_size=4)
                l_out = run_forward(batch_v, m)
                
                labels_v = batch_v["input_ids"].clone()
                s_log = l_out[..., :-1, :].contiguous()
                s_lab = labels_v[..., 1:].contiguous()
                loss_val = loss_fct(s_log.view(-1, s_log.size(-1)), s_lab.view(-1))
                
                losses_v.append(loss_val.item())
                masks_stored.append(m)
        
        exp_losses = [math.exp(-l) for l in losses_v]
        baseline = sum(exp_losses) / len(exp_losses) if len(exp_losses) > 0 else 0
        
        current_probs = engine.get_probabilities().detach()
        score_grad = torch.zeros_like(engine.importance_scores)
        
        for j in range(Nc):
            r_j = exp_losses[j] - baseline
            mask_j = masks_stored[j]
            sigmoid_derivative = current_probs * (1 - current_probs)
            update_step = r_j * sigmoid_derivative
            score_grad += (update_step * mask_j.float())
        
        with torch.no_grad():
            engine.importance_scores += (score_grad * 0.1)

        if step % 10 == 0:
            logger.info(f"Step {step} | DL Loss: {loss_dl.item():.4f} | RL Baseline Reward: {baseline:.4f}")

    return engine

# ==========================================
# 4. Emulator Generation [cite: 84]
# ==========================================
def generate_emulator(full_model, engine, alpha=0.5):
    """
    Selects layers based on learned scores and builds the final emulator.
    alpha: Proportion of layers replaced [cite: 258]
    """
    scores = engine.importance_scores.detach().cpu().numpy()
    num_layers = len(scores)
    num_keep = int(num_layers * (1 - alpha))
    
    # Sort indices by score descending
    sorted_indices = np.argsort(scores)[::-1]
    keep_indices = sorted(sorted_indices[:num_keep])
    keep_set = set(keep_indices)
    
    logger.info(f"🏆 Final Selected Layers (Top {len(keep_indices)}): {keep_indices}")
    
    emulator = copy.deepcopy(full_model)
    emu_layers = emulator.model.layers
    new_layers_list = []
    
    layer_map = {} # Maps emulator layer idx -> original layer idx
    
    # We reconstruct the list.
    # If index in keep_set -> Use Original Layer
    # If not -> Use Harmonizer (trained)
    
    # However, ScaleOT Emulator structure in Fig 2(c) shows:
    # "Adapter Layer | Frozen Layer | Harmonizer"
    # Usually, we physically replace the module in the list.
    
    for i in range(num_layers):
        if i in keep_set:
            new_layers_list.append(emu_layers[i])
            layer_map[len(new_layers_list)-1] = i
        else:
            # Insert Harmonizer
            # Paper Fig 1(d) says: combine original layers and harmonizers.
            # But usually for "Emulator", we want to reduce size.
            # If we keep Harmonizer, size is small.
            harm = copy.deepcopy(engine.harmonizers[i])
            # Hack: Wrap Harmonizer to look like a decoder layer for compatibility?
            # Or just append it.
            # For strict reproduction, we append the lightweight module.
            new_layers_list.append(harm)
            
    # Replace layers in emulator
    emulator.model.layers = nn.ModuleList(new_layers_list)
    emulator.config.num_hidden_layers = len(new_layers_list)
    emulator.config.layer_map = layer_map
    
    return emulator

# ==========================================
# 5. Execution Entry Point
# ==========================================
def main():
    # 1. Setup
    accelerator = Accelerator()
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    
    # 2. Data (Use Local PIQA logic from before)
    # [Placeholder for Data Loading Code - assume train_dl/val_dl exists]
    # For reproduction script, create dummy if not exists
    class DummyDL:
        def __iter__(self): 
            while True: 
                yield {"input_ids": torch.randint(0, 1000, (4, 128))}
    train_dl = DummyDL()
    val_dl = DummyDL()
    
    args = type('Args', (), {'steps': 200})() # 200 steps for demo
    
    # 3. Train Dynamic LayerReplace (RL Phase)
    engine = train_scaleot_exact(args, model, tokenizer, train_dl, val_dl, accelerator)
    
    # 4. Generate Emulator
    # alpha=0.5 means keep 50% layers
    emulator = generate_emulator(model, engine, alpha=0.25)
    
    # 5. Apply Selective Rank Compression (SRC)
    # beta=0.8 [cite: 275]
    apply_src_to_emulator(emulator, beta=0.8)
    
    logger.info("✅ Exact ScaleOT Emulator Created.")

if __name__ == "__main__":
    main()