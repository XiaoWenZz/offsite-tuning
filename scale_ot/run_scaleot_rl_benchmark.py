import argparse
import logging
import sys
import copy
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, default_data_collator, set_seed
from accelerate import Accelerator
from accelerate.logging import get_logger
import wandb

# 导入核心库
from scaleot_core import (
    build_emulator, 
    server_side_alignment, 
    evaluate_plugback, 
    load_local_piqa,
    get_module_layers,
    get_trainable_keys
)

logger = get_logger(__name__)

# ==========================================
# RL Component
# ==========================================
def compute_sensitivity_profile(model, dataloader, accelerator):
    model.eval()
    layers = get_module_layers(model)
    sens_vec = np.zeros(len(layers))
    
    for i, batch in enumerate(dataloader):
        if i >= 5: break
        batch = {k: v.to(accelerator.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        outputs = model(**batch, labels=batch['input_ids'])
        loss = outputs.loss
        loss.backward()
        
        for idx, layer in enumerate(layers):
            grad_norm = 0.0
            for p in layer.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.detach().float().norm(2).item()
            sens_vec[idx] += grad_norm
        model.zero_grad()
        
    if np.sum(sens_vec) > 0: sens_vec /= np.sum(sens_vec)
    return sens_vec

def rl_layer_selection(sensitivity_vector, budget, total_layers):
    temperature = 2.0 
    logits = np.exp(sensitivity_vector * temperature)
    probs = logits / np.sum(logits)
    
    required = {0, total_layers - 1}
    remaining_budget = budget - len(required)
    
    if remaining_budget <= 0: return sorted(list(required))
    
    probs[0] = 0; probs[-1] = 0
    probs /= np.sum(probs) 
    
    chosen = np.random.choice(range(total_layers), size=remaining_budget, replace=False, p=probs)
    final_indices = sorted(list(required.union(set(chosen))))
    
    # [FIX] Clean numpy types
    return [int(x) for x in final_indices]

# ==========================================
# Experiment Loop
# ==========================================
def run_experiment(strategy_name, full_model, tokenizer, train_dl, test_dl, accelerator, args):
    logger.info(f"\n{'='*50}")
    logger.info(f"🚀 Starting Experiment: {strategy_name}")
    logger.info(f"{'='*50}")
    
    total_layers = len(get_module_layers(full_model))
    
    # 1. Policy
    if strategy_name == "Uniform_Baseline":
        indices = np.linspace(0, total_layers-1, args.layer_budget, dtype=int).tolist()
        indices = [int(x) for x in indices]
        logger.info(f"   📏 Strategy: Uniform Stride")
        logger.info(f"   📋 Selected Layers: {indices}")
    else:
        logger.info("   🤖 RL Agent: Probing environment (sensitivity)...")
        sens_vec = compute_sensitivity_profile(full_model, train_dl, accelerator)
        
        ranked_indices = np.argsort(sens_vec)[::-1] 
        logger.info("\n   📊 RL Importance Ranking (Top 10):")
        logger.info("   ------------------------------------------------")
        for rank, idx in enumerate(ranked_indices[:10]):
            logger.info(f"   Rank {rank+1:2d}: Layer {idx:2d} | Score: {sens_vec[idx]:.4f}")
        logger.info("   ------------------------------------------------")
        
        indices = rl_layer_selection(sens_vec, args.layer_budget, total_layers)
        logger.info(f"   🎲 RL Policy Action: {indices}\n")
        
    # 2. Build & Align
    emulator = build_emulator(full_model, indices, use_src=True)
    server_side_alignment(full_model, emulator, tokenizer, accelerator, num_steps=100)
    
    # 3. Train
    trainable_keys = get_trainable_keys(emulator)
    emulator.to(accelerator.device); emulator.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, emulator.parameters()), lr=args.lr)
    
    for round_idx in range(args.rounds):
        loss_accum = 0; steps = 0
        for i, batch in enumerate(train_dl):
            if i >= args.local_steps: break
            batch = {k: v.to(accelerator.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            labels = batch['input_ids'].clone(); labels[labels==tokenizer.pad_token_id] = -100
            
            outputs = emulator(**batch, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step(); optimizer.zero_grad()
            loss_accum += loss.item(); steps += 1
            
        avg_emu_loss = loss_accum / steps if steps > 0 else 0
        plug_loss = evaluate_plugback(full_model, emulator, test_dl, accelerator, tokenizer)
        
        try: emu_ppl = math.exp(avg_emu_loss)
        except: emu_ppl = float('inf')
        try: plug_ppl = math.exp(plug_loss)
        except: plug_ppl = float('inf')
        
        logger.info(f"   Round {round_idx+1:2d} | Emu PPL: {emu_ppl:6.2f} | Plug PPL: {plug_ppl:6.2f}")
        
        # [FIX] Removed 'step' argument to allow auto-increment and avoid conflict
        wandb.log({
            f"{strategy_name}/emu_loss": avg_emu_loss,
            f"{strategy_name}/plug_loss": plug_loss,
            f"{strategy_name}/plug_ppl": plug_ppl,
            "strategy_round": round_idx + 1 
        })
        
    del emulator; torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layer_budget", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--local_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    accelerator = Accelerator(log_with="wandb")
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    
    accelerator.init_trackers("scaleot_rl_benchmark", config=vars(args))
    
    logger.info("loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    full_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    
    ds = load_local_piqa(tokenizer, num_samples=1000)
    split = ds.train_test_split(test_size=0.1)
    train_dl = DataLoader(split['train'], batch_size=4, collate_fn=default_data_collator, shuffle=True)
    test_dl = DataLoader(split['test'], batch_size=4, collate_fn=default_data_collator)
    
    run_experiment("Uniform_Baseline", full_model, tokenizer, train_dl, test_dl, accelerator, args)
    run_experiment("RL_Adaptive", full_model, tokenizer, train_dl, test_dl, accelerator, args)
    
    accelerator.end_training()

if __name__ == "__main__": main()