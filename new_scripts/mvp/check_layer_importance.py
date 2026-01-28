import os
import sys

# ==========================================
# [配置区]
# ==========================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
CACHE_DIR = "/data/xiaowen"
os.environ["HF_TOKEN"] = "MY_TOKEN"  # 替换为你的 Hugging Face 访问令牌

if not os.path.exists(CACHE_DIR):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
    except:
        pass

# ==========================================
# 导入依赖
# ==========================================
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_model_layers(model):
    if hasattr(model, "model"):
        if hasattr(model.model, "layers"): return model.model.layers 
        elif hasattr(model.model, "decoder"): return model.model.decoder.layers
    if hasattr(model, "decoder"): return model.decoder.layers
    raise ValueError(f"Unknown architecture: {type(model)}")

def plot_vertical_heatmap(results, output_filename="layer_sensitivity_14b.png"):
    print(f"\nGenerating heatmap to {output_filename}...")
    tasks = list(results.keys())
    layers = sorted(list(results[tasks[0]].keys()))
    
    # 14B 有 48 层，跳过前 2 层 Embedding 噪音
    start_layer = 2
    display_layers = layers[start_layer:]
    
    matrix = []
    for i in display_layers:
        row = []
        for task in tasks:
            row.append(results[task][i])
        matrix.append(row)
    matrix = np.array(matrix)
    
    # 归一化
    col_sums = matrix.sum(axis=0)
    norm_matrix = matrix / col_sums
    
    # 计算差异 (Yelp - GSM8K)
    diff_col = (norm_matrix[:, 0] - norm_matrix[:, 1]).reshape(-1, 1)
    
    plt.figure(figsize=(10, 14)) # 14B 层数适中，画布不用太长
    sns.set_theme(style="white")
    gs = plt.GridSpec(1, 2, width_ratios=[2, 1]) 
    
    ax1 = plt.subplot(gs[0])
    sns.heatmap(
        norm_matrix,
        annot=True, fmt=".3f", cmap="Blues",
        xticklabels=tasks, yticklabels=[f"L{i}" for i in display_layers],
        ax=ax1, cbar=False, annot_kws={"size": 9}
    )
    ax1.set_title("Normalized Sensitivity", fontsize=12)
    
    ax2 = plt.subplot(gs[1])
    sns.heatmap(
        diff_col,
        annot=True, fmt=".3f", cmap="vlag", center=0,
        xticklabels=["Diff"], yticklabels=[],
        ax=ax2, cbar=True, annot_kws={"size": 9}
    )
    ax2.set_title("Divergence (>0 Yelp, <0 GSM)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Saved to {os.path.abspath(output_filename)}")

def get_layer_sensitivity(model, dataloader, device):
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
    
    # 14B 单卡能跑，无需 device_map="auto" 的复杂处理，直接 to(device)
    model.to(device)
    
    layers = get_model_layers(model)
    num_layers = len(layers)
    layer_grads = {i: 0.0 for i in range(num_layers)}
    
    criterion = torch.nn.CrossEntropyLoss()
    MAX_STEPS = 20
    
    print(f"Profiling {num_layers} layers...")
    
    for step, batch in enumerate(tqdm(dataloader, desc="Computing")):
        if step >= MAX_STEPS: break
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = criterion(outputs.logits[..., :-1, :].contiguous().view(-1, outputs.logits.size(-1)), 
                         batch["input_ids"][..., 1:].contiguous().view(-1))
        loss.backward()
        
        for i, layer in enumerate(layers):
            grad_sum = 0.0
            for param in layer.parameters():
                if param.grad is not None:
                    grad_sum += param.grad.detach().norm(2).item()
            layer_grads[i] += grad_sum
        model.zero_grad()
        
    total = sum(layer_grads.values())
    if total == 0: total = 1e-9
    return {k: v / total for k, v in layer_grads.items()}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # === [核心修改] 换用 Qwen2.5-14B ===
    # 14B 占用约 28GB 显存 (BF16)。
    # 这种情况下，自动启用 device_map="auto" 会把溢出的几层放到内存，速度依然很快。
    # model_name = "Qwen/Qwen2.5-14B" 
    # 如果 14B 还是跑不动，请解除下面这行的注释用 7B (绝对稳)：
    model_name = "Qwen/Qwen2.5-7B"
    
    print(f"Loading Model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=CACHE_DIR, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=CACHE_DIR, 
            trust_remote_code=True,
            device_map="auto", # 显存不够自动借用内存
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    tasks = [
        {"name": "Yelp", "id": "yelp_review_full", "col": "text", "config": None},
        {"name": "GSM8K", "id": "gsm8k", "col": "question", "config": "main"}
    ]
    
    results = {}
    for task in tasks:
        print(f"\nProcessing {task['name']}...")
        try:
            ds = load_dataset(task["id"], task["config"], split="train[:200]", cache_dir=CACHE_DIR)
            def tok(x): return tokenizer(x[task["col"]], padding="max_length", truncation=True, max_length=128)
            dl = DataLoader(ds.map(tok, batched=True, remove_columns=ds.column_names), batch_size=2, collate_fn=default_data_collator)
            # 传入 model.device 兼容 device_map
            results[task["name"]] = get_layer_sensitivity(model, dl, model.device)
        except Exception as e:
            print(f"Skipping {task['name']}: {e}")

    if results:
        plot_vertical_heatmap(results, output_filename="layer_sensitivity_14b.png")

if __name__ == "__main__":
    main()