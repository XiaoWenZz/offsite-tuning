import argparse
import logging
import sys
import copy
import random
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

logger = get_logger(__name__)

# ==========================================
# 工具函数
# ==========================================
def create_emulator(full_model, keep_k=2):
    emulator = copy.deepcopy(full_model)
    if hasattr(emulator, "model"): layers = emulator.model.decoder.layers
    else: layers = emulator.decoder.layers
    total_layers = len(layers)
    indices_to_keep = list(range(keep_k)) + list(range(total_layers - keep_k, total_layers))
    new_layers = nn.ModuleList([layers[i] for i in indices_to_keep])
    if hasattr(emulator, "model"):
        emulator.model.decoder.layers = new_layers
        emulator.config.num_hidden_layers = len(new_layers)
    else:
        emulator.decoder.layers = new_layers
        emulator.config.num_hidden_layers = len(new_layers)
    # 必须禁用 use_cache，否则被剪枝的模型在前向传播时会因为缓存索引不匹配报错
    emulator.config.use_cache = False
    return emulator

def flatten_gradients(model):
    grads = []
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            grads.append(param.grad.view(-1).detach().cpu())
    return torch.cat(grads) if grads else None

def partition_data_dirichlet(dataset, num_clients, alpha=0.1, seed=42):
    np.random.seed(seed)
    labels = np.array(dataset['label'])
    num_classes = len(np.unique(labels))
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    class_indices = [np.argwhere(labels == y).flatten() for y in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    for c_idx, (c_data_indices, fracs) in enumerate(zip(class_indices, label_distribution)):
        fracs = fracs / fracs.sum()
        split_points = (np.cumsum(fracs) * len(c_data_indices)).astype(int)[:-1]
        client_data_split = np.split(c_data_indices, split_points)
        for i, split in enumerate(client_data_split):
            client_indices[i].append(split)
    return [np.concatenate(idcs) for idcs in client_indices]

# === [新增] 评估函数 ===
def evaluate_model(model, dataloader, accelerator):
    model.eval()
    total_loss = 0
    steps = 0
    # 为了速度，只评估少量 batch
    MAX_EVAL_STEPS = 10 
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= MAX_EVAL_STEPS: break
            
            # 显式传入 labels 以计算 loss
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            total_loss += loss.item()
            steps += 1
            
    return total_loss / steps if steps > 0 else 0.0

class VirtualClient:
    def __init__(self, client_id, train_loader, test_loader, label_dist_str):
        self.id = client_id
        self.train_dataloader = train_loader
        self.test_dataloader = test_loader # 新增测试集
        self.label_info = label_dist_str

# ==========================================
# 参数解析
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run Clustered Federated Offsite-Tuning")
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
    parser.add_argument("--dataset_name", type=str, default="ag_news")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--num_clusters", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--keep_layers", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="fed_offsite_tuning")
    parser.add_argument("--wandb_run_name", type=str, default="cluster_fedot")

    parser.add_argument("--cache-dir", type=str, default=None, help="Directory to cache the dataset.")
    return parser.parse_args()

# ==========================================
# 主程序
# ==========================================
def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator()
    
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    logger.info("=== [Step 1] Initializing Model & Data ===")
    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    full_model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)
    
    base_emulator = create_emulator(full_model, keep_k=args.keep_layers)
    model = accelerator.prepare(base_emulator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = accelerator.prepare(optimizer)

    # Prepare Data
    logger.info(f"Loading dataset {args.dataset_name} with cache_dir={args.cache_dir}...")
    
    # 通用加载逻辑 (结合之前的修改)
    # ... (args = parse_args() 等代码) ...

    # Prepare Data
    # 这一步如果不加 split 参数，对于 Yelp 会返回 {'train': ..., 'test': ...}
    dataset = datasets.load_dataset(args.dataset_name, cache_dir=args.cache_dir)

    # === [关键修复] 处理 DatasetDict ===
    # 如果加载出来的是个字典（包含 train/test），我们只取 train 部分
    if isinstance(dataset, datasets.DatasetDict):
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            # 如果没有 train，就取第一个 split
            dataset = dataset[list(dataset.keys())[0]]
            
    # === [优化] 大数据集切片 ===
    # Yelp 有 65万条，全量跑太慢，这里保留之前的逻辑，只取前 2万条
    if len(dataset) > 20000:
        logger.info(f"Dataset is too large ({len(dataset)}), slicing first 20,000 examples.")
        dataset = dataset.select(range(20000))

    # === [Modification] 基于参数的列名标准化 (No Hard-coding) ===
    logger.info(f"Target columns from args: Text='{args.text_column}', Label='{args.label_column}'")
    
    # === [Modification] 基于参数的列名标准化 (No Hard-coding) ===
    
    logger.info(f"Target columns from args: Text='{args.text_column}', Label='{args.label_column}'")
    
    # 1. 检查列名是否存在
    if args.text_column not in dataset.column_names:
        raise ValueError(f"Column '{args.text_column}' not found in dataset. Available columns: {dataset.column_names}")
    
    # 部分数据集可能没有 label 列（如果是纯无监督训练），这里做一个容错，或者强制要求
    if args.label_column not in dataset.column_names:
         raise ValueError(f"Column '{args.label_column}' not found in dataset. Available columns: {dataset.column_names}")

    # 2. 统一重命名为内部通用的 'text' 和 'label'
    # 这样后续的 tokenizer 和 partition 逻辑都不用改
    if args.text_column != "text":
        dataset = dataset.rename_column(args.text_column, "text")
        logger.info(f"Renamed column '{args.text_column}' -> 'text'")
    
    if args.label_column != "label":
        dataset = dataset.rename_column(args.label_column, "label")
        logger.info(f"Renamed column '{args.label_column}' -> 'label'")

    client_indices = partition_data_dirichlet(dataset, args.num_clients, alpha=args.alpha)
    
    clients = []
    for cid, indices in enumerate(client_indices):
        sub_ds = dataset.select(indices)
        
        # === [修改] 划分 Train/Test ===
        # 每个 Client 切分 10% 做测试
        split_ds = sub_ds.train_test_split(test_size=0.1)
        
        u, c = np.unique(sub_ds['label'], return_counts=True)
        dist_info = dict(zip(u, c))
        
        def tokenize_fn(examples):
            return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)
        
        tokenized_train = split_ds['train'].map(tokenize_fn, batched=True, remove_columns=['text', 'label'])
        tokenized_test = split_ds['test'].map(tokenize_fn, batched=True, remove_columns=['text', 'label'])
        
        train_loader = DataLoader(tokenized_train, batch_size=args.batch_size, collate_fn=default_data_collator, shuffle=True)
        test_loader = DataLoader(tokenized_test, batch_size=args.batch_size, collate_fn=default_data_collator)
        
        train_loader = accelerator.prepare(train_loader)
        test_loader = accelerator.prepare(test_loader)
        
        clients.append(VirtualClient(cid, train_loader, test_loader, str(dist_info)))

    # === Phase 1: Pilot Round ===
    logger.info("=== [Step 2] Pilot Round ===")
    gradient_sketches = []

    # === [修复] 强制将初始状态移到 CPU，避免显存溢出和设备冲突 ===
    # 这样后续所有的 cluster_global_states 都会在 CPU 上维护
    initial_state_dict = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

    for client in clients:
        model.load_state_dict(initial_state_dict)
        model.train()
        model.zero_grad()
        try:
            batch = next(iter(client.train_dataloader))
        except StopIteration:
            continue
        outputs = model(**batch, labels=batch["input_ids"]) # 修复 labels
        accelerator.backward(outputs.loss)
        
        sketch = flatten_gradients(model)
        if sketch is not None:
            gradient_sketches.append(sketch.numpy())
        else:
            gradient_sketches.append(np.zeros(1))

    # Clustering
    logger.info("Running K-Means...")
    grad_matrix = np.stack(gradient_sketches)
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(grad_matrix)
    
    clusters = {i: [] for i in range(args.num_clusters)}
    for cid, label in enumerate(labels):
        clusters[label].append(clients[cid])
        logger.info(f"Client {cid} -> Cluster {label}")
        
    if accelerator.is_main_process:
        wandb.log({f"cluster_{k}_size": len(v) for k, v in clusters.items()})

    # === Phase 2: Federated Training ===
    logger.info("=== [Step 3] Clustered Federated Training ===")
    cluster_global_states = {k: copy.deepcopy(initial_state_dict) for k in range(args.num_clusters)}
    
    for round_idx in range(args.rounds):
        logger.info(f"--- Round {round_idx + 1} ---")
        round_metrics = {}
        
        for cluster_id, cluster_clients in clusters.items():
            if not cluster_clients: continue
            
            global_update_accumulator = {}
            current_cluster_state = cluster_global_states[cluster_id]

            # 1. 训练阶段
            for client in cluster_clients:
                model.load_state_dict(current_cluster_state)
                model.train()

                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
                optimizer.zero_grad()
                
                for step, batch in enumerate(client.train_dataloader):
                    if step >= args.local_steps: break
                    outputs = model(**batch, labels=batch["input_ids"]) # 修复 labels
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Compute Delta
                client_final_state = model.state_dict()
                for key, value in client_final_state.items():
                    delta = value.detach().cpu() - current_cluster_state[key].cpu()
                    if key not in global_update_accumulator:
                        global_update_accumulator[key] = delta
                    else:
                        global_update_accumulator[key] += delta

            # 2. 聚合阶段
            if global_update_accumulator:
                for key in cluster_global_states[cluster_id]:
                    if key in global_update_accumulator:
                        cluster_global_states[cluster_id][key] += global_update_accumulator[key] / len(cluster_clients)
            
            # 3. === [新增] 评估阶段 ===
            # 将更新后的簇模型，加载到 GPU
            model.load_state_dict(cluster_global_states[cluster_id])
            
            # 评估该簇在“簇内 Client 测试集”上的平均表现
            test_losses = []
            for client in cluster_clients:
                # 每个 Client 在自己的 Test Set 上跑 eval
                l = evaluate_model(model, client.test_dataloader, accelerator)
                test_losses.append(l)
            
            avg_cluster_test_loss = np.mean(test_losses)
            logger.info(f"Cluster {cluster_id} Test Loss: {avg_cluster_test_loss:.4f}")
            round_metrics[f"cluster_{cluster_id}_test_loss"] = avg_cluster_test_loss

        if accelerator.is_main_process:
            round_metrics["round"] = round_idx + 1
            wandb.log(round_metrics)

    # === Verify Divergence ===
    logger.info("=== Verification ===")
    diff_sum = 0
    if 0 in cluster_global_states and 1 in cluster_global_states:
        for key in cluster_global_states[0]:
            diff = (cluster_global_states[0][key] - cluster_global_states[1][key]).float().norm().item()
            diff_sum += diff
    
    logger.info(f"Model Divergence: {diff_sum:.4f}")
    if accelerator.is_main_process:
        wandb.log({"final_divergence": diff_sum})
        wandb.finish()

if __name__ == "__main__":
    main()