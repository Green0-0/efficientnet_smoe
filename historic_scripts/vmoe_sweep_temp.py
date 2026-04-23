import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import wandb
import os
import time
import math

# ==========================================
# 1. ARCHITECTURE (V-MoE)
# ==========================================

class Mlp(nn.Module):
    """Standard dense MLP for ViT blocks."""
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SparseMoEMlp(nn.Module):
    """Mixture of Experts MLP replacing the dense MLP in selected blocks."""
    def __init__(self, in_features, hidden_features, num_experts, k, capacity_factor, drop=0.1, noise_mult=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.capacity_factor = capacity_factor
        
        # Router
        self.router = nn.Linear(in_features, num_experts, bias=False)
        self.noise_std = noise_mult / num_experts 
        
        # Experts
        self.experts = nn.ModuleList([
            Mlp(in_features, hidden_features, drop=drop) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        N = B * S
        
        # 1. Routing & Noise addition
        logits = self.router(x_flat).float()
        clean_probs = F.softmax(logits, dim=-1)
        
        if self.training:
            noise = torch.randn_like(logits) * self.noise_std
            noisy_logits = logits + noise
        else:
            noisy_logits = logits
            
        noisy_probs = F.softmax(noisy_logits, dim=-1)
        
        # 2. Top-K Selection
        topk_probs, topk_indices = torch.topk(noisy_probs, self.k, dim=-1)
        topk_probs = topk_probs.to(x.dtype)
        
        # Track Expert Utilization
        expert_counts = torch.bincount(topk_indices.flatten(), minlength=self.num_experts).float()
        
        # 3. Auxiliary Loss
        importance = clean_probs.sum(dim=0)
        mean_importance = importance.mean()
        loss_importance = importance.var(unbiased=False) / (mean_importance ** 2 + 1e-6)
        
        # FIX: Differentiable Load Proxy Loss using Noisy Logits threshold
        threshold_idx = self.num_experts - self.k + 1
        threshold = torch.kthvalue(noisy_logits, threshold_idx, dim=-1).values.unsqueeze(-1)
        
        dist = torch.distributions.Normal(
            torch.tensor(0.0, device=logits.device, dtype=logits.dtype),
            torch.tensor(self.noise_std, device=logits.device, dtype=logits.dtype)
        )
        prob_selected = 1.0 - dist.cdf(threshold - logits)
        load = prob_selected.sum(dim=0)
        mean_load = load.mean()
        loss_load = load.var(unbiased=False) / (mean_load ** 2 + 1e-6)
        
        aux_loss = 0.5 * (loss_importance + loss_load)
        
        # 4. Dispatch with Capacity Limit & Batch Prioritized Routing (BPR)
        capacity = int(round((self.k * N * self.capacity_factor) / self.num_experts))
        
        out_flat = torch.zeros_like(x_flat)
        dropped_assignments = 0
        
        priority_scores = clean_probs.max(dim=-1)[0] 
        
        for i in range(self.num_experts):
            expert_mask = (topk_indices == i)
            token_indices, k_idx = torch.where(expert_mask)
            
            if len(token_indices) == 0:
                continue
                
            tok_priorities = priority_scores[token_indices]
            combined_scores = -k_idx.float() + tok_priorities
            _, sorted_idx = torch.sort(combined_scores, descending=True)
            
            token_indices = token_indices[sorted_idx]
            k_idx = k_idx[sorted_idx]
            
            if len(token_indices) > capacity:
                dropped_assignments += (len(token_indices) - capacity)
                token_indices = token_indices[:capacity]
                k_idx = k_idx[:capacity]
                
            expert_inputs = x_flat[token_indices]
            expert_outputs = self.experts[i](expert_inputs)
            
            weights = topk_probs[token_indices, k_idx].unsqueeze(-1)
            weighted_outputs = expert_outputs * weights
            
            out_flat.index_add_(0, token_indices, weighted_outputs)
            
        out = out_flat.view(B, S, D)
        return out, aux_loss, dropped_assignments, expert_counts


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., is_moe=False, num_experts=8, 
                 k=2, capacity_factor=1.05, drop=0., noise_mult=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        
        self.is_moe = is_moe
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        if is_moe:
            self.mlp = SparseMoEMlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                                    num_experts=num_experts, k=k, 
                                    capacity_factor=capacity_factor, drop=drop, noise_mult=noise_mult)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.is_moe:
            mlp_out, aux_loss, dropped, expert_counts = self.mlp(self.norm2(x))
            x = x + mlp_out
            return x, aux_loss, dropped, expert_counts
        else:
            x = x + self.mlp(self.norm2(x))
            return x, 0.0, 0, None


class VMoE(nn.Module):
    """Vision Mixture of Experts architecture (CLS Token Based)."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=10000, 
                 embed_dim=256, depth=8, num_heads=8, mlp_ratio=4., 
                 num_experts=8, k=2, capacity_factor=1.05, placement='every_1', drop=0.1, noise_mult=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # FIX: Restored CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            # Enforced every_1 per user prompt
            is_moe = True if placement == 'every_1' else (i % 2 != 0)
                
            self.blocks.append(
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      is_moe=is_moe, num_experts=num_experts, k=k, 
                      capacity_factor=capacity_factor, drop=drop, noise_mult=noise_mult)
            )
            
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # FIX: Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        total_aux_loss = 0.0
        total_dropped_assignments = 0
        total_moe_tokens = 0
        all_expert_counts = None
        
        for block in self.blocks:
            if block.is_moe:
                x, aux, dropped, e_counts = block(x)
                total_aux_loss += aux
                total_dropped_assignments += dropped
                total_moe_tokens += (x.shape[0] * x.shape[1] * block.mlp.k)
                
                if all_expert_counts is None:
                    all_expert_counts = e_counts
                else:
                    all_expert_counts += e_counts
            else:
                x, _, _, _ = block(x)

        x = self.norm(x)
        # FIX: Readout from CLS token instead of GAP
        x = x[:, 0] 
        out = self.head(x)
        
        return out, total_aux_loss, total_dropped_assignments, total_moe_tokens, all_expert_counts


def estimate_vmoe_params(num_classes, embed_dim, depth, num_heads, num_experts, k, placement):
    """Analytically estimates parameters without building nn.Modules."""
    mlp_ratio = 4.0
    mlp_hidden = int(embed_dim * mlp_ratio)

    patch_embed = 3 * 16 * 16 * embed_dim + embed_dim
    cls_embed = embed_dim
    pos_embed = 197 * embed_dim # 196 patches + 1 cls
    head = embed_dim * num_classes + num_classes
    final_norm = 2 * embed_dim # FIX: Final LayerNorm included

    attn_params = (embed_dim * 3 * embed_dim + 3 * embed_dim) + (embed_dim * embed_dim + embed_dim)
    norm_params = 2 * embed_dim * 2

    dense_mlp = (embed_dim * mlp_hidden + mlp_hidden) + (mlp_hidden * embed_dim + embed_dim)
    router = embed_dim * num_experts 
    expert_mlp = dense_mlp

    total_params = patch_embed + cls_embed + pos_embed + head + final_norm
    active_params = patch_embed + cls_embed + pos_embed + head + final_norm

    for i in range(depth):
        is_moe = (placement == 'every_1') or (placement == 'every_2' and i % 2 != 0)
        total_params += attn_params + norm_params
        active_params += attn_params + norm_params

        if is_moe:
            total_params += router + num_experts * expert_mlp
            active_params += router + k * expert_mlp
        else:
            total_params += dense_mlp
            active_params += dense_mlp

    return total_params, active_params


def build_10m_vmoe(num_classes, num_experts, k, capacity_factor, placement, drop, noise_mult):
    best_diff = float('inf')
    best_dim = 128
    
    depth = 8
    num_heads = 8
    target_total_params = 10_000_000
    
    for dim in range(64, 512, 8):
        if dim % num_heads != 0: continue
        total_params, active_params = estimate_vmoe_params(num_classes, dim, depth, num_heads, num_experts, k, placement)
        diff = abs(total_params - target_total_params)
        
        if diff < best_diff:
            best_diff = diff
            best_dim = dim

    final_model = VMoE(num_classes=num_classes, embed_dim=best_dim, depth=depth, 
                       num_heads=num_heads, num_experts=num_experts, k=k, 
                       capacity_factor=capacity_factor, placement=placement, drop=drop, noise_mult=noise_mult)
                  
    final_total, final_active = estimate_vmoe_params(num_classes, best_dim, depth, num_heads, num_experts, k, placement)
    actual_sparsity = final_active / final_total
    
    return final_model, best_dim, actual_sparsity, final_total

# ==========================================
# 2. DATA LOADING
# ==========================================
def get_dataloaders(data_dir, batch_size, num_workers=8):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train_dataset = datasets.INaturalist(root=data_dir, version='2021_train_mini', 
                                              transform=train_transforms, download=True)
    full_val_dataset = datasets.INaturalist(root=data_dir, version='2021_train_mini', 
                                            transform=val_transforms, download=True)

    num_classes = len(full_train_dataset.all_categories)
    class_indices = {}
    for i, (cat_id, _) in enumerate(full_train_dataset.index):
        if cat_id not in class_indices:
            class_indices[cat_id] = []
        class_indices[cat_id].append(i)

    min_points = min(len(idxs) for idxs in class_indices.values())
    val_pts_per_class = max(1, int(0.1 * min_points))
    train_pts_per_class = min_points - val_pts_per_class

    train_indices, val_indices = [], []
    g = torch.Generator().manual_seed(42)
    for cat_id, idxs in class_indices.items():
        shuffled = torch.tensor(idxs)[torch.randperm(len(idxs), generator=g)].tolist()
        val_indices.extend(shuffled[:val_pts_per_class])
        train_indices.extend(shuffled[val_pts_per_class:val_pts_per_class+train_pts_per_class])

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=min(2, num_workers), pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, num_classes

# ==========================================
# 3. OPTUNA + WANDB OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    trial_start_time = time.time()
    
    base_lr_mult = trial.suggest_float("base_lr_mult", 0.0001, 0.01)
    batch_size = 256
    grad_accum_steps = trial.suggest_int("grad_accum_steps", 4, 16, 2)
    global_batch_size = batch_size * grad_accum_steps
    lr = base_lr_mult * (global_batch_size / 256.0)
    
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.2)
    # FIX: Excluded 32 to ensure viable experts, 4 added for viable sparsity math
    num_experts = trial.suggest_categorical("num_experts", [4, 8, 16]) 
    k = trial.suggest_int("k", 1, 2)
    
    # FIX: Widened capacity bounds to test extreme token dropping
    capacity_factor = trial.suggest_categorical("capacity_factor", [0.65, 0.8, 1.0, 1.25])
    placement = "every_1" # Hardcoded per user constraint
    
    noise_mult = trial.suggest_categorical("noise_mult", [0.5, 1.0, 2.0])
    
    # FIX: Crucial swept hyperparameters
    moe_loss_weight = trial.suggest_categorical("moe_loss_weight", [0.001, 0.01, 0.05, 0.1])
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
    label_smooth = trial.suggest_float("label_smoothing", 0.0, 0.2)
    
    run = wandb.init(
        project="vmoe_tpe_sweep_final",
        group="slurm_sweep",
        name=f"trial_{trial.number}_E{num_experts}_K{k}_C{capacity_factor}",
        config={
            "learning_rate": lr,
            "base_lr_mult": base_lr_mult,
            "global_batch_size": global_batch_size,
            "num_experts": num_experts,
            "k_active_experts": k,
            "capacity_factor": capacity_factor,
            "placement": placement,
            "noise_multiplier": noise_mult,
            "moe_loss_weight": moe_loss_weight,
            "weight_decay": weight_decay,
            "label_smoothing": label_smooth,
            "dropout": dropout_rate,
            "optimizer": "AdamW", 
            "model": "V-MoE (~10M Params)",
        },
        reinit=True
    )
    
    data_dir = '/usr/project/xtmp/inaturalistdata_store' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    num_workers = max(1, slurm_cpus - 2)
    train_loader, val_loader, num_classes = get_dataloaders(data_dir, batch_size, num_workers=num_workers)
    
    model, best_dim, actual_sparsity, actual_params = build_10m_vmoe(
        num_classes, num_experts, k, capacity_factor, placement, dropout_rate, noise_mult
    )
    
    if not (9.5e6 <= actual_params <= 10.5e6):
        wandb.run.summary["state"] = "pruned (param limit)"
        raise optuna.exceptions.TrialPruned()
        
    if not (0.30 <= actual_sparsity <= 0.50):
        wandb.run.summary["state"] = f"pruned (sparsity {actual_sparsity:.2f})"
        raise optuna.exceptions.TrialPruned()
        
    model = model.to(device)
    # FIX: torch.compile completely removed to prevent dynamic flow graph breaks
    
    wandb.config.update({
        "embed_dim": best_dim,
        "actual_parameters": actual_params,
        "actual_sparsity_percentage": actual_sparsity * 100.0,
    })
    
    TARGET_EPOCHS = 20
    WARMUP_EPOCHS = 2
    
    effective_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    warmup_steps = int(WARMUP_EPOCHS * effective_steps_per_epoch)
    total_steps = int(TARGET_EPOCHS * effective_steps_per_epoch)
    cosine_steps = max(1, total_steps - warmup_steps)
    
    # FIX: Position/CLS tokens get a 10x smaller learning rate
    decay_params, no_decay_params, embed_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if "pos_embed" in name or "cls_token" in name:
            embed_params.append(param)
        elif len(param.shape) == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
        {'params': embed_params, 'weight_decay': 0.0, 'lr': lr * 0.1}
    ], lr=lr, betas=(0.9, 0.999), eps=1e-8)

    def lr_lambda(step):
        if step < warmup_steps: return float(step + 1) / float(max(1, warmup_steps))
        else: return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / cosine_steps))
        
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    
    print(f"\n--- Starting V-MoE Trial {trial.number} ---")
    print(f"Dim={best_dim}, E={num_experts}, k={k} ({actual_sparsity*100:.1f}% Active), C={capacity_factor}")
    
    global_step = 0
    final_val_acc = 0.0
    
    accum_dropped_assignments = 0
    accum_total_tokens = 0
    
    try:
        for epoch in range(TARGET_EPOCHS):
            model.train()
            epoch_loss = 0.0
            
            optimizer.zero_grad()
            for i, (images, labels) in enumerate(train_loader):

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs, aux_loss, dropped, batch_total_tokens, expert_counts = model(images)
                    main_loss = criterion(outputs, labels)
                    loss = (main_loss + moe_loss_weight * aux_loss) / grad_accum_steps
                
                loss.backward()
                epoch_loss += loss.item() * grad_accum_steps
                
                accum_dropped_assignments += dropped
                accum_total_tokens += batch_total_tokens
                
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 50 == 0:
                        dropped_ratio = (accum_dropped_assignments / max(1, accum_total_tokens)) * 100.0
                        
                        # FIX: Log expert utilization statistics to diagnose collapse
                        utilization_mean = expert_counts.mean().item() if expert_counts is not None else 0
                        utilization_max = expert_counts.max().item() if expert_counts is not None else 0
                        
                        wandb.log({
                            "train/step_loss": loss.item() * grad_accum_steps,
                            "train/main_loss": main_loss.item(),
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "moe/auxiliary_loss": aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
                            "moe/dropped_assignments_percent": dropped_ratio,
                            "moe/expert_utilization_max": utilization_max,
                            "moe/expert_utilization_mean": utilization_mean
                        }, step=global_step)
                        
                        accum_dropped_assignments = 0
                        accum_total_tokens = 0

            # Validation Loop
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs, _, _, _, _ = model(images)
                        loss = criterion(outputs, labels)
                        
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
            final_val_acc = 100. * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            wandb.log({
                "val/accuracy": final_val_acc,
                "val/loss": avg_val_loss,
                "train/epoch_loss": epoch_loss / len(train_loader),
                "epoch": epoch,
            }, step=global_step)
            
            trial.report(final_val_acc, epoch)
            if trial.should_prune():
                total_runtime = time.time() - trial_start_time
                wandb.run.summary["state"] = "pruned"
                wandb.run.summary["total_runtime_seconds"] = total_runtime
                raise optuna.exceptions.TrialPruned()

        # FIX: Ensure successful runs log their timing
        total_runtime = time.time() - trial_start_time
        wandb.run.summary["state"] = "completed"
        wandb.run.summary["final_val_acc"] = final_val_acc
        wandb.run.summary["total_runtime_seconds"] = total_runtime
        
    except optuna.exceptions.TrialPruned as e:
        raise e
    finally:
        del model, optimizer, train_loader, val_loader, scheduler
        torch.cuda.empty_cache()
        if wandb.run is not None:
            wandb.finish()
    
    return final_val_acc

# ==========================================
# 4. SLURM CLUSTER ENTRY POINT
# ==========================================
if __name__ == "__main__":
    SHARED_DIR = "./optuna_slurm_logs"
    os.makedirs(SHARED_DIR, exist_ok=True)
    journal_file = os.path.join(SHARED_DIR, "vmoe_journal.log")
    
    storage = JournalStorage(JournalFileStorage(journal_file))
    sampler = optuna.samplers.TPESampler(multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10, interval_steps=5)
    
    study = optuna.create_study(
        study_name="vmoe_tpe_sweep_final",
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction="maximize"
    )
    
    N_TRIALS_PER_NODE = 7
    print(f"Node startup: attempting {N_TRIALS_PER_NODE} trials...")
    study.optimize(objective, n_trials=N_TRIALS_PER_NODE)