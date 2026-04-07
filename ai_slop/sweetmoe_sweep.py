import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision.ops import StochasticDepth
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import wandb
import os
import time
import math

# ==========================================
# 1. ARCHITECTURE (SE + MBConv + MoE + B0 Baseline)
# ==========================================
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(), 
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_prob):
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = int(in_channels * expand_ratio)
        self.expand = in_channels != hidden_dim
        reduced_dim = max(1, int(in_channels * se_ratio))
        
        layers = []
        if self.expand:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01),
                nn.SiLU(),
            ])
            
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, 
                      padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01),
            nn.SiLU(),
        ])
        layers.append(SqueezeExcitation(hidden_dim, reduced_dim))
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        ])
        
        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(p=drop_prob, mode="row")

    def forward(self, x):
        if self.use_residual:
            return x + self.stochastic_depth(self.block(x))
        else:
            return self.block(x)

class MoELayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_prob, num_experts):
        super().__init__()
        assert stride == 1, "MoE expects stride=1 feature alignments for spatial continuity."
        self.num_experts = num_experts
        self.hidden_dim = int(in_channels * expand_ratio)
        self.out_channels = out_channels

        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, 
                      padding=kernel_size//2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01),
            nn.SiLU()
        )
        self.se = SqueezeExcitation(in_channels, max(1, int(in_channels * se_ratio)))
        
        self.router = nn.Conv2d(in_channels, num_experts, kernel_size=1)
        nn.init.normal_(self.router.weight, std=0.01) # FIX: Stabilize early routing
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)

        # FIX: Added LayerNorm and removed biases to prevent severe performance degradation
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, self.hidden_dim, bias=False),
                nn.LayerNorm(self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, out_channels, bias=False)
            ) for _ in range(num_experts)
        ])
        
        self.shared_bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.use_residual = (in_channels == out_channels and stride == 1)
        self.stochastic_depth = StochasticDepth(p=drop_prob, mode="row")

    def forward(self, x):
        b, c, h, w = x.shape
        num_pixels = b * h * w
        
        x_spatial = self.dw(x)
        x_spatial = self.se(x_spatial)
        
        router_logits = self.router(x_spatial)
        probs = F.softmax(router_logits.float(), dim=1).to(x.dtype)
        
        # Shazeer Load Balancing Loss
        mean_probs = probs.mean(dim=(0, 2, 3))
        _, max_idx = torch.max(probs, dim=1, keepdim=True)
        mask = torch.zeros_like(probs).scatter_(1, max_idx, 1.0)
        mean_mask = mask.mean(dim=(0, 2, 3))
        aux_loss = self.num_experts * torch.sum(mean_probs * mean_mask)
        
        x_flat = x_spatial.permute(0, 2, 3, 1).reshape(num_pixels, c) 
        probs_flat = probs.permute(0, 2, 3, 1).reshape(num_pixels, self.num_experts) 
        
        top_k_probs_flat, top_k_idx_flat = torch.topk(probs_flat, 1, dim=1)
        top_k_idx_flat = top_k_idx_flat.squeeze(1) 
        
        final_flat = torch.zeros(num_pixels, self.out_channels, device=x.device, dtype=x.dtype)
        
        for i in range(self.num_experts):
            mask_i = (top_k_idx_flat == i)
            idx_i = mask_i.nonzero(as_tuple=False).squeeze(1)
            
            if idx_i.numel() > 0:
                x_gather = x_flat[idx_i]
                out_gather = self.experts[i](x_gather)
                out_gather = out_gather * top_k_probs_flat[idx_i] 
                final_flat.index_add_(0, idx_i, out_gather) # FIX: In-place operation saves VRAM

        final_out = final_flat.reshape(b, h, w, self.out_channels).permute(0, 3, 1, 2)
        final_out = self.shared_bn(final_out)

        if self.use_residual:
            return x + self.stochastic_depth(final_out), aux_loss
        else:
            return final_out, aux_loss

class EfficientNetB0_MoE(nn.Module):
    def __init__(self, num_classes=10000, dropout_rate=0.2, stochastic_depth_prob=0.2, num_experts=3, expert_expand_ratio=2.0):
        super().__init__()
        
        self.b0_config = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),
            nn.SiLU()
        )

        self.blocks = nn.ModuleList()
        in_channels = 32
        total_blocks = sum([repeat for _, _, repeat, _, _ in self.b0_config])
        block_idx = 0

        # Matches the paper's "Last 2" strategy (Final block of Stage 6, Only block of Stage 7)
        moe_target_blocks = [14, 15] 
        
        for expand_ratio, out_channels, repeats, stride, kernel_size in self.b0_config:
            for i in range(repeats):
                s = stride if i == 0 else 1
                drop_prob = stochastic_depth_prob * float(block_idx) / max(1, total_blocks - 1)
                
                if block_idx in moe_target_blocks:
                    self.blocks.append(
                        MoELayer(in_channels, out_channels, kernel_size, s, 
                                 expand_ratio=expert_expand_ratio, se_ratio=0.25, 
                                 drop_prob=drop_prob, num_experts=num_experts)
                    )
                else:
                    self.blocks.append(
                        MBConvBlock(in_channels, out_channels, kernel_size, s, 
                                    expand_ratio, se_ratio=0.25, drop_prob=drop_prob)
                    )
                in_channels = out_channels
                block_idx += 1
                
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280, eps=1e-3, momentum=0.01),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, num_classes)
        )
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        for block in self.blocks:
            if isinstance(block, MoELayer):
                x, aux = block(x)
                total_aux_loss += aux
            else:
                x = block(x)
                
        x = self.head(x)
        return x, total_aux_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 2. DATA LOADING
# ==========================================
# (Retained identical to your implementation)
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

    train_indices, val_indices = [] , []
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
    
    base_lr_mult = trial.suggest_float("base_lr_mult", 1e-4, 1.0, log=True)
    batch_size = 256
    grad_accum_steps = trial.suggest_int("grad_accum_steps", 1, 8)
    global_batch_size = batch_size * grad_accum_steps
    lr = base_lr_mult * (global_batch_size / 256.0)
    
    # Targeting 25% to 50% Sparsity (1 active / 4 total -> 1 active / 2 total)
    num_experts = trial.suggest_int("num_experts", 2, 4)
    
    # FIX: Expanded upper limit. 10.0 / num_experts easily fits the +20% constraint 
    # while allowing the NAS to search for a more performant model.
    expert_expand_ratio = trial.suggest_float("expert_expand_ratio", 6.0 / num_experts, 10.0 / num_experts)
    
    run = wandb.init(
        project="efficientnet_tpe_sweep_moe",
        group="slurm_sweep",
        name=f"trial_{trial.number}_moe_E{num_experts}_exp{expert_expand_ratio:.2f}",
        config={
            "learning_rate": lr,
            "base_lr_mult": base_lr_mult,
            "global_batch_size": global_batch_size,
            "num_experts": num_experts,
            "expert_expand_ratio": expert_expand_ratio,
            "optimizer": "RMSProp",
            "model": "EfficientNet-B0 MoE (Proxy 224x224)",
        },
        reinit=True
    )
    
    data_dir = '/usr/project/xtmp/inaturalistdata_store' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    num_workers = max(1, slurm_cpus - 2)
    train_loader, val_loader, num_classes = get_dataloaders(data_dir, batch_size, num_workers=num_workers)
    
    model = EfficientNetB0_MoE(
        num_classes=num_classes, 
        num_experts=num_experts, 
        expert_expand_ratio=expert_expand_ratio
    ).to(device)
    model = torch.compile(model)
    
    total_params = count_parameters(model)
    wandb.config.update({"total_parameters": total_params})
    print(f"Model Parameters: {total_params / 1e6:.2f}M")
    
    TARGET_EPOCHS = 20
    WARMUP_EPOCHS = 2
    
    effective_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    warmup_steps = int(WARMUP_EPOCHS * effective_steps_per_epoch)
    total_steps = int(TARGET_EPOCHS * effective_steps_per_epoch)
    cosine_steps = max(1, total_steps - warmup_steps)
    
    decay_params, no_decay_params, expert_decay_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if len(param.shape) == 1 or name.endswith(".bias") or "bn" in name.lower() or "batchnorm" in name.lower() or "layernorm" in name.lower():
            no_decay_params.append(param)
        elif "experts" in name:
            expert_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.RMSprop([
        {'params': decay_params, 'weight_decay': 1e-5},
        {'params': expert_decay_params, 'weight_decay': 1e-4},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=lr, alpha=0.9, momentum=0.9, eps=1e-3)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / cosine_steps))
        
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    global_step = 0
    final_val_acc = 0.0
    
    amp_dtype = torch.bfloat16
    device_type = 'cuda'

    try:
        for epoch in range(TARGET_EPOCHS):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()
            
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    outputs, aux_loss = model(images)
                    main_loss = criterion(outputs, labels)
                    loss = (main_loss + 0.01 * aux_loss) / grad_accum_steps
                
                loss.backward()
                epoch_loss += loss.item() * grad_accum_steps
                
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if global_step % 50 == 0:
                    wandb.log({
                        "train/step_loss": loss.item() * grad_accum_steps,
                        "train/aux_loss": aux_loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                    }, step=global_step)
                    
                global_step += 1

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    
                    with torch.autocast(device_type=device_type, dtype=amp_dtype):
                        outputs, aux_loss = model(images)
                        loss = criterion(outputs, labels)
                        
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
            final_val_acc = 100. * val_correct / val_total
            
            wandb.log({
                "val/accuracy": final_val_acc,
                "val/loss": val_loss / len(val_loader),
                "train/epoch_loss": epoch_loss / len(train_loader),
                "epoch": epoch,
            }, step=global_step)
            
            trial.report(final_val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if wandb.run is not None:
            wandb.run.summary["state"] = "completed"
            wandb.run.summary["final_val_acc"] = final_val_acc
        
    except optuna.exceptions.TrialPruned as e:
        if wandb.run is not None: wandb.run.summary["state"] = "pruned"
        raise e
    except Exception as e:
        if wandb.run is not None: wandb.run.summary["state"] = "crashed"
        raise e
    finally:
        try:
            del images, labels, outputs, loss
        except NameError:
            pass
        del model, optimizer, train_loader, val_loader, scheduler
        torch.cuda.empty_cache()
        if wandb.run is not None: wandb.finish()
    
    return final_val_acc

# ==========================================
# 4. SLURM CLUSTER ENTRY POINT
# ==========================================
if __name__ == "__main__":
    SHARED_DIR = "./optuna_slurm_logs"
    os.makedirs(SHARED_DIR, exist_ok=True)
    journal_file = os.path.join(SHARED_DIR, "optuna_moe_journal.log")
    
    storage = JournalStorage(JournalFileStorage(journal_file))
    sampler = optuna.samplers.TPESampler(multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=10, interval_steps=5)
    
    study = optuna.create_study(
        study_name="efficientnet_moe_sweep",
        storage=storage, sampler=sampler, pruner=pruner,
        load_if_exists=True, direction="maximize"
    )
    
    study.optimize(objective, n_trials=30)