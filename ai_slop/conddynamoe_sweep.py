"""WARNING:

This script is only for safekeeping. There are currently several issues:

- Model is bigger by more than 20%
- Model is forcefully shrunken in some areas to be even close to the 20% param limit.
- Validation drop_last=True
- Lost trial cleanup on trial end
"""

import torch
import torch.nn as nn
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
# 1. ARCHITECTURE (CondConv / DynaConv MoE)
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

class MoEPointwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=4):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels, 1, 1))
        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            for i in range(self.weight.size(0)):
                nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def forward(self, x, routing_weights):
        b, c_in, h, w = x.size()
        c_out = self.weight.size(1)
        
        agg_weights = torch.matmul(
            routing_weights, 
            self.weight.view(self.weight.size(0), -1)
        ).view(b, c_out, c_in, 1, 1)

        # Batch-wise dynamic grouped conv trick (Pointwise)
        x_reshaped = x.view(1, b * c_in, h, w)
        agg_weights_reshaped = agg_weights.view(b * c_out, c_in, 1, 1)
        
        out = torch.nn.functional.conv2d(
            x_reshaped, 
            agg_weights_reshaped, 
            bias=None, stride=1, padding=0, 
            groups=b 
        )
        return out.view(b, c_out, h, w)

class MoEDepthwiseConv2d(nn.Module):
    def __init__(self, channels, kernel_size, stride, num_experts=4):
        super().__init__()
        self.stride = stride
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.Tensor(num_experts, channels, 1, kernel_size, kernel_size))
        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            for i in range(self.weight.size(0)):
                nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def forward(self, x, routing_weights):
        b, c, h, w = x.size()
        
        agg_weights = torch.matmul(
            routing_weights, 
            self.weight.view(self.weight.size(0), -1)
        ).view(b, c, 1, self.weight.size(3), self.weight.size(4))

        # Batch-wise dynamic grouped conv trick (Depthwise)
        x_reshaped = x.view(1, b * c, h, w)
        agg_weights_reshaped = agg_weights.view(b * c, 1, self.weight.size(3), self.weight.size(4))
        
        out = torch.nn.functional.conv2d(
            x_reshaped, 
            agg_weights_reshaped, 
            bias=None, stride=self.stride, padding=self.padding, 
            groups=b * c 
        )
        return out.view(b, c, out.size(2), out.size(3))

class MBConvBlockMoE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_prob,
                 use_moe_project=False, use_moe_dw=False, num_experts=4, routing_fn='softmax'):
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        self.use_moe_project = use_moe_project
        self.use_moe_dw = use_moe_dw
        self.routing_fn = routing_fn
        self.num_experts = num_experts
        self.has_moe = use_moe_project or use_moe_dw
        
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = max(1, int(in_channels * se_ratio))
        
        if self.has_moe:
            if self.routing_fn == 'softmax':  # DynaConv Implementation: Squeezed dimensionality
                self.routing_net = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(in_channels, max(16, in_channels // 4)),
                    nn.ReLU(),
                    nn.Linear(max(16, in_channels // 4), num_experts)
                )
            else:  # CondConv Implementation: Direct mapping
                self.routing_net = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(in_channels, num_experts)
                )
            self.last_routing_weights = None

        # Expand conv remains static to prevent extreme parameter explosion
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, 1, bias=False) if self.expand else nn.Identity()
        self.bn1 = nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01) if self.expand else nn.Identity()
        self.act1 = nn.SiLU() if self.expand else nn.Identity()

        self.dw_conv = MoEDepthwiseConv2d(hidden_dim, kernel_size, stride, num_experts) if self.use_moe_dw else \
                       nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=kernel_size//2, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01)
        self.act2 = nn.SiLU()
        
        self.se = SqueezeExcitation(hidden_dim, reduced_dim)
        
        # DynaConv paper strongly recommends making the projection conv dynamic if prioritizing a single layer
        self.project_conv = MoEPointwiseConv2d(hidden_dim, out_channels, num_experts) if self.use_moe_project else \
                            nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        
        self.stochastic_depth = StochasticDepth(p=drop_prob, mode="row")

    def forward(self, x, temperature=1.0):
        routing_weights = None
        
        if self.has_moe:
            logits = self.routing_net(x).float()
            if self.routing_fn == 'softmax':
                rw = torch.softmax(logits / temperature, dim=-1)
            else:
                rw = torch.sigmoid(logits)
            
            self.last_routing_weights = rw.detach()
            routing_weights = rw.to(x.dtype)

        # Execution
        out = x
        if self.expand:
            out = self.expand_conv(out)
            out = self.act1(self.bn1(out))
            
        out = self.dw_conv(out, routing_weights) if self.use_moe_dw else self.dw_conv(out)
        out = self.act2(self.bn2(out))
        out = self.se(out)
        
        out = self.project_conv(out, routing_weights) if self.use_moe_project else self.project_conv(out)
        out = self.bn3(out)
        
        if self.use_residual:
            return x + self.stochastic_depth(out)
        return out

class EfficientNetB0(nn.Module):
    # Set base width to 0.825 to drop original params to ~2.7M. 
    def __init__(self, num_classes=10000, width_mult=0.825, dropout_rate=0.2, stochastic_depth_prob=0.2, 
                 moe_routing="softmax", num_experts=4):
        super().__init__()
        
        b0_config = [
            [1, 16, 1, 1, 3], [6, 24, 2, 2, 3], [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3], [6, 112, 3, 1, 5], [6, 192, 4, 2, 5], [6, 320, 1, 1, 3]
        ]
        
        def scale_width(w):
            w *= width_mult
            new_w = max(8, int(w + 4) // 8 * 8)
            if new_w < 0.9 * w: new_w += 8
            return int(new_w)
            
        in_channels = scale_width(32)
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01),
            nn.SiLU()
        )
        
        self.blocks = nn.ModuleList()
        total_blocks = sum([c[2] for c in b0_config])
        block_idx = 0
        
        for stage_idx, (expand_ratio, out_channels, repeats, stride, kernel_size) in enumerate(b0_config):
            out_channels = scale_width(out_channels)
            
            # Applying MoE to Project and DW hits the 30-50% sparsity goal without blowing up the 20% budget
            use_moe_dw = True 
            
            # THE FIX: Restrict Pointwise MoE to stages 4 and 5 only to stay under the 4.8M parameter cap
            use_moe_project = (4 <= stage_idx < 6)
            
            for i in range(repeats):
                s = stride if i == 0 else 1
                drop_prob = stochastic_depth_prob * float(block_idx) / max(1, total_blocks - 1)
                self.blocks.append(
                    MBConvBlockMoE(in_channels, out_channels, kernel_size, s, expand_ratio, 0.25, drop_prob,
                                   use_moe_project=use_moe_project, use_moe_dw=use_moe_dw, 
                                   num_experts=num_experts, routing_fn=moe_routing)
                )
                in_channels = out_channels
                block_idx += 1
                
        last_channels = 1280 
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, last_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channels, eps=1e-3, momentum=0.01),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(last_channels, num_classes)
        )

        # Restore weight initialization for static components
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, temperature=1.0):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x, temperature=temperature)
        return self.head(x)

# ==========================================
# 2. DATA LOADING (iNat 2021 Mini Proxy)
# ==========================================
def get_dataloaders(data_dir, batch_size, num_workers=8):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train_dataset = datasets.INaturalist(root=data_dir, version='2021_train_mini', transform=train_transforms, download=True)
    full_val_dataset = datasets.INaturalist(root=data_dir, version='2021_train_mini', transform=val_transforms, download=True)
    num_classes = len(full_train_dataset.all_categories)

    class_indices = {}
    for i, (cat_id, _) in enumerate(full_train_dataset.index):
        if cat_id not in class_indices: class_indices[cat_id] = []
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

    # drop_last=True prevents Variable-batch-size torch.compile graph re-evaluations
    train_loader = DataLoader(torch.utils.data.Subset(full_train_dataset, train_indices), batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(torch.utils.data.Subset(full_val_dataset, val_indices), batch_size=batch_size, shuffle=False, 
                            num_workers=min(2, num_workers), pin_memory=True, persistent_workers=True, drop_last=True)
    return train_loader, val_loader, num_classes

# ==========================================
# 3. OPTUNA + WANDB OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    trial_start_time = time.time()
    
    # Sane Optimizer Sweeps
    base_lr_mult = trial.suggest_float("base_lr_mult", 0.0005, 0.005)
    batch_size = 128
    grad_accum_steps = trial.suggest_int("grad_accum_steps", 2, 4)
    global_batch_size = batch_size * grad_accum_steps
    lr = base_lr_mult * (global_batch_size / 256.0)
    
    # MoE Sweeps
    moe_routing = trial.suggest_categorical("moe_routing", ["softmax", "sigmoid"])
    num_experts = 4 

    run = wandb.init(
        project="efficientnet_moe_sweep", group="slurm_sweep",
        name=f"trial_{trial.number}_{moe_routing}_lr{base_lr_mult:.4f}",
        config={
            "learning_rate": lr, "batch_size": batch_size, "grad_accum_steps": grad_accum_steps,
            "moe_routing": moe_routing, "width_mult": 0.825, "num_experts": num_experts
        },
        reinit=True
    )
    
    data_dir = '/usr/project/xtmp/inaturalistdata_store' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', 4)) - 2)
    
    train_loader, val_loader, num_classes = get_dataloaders(data_dir, batch_size, num_workers=num_workers)
    
    model = EfficientNetB0(
        num_classes=num_classes, width_mult=0.825,
        moe_routing=moe_routing, num_experts=num_experts
    ).to(device)
    
    model = torch.compile(model)
    
    TARGET_EPOCHS = 20
    WARMUP_EPOCHS = 2
    ANNEAL_EPOCHS = 10
    
    effective_steps = math.ceil(len(train_loader) / grad_accum_steps)
    warmup_steps = int(WARMUP_EPOCHS * effective_steps)
    total_steps = int(TARGET_EPOCHS * effective_steps)
    
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if len(param.shape) == 1 or name.endswith(".bias"): no_decay_params.append(param)
        else: decay_params.append(param)

    optimizer = torch.optim.RMSprop([
        {'params': decay_params, 'weight_decay': 1e-5},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=lr, alpha=0.9, momentum=0.9, eps=1e-3)

    def lr_lambda(step):
        if step < warmup_steps: return float(step + 1) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    global_step = 0
    
    try:
        for epoch in range(TARGET_EPOCHS):
            model.train()
            epoch_loss = 0.0
            
            # DynaConv Temperature Annealing (30.0 -> 1.0 linearly)
            current_temp = 1.0
            if moe_routing == "softmax":
                current_temp = max(1.0, 30.0 - (29.0 * (epoch / max(1, ANNEAL_EPOCHS - 1))))

            optimizer.zero_grad()
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(images, temperature=current_temp)
                    loss = criterion(outputs, labels) / grad_accum_steps
                
                loss.backward()
                epoch_loss += loss.item() * grad_accum_steps
                
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                global_step += 1

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            moe_stats = {}
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        # Reverted back to current_temp to avoid Batch Norm covariate shift issues
                        outputs = model(images, temperature=current_temp) 
                        val_loss += criterion(outputs, labels).item()
                        
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    for name, module in model.named_modules():
                        if isinstance(module, MBConvBlockMoE) and getattr(module, 'last_routing_weights', None) is not None:
                            moe_stats.setdefault(name, []).append(module.last_routing_weights.mean(dim=0).cpu())
            
            log_dict = {}
            for name, weight_list in moe_stats.items():
                avg_weights = torch.stack(weight_list).mean(dim=0).numpy()
                for exp_idx, weight in enumerate(avg_weights):
                    log_dict[f"moe_routing/{name}_expert_{exp_idx}_util"] = weight
            
            final_val_acc = 100. * val_correct / val_total
            log_dict.update({
                "val/accuracy": final_val_acc, 
                "val/loss": val_loss / len(val_loader), 
                "epoch": epoch, 
                "moe_temp": current_temp
            })
            wandb.log(log_dict, step=global_step)
            
            trial.report(final_val_acc, epoch)
            if trial.should_prune():
                total_runtime = time.time() - trial_start_time
                wandb.run.summary["state"] = "pruned"
                wandb.run.summary["total_runtime_seconds"] = total_runtime
                wandb.run.summary["final_val_acc"] = final_val_acc
                raise optuna.exceptions.TrialPruned()

        wandb.run.summary.update({"state": "completed", "final_val_acc": final_val_acc, "total_runtime_seconds": time.time() - trial_start_time})
        
    except optuna.exceptions.TrialPruned as e:
        raise e
    except Exception as e:
        wandb.run.summary["state"] = "crashed"
        raise e
    finally:
        if wandb.run is not None:
            wandb.finish()
    
    return final_val_acc

# ==========================================
# 4. SLURM CLUSTER ENTRY POINT
# ==========================================
if __name__ == "__main__":
    SHARED_DIR = "./optuna_slurm_logs"
    os.makedirs(SHARED_DIR, exist_ok=True)
    journal_file = os.path.join(SHARED_DIR, "optuna_journal.log")
    
    storage = JournalStorage(JournalFileStorage(journal_file))
    sampler = optuna.samplers.TPESampler(multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=10, interval_steps=5)
    
    study = optuna.create_study(study_name="efficientnet_moe_sweep_final", storage=storage, sampler=sampler, pruner=pruner, load_if_exists=True, direction="maximize")
    study.optimize(objective, n_trials=30)