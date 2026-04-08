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
# 1. ARCHITECTURE (DeepMoE + EfficientNet-B0)
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

class ShallowEmbeddingNet(nn.Module):
    """
    Shallow Embedding Network from DeepMoE[cite: 28, 29].
    Extracts global features to formulate a latent mixture weight (e).
    """
    def __init__(self, num_classes, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.aux_head = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        pre_soft = self.net(x)
        e = self.softmax(pre_soft)
        aux_logits = self.aux_head(pre_soft)
        return e, aux_logits

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_prob):
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        self.hidden_dim = in_channels * expand_ratio
        self.in_channels = in_channels
        self.expand = in_channels != self.hidden_dim
        reduced_dim = max(1, int(in_channels * se_ratio))
        
        if self.expand:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.hidden_dim, eps=1e-3, momentum=0.01),
                nn.SiLU(),
            )
        else:
            self.expand_conv = nn.Identity()
            
        self.depthwise = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, stride=stride, 
                      padding=kernel_size//2, groups=self.hidden_dim, bias=False),
            nn.BatchNorm2d(self.hidden_dim, eps=1e-3, momentum=0.01),
            nn.SiLU(),
        )
        
        self.se = SqueezeExcitation(self.hidden_dim, reduced_dim)
        
        self.project = nn.Sequential(
            nn.Conv2d(self.hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        )
        
        self.stochastic_depth = StochasticDepth(p=drop_prob, mode="row")

    def forward(self, x, gate=None):
        identity = x
        
        x = self.expand_conv(x)
        
        x = self.depthwise(x)

        if gate is not None:
            x = x * gate.view(x.size(0), x.size(1), 1, 1)

        x = self.se(x)
            
        x = self.project(x)
        
        if self.use_residual:
            return identity + self.stochastic_depth(x)
        else:
            return x

class EfficientNetB0_DeepMoE(nn.Module):
    def __init__(self, num_classes=10000, dropout_rate=0.2, stochastic_depth_prob=0.2, 
                 moe_start_stage=4):
        super().__init__()
        
        self.b0_config = [
            [1, 16, 1, 1, 3],  # Stage 0
            [6, 24, 2, 2, 3],  # Stage 1
            [6, 40, 2, 2, 5],  # Stage 2
            [6, 80, 3, 2, 3],  # Stage 3
            [6, 112, 3, 1, 5], # Stage 4
            [6, 192, 4, 2, 5], # Stage 5
            [6, 320, 1, 1, 3], # Stage 6
        ]
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),
            nn.SiLU()
        )
        
        latent_dim = 32
        self.embedding_net = ShallowEmbeddingNet(num_classes, latent_dim=latent_dim)
        self.blocks = nn.ModuleList()
        self.gates = nn.ModuleDict()
        
        in_channels = 32
        total_blocks = sum([repeat for _, _, repeat, _, _ in self.b0_config])
        block_idx = 0
        
        for stage_idx, (expand_ratio, out_channels, repeats, stride, kernel_size) in enumerate(self.b0_config):
            
            for i in range(repeats):
                s = stride if i == 0 else 1
                drop_prob = stochastic_depth_prob * float(block_idx) / max(1, total_blocks - 1)
                
                block = MBConvBlock(in_channels, out_channels, kernel_size, s, 
                                    expand_ratio, se_ratio=0.25, drop_prob=drop_prob)
                self.blocks.append(block)
                
                if stage_idx >= moe_start_stage:
                    gate_linear = nn.Linear(latent_dim, block.hidden_dim, bias=True)
                    nn.init.normal_(gate_linear.weight, std=0.01)
                    nn.init.constant_(gate_linear.bias, 0.1)
                    
                    self.gates[str(block_idx)] = nn.Sequential(
                        gate_linear,
                        nn.ReLU()
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
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m.weight.shape[1] != latent_dim:
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    if m.weight.shape[1] != latent_dim:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        e, aux_logits = self.embedding_net(x)
        
        l1_loss = 0.0
        active_experts = 0.0
        total_experts = 0.0
        
        x = self.stem(x)
        for i, block in enumerate(self.blocks):
            idx_str = str(i)
            if idx_str in self.gates:
                gate = self.gates[idx_str](e)
                x = block(x, gate=gate)
                
                if self.training:
                    l1_loss += gate.abs().sum(dim=1).mean() 
                active_experts += (gate > 0).float().sum()
                total_experts += gate.numel()
            else:
                x = block(x)
                active_experts += block.hidden_dim * x.size(0)
                total_experts += block.hidden_dim * x.size(0)
                
        x = self.head(x)
        active_pct = (active_experts / max(1.0, total_experts)) if total_experts > 0 else torch.tensor(1.0, device=x.device)
        
        if self.training:
            return x, aux_logits, l1_loss, active_pct
        return x, active_pct

# ==========================================
# 2. DATA LOADING (Unchanged)
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

    train_indices = []
    val_indices = []

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
    
    # Baseline hyperparameters
    base_lr_mult = trial.suggest_float("base_lr_mult", 1e-4, 1.0, log=True)
    batch_size = 256
    grad_accum_steps = trial.suggest_int("grad_accum_steps", 1, 8)
    global_batch_size = batch_size * grad_accum_steps
    lr = base_lr_mult * (global_batch_size / 256.0)
    
    # MoE Hyperparameter Sweep
    moe_start_stage = trial.suggest_int("moe_start_stage", 0, 2)
    lambda_g = trial.suggest_float("lambda_g", 1e-6, 1.0, log=True)
    mu = trial.suggest_float("mu", 0.0, 1.0)
    
    run = wandb.init(
        project="efficientnet_tpe_sweep_deepmoe",
        group="slurm_sweep",
        name=f"trial_{trial.number}_moe_s0_lg{lambda_g:.4f}",
        config={
            "learning_rate": lr,
            "global_batch_size": global_batch_size,
            "moe_start_stage": moe_start_stage,
            "lambda_g": lambda_g,
            "mu": mu,
            "model": "EfficientNet-B0 + DeepMoE",
        },
        reinit=True
    )
    
    data_dir = '/usr/project/xtmp/inaturalistdata_store' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    num_workers = max(1, slurm_cpus - 2)
    train_loader, val_loader, num_classes = get_dataloaders(data_dir, batch_size, num_workers=num_workers)
    
    model = EfficientNetB0_DeepMoE(
        num_classes=num_classes, 
        moe_start_stage=moe_start_stage
    ).to(device)
    model = torch.compile(model)
    
    TARGET_EPOCHS = 20
    WARMUP_EPOCHS = 2
    
    effective_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    warmup_steps = int(WARMUP_EPOCHS * effective_steps_per_epoch)
    total_steps = int(TARGET_EPOCHS * effective_steps_per_epoch)
    cosine_steps = max(1, total_steps - warmup_steps)
    
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.RMSprop([
        {'params': decay_params, 'weight_decay': 1e-5},
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
    avg_val_active_pct = 0.0
    
    try:
        for epoch in range(TARGET_EPOCHS):
            model.train()
            epoch_loss = 0.0
            
            optimizer.zero_grad()
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs, aux_logits, l1_loss, active_pct = model(images)
                    loss_main = criterion(outputs, labels)
                    loss_aux = criterion(aux_logits, labels)
                    loss = (loss_main + mu * loss_aux + lambda_g * l1_loss) / grad_accum_steps
                
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
                        "train/step_loss_main": loss_main.item(),
                        "train/step_loss_aux": loss_aux.item(),
                        "train/step_l1_loss": l1_loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/active_experts_pct": active_pct.item(),
                    }, step=global_step)
                    
                global_step += 1

            # Validation Loop
            model.eval()
            val_loss = 0.0
            val_correct, val_total = 0, 0
            val_active_sum = 0.0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs, active_pct = model(images)
                        loss = criterion(outputs, labels)
                        
                    val_loss += loss.item()
                    val_active_sum += active_pct.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
            final_val_acc = 100. * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            avg_val_active_pct = val_active_sum / len(val_loader)
            
            epoch_score = final_val_acc
            penalty = 0.0
            if avg_val_active_pct > 0.3:
                penalty += (avg_val_active_pct - 0.3) * 20.0
            if avg_val_active_pct > 0.5:
                penalty += (avg_val_active_pct - 0.5) * 80.0
            epoch_score -= penalty
            
            wandb.log({
                "val/accuracy": final_val_acc,
                "val/score": epoch_score,
                "val/loss": avg_val_loss,
                "val/active_experts_pct": avg_val_active_pct,
                "train/epoch_loss": epoch_loss / len(train_loader),
                "epoch": epoch,
            }, step=global_step)

            trial.report(epoch_score, epoch)
            if trial.should_prune():
                wandb.run.summary["state"] = "pruned"
                raise optuna.exceptions.TrialPruned()

        wandb.run.summary["state"] = "completed"
        wandb.run.summary["final_val_acc"] = final_val_acc
        wandb.run.summary["final_val_active_pct"] = avg_val_active_pct
        
        optuna_score = final_val_acc
        penalty = 0.0
        if avg_val_active_pct > 0.3:
            penalty += (avg_val_active_pct - 0.3) * 20.0
        if avg_val_active_pct > 0.5:
            penalty += (avg_val_active_pct - 0.5) * 80.0
        optuna_score -= penalty
            
        wandb.run.summary["optuna_score"] = optuna_score
        
    except optuna.exceptions.TrialPruned as e:
        wandb.run.summary["state"] = "pruned"
        raise e
    except Exception as e:
        wandb.run.summary["state"] = "crashed"
        raise e
    finally:
        del model, optimizer, train_loader, val_loader, scheduler
        torch.cuda.empty_cache()
        if wandb.run is not None: wandb.finish()
    
    return optuna_score

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
    
    study = optuna.create_study(
        study_name="efficientnet_tpe_sweep_deepmoe",
        storage=storage, sampler=sampler, pruner=pruner,
        load_if_exists=True, direction="maximize"
    )
    
    N_TRIALS_PER_NODE = 30
    study.optimize(objective, n_trials=N_TRIALS_PER_NODE)