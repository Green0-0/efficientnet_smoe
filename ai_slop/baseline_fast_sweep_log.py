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
# 1. ARCHITECTURE (Squeeze & Excitation + MBConv + B0 Baseline)
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
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels * se_ratio)
        
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


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=10000, dropout_rate=0.2, stochastic_depth_prob=0.2):
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
        
        for expand_ratio, out_channels, repeats, stride, kernel_size in self.b0_config:
            for i in range(repeats):
                s = stride if i == 0 else 1
                drop_prob = stochastic_depth_prob * float(block_idx) / max(1, total_blocks - 1)
                
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

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x

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
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # WARNING: You should NOT submit 2 jobs while the dataset is still downloading, or it will become corrupted
    # This is intended behavior...
    # Also, reusing the train dataset to prevent val leakage from optuna hyperparameter sweeping
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
    
    # Adhere to linear scaling dynamically for batch size sweeping
    base_lr_mult = trial.suggest_float("base_lr_mult", 1e-4, 1.0, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    lr = base_lr_mult * (batch_size / 256.0)
    
    # Initialize WandB
    run = wandb.init(
        project="efficientnet_tpe_sweep",
        group="slurm_sweep",
        name=f"trial_{trial.number}_bs{batch_size}",
        config={
            "learning_rate": lr,
            "base_lr_mult": base_lr_mult,
            "batch_size": batch_size,
            "optimizer": "RMSProp",
            "model": "EfficientNet-B0 (Proxy 224x224)",
            "dataset": "iNaturalist 2021 Mini"
        },
        reinit=True
    )
    
    # WARNING: THE DATASET IS 44GB!
    data_dir = '/usr/project/xtmp/inaturalistdata_store' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    num_workers = max(1, slurm_cpus - 2)
    train_loader, val_loader, num_classes = get_dataloaders(data_dir, batch_size, num_workers=num_workers)
    model = EfficientNetB0(num_classes=num_classes).to(device)
    
    TARGET_EPOCHS = 20
    WARMUP_EPOCHS = 2
    
    warmup_steps = int(WARMUP_EPOCHS * len(train_loader))
    total_steps = int(TARGET_EPOCHS * len(train_loader))
    cosine_steps = max(1, total_steps - warmup_steps)
    
    decay_params = []
    no_decay_params = []
    
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
        else:
            decay_step = step - warmup_steps
            return 0.5 * (1.0 + math.cos(math.pi * decay_step / cosine_steps))
        
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"LR: {lr:.6f} | Batch Size: {batch_size}")
    
    global_step = 0
    final_val_acc = 0.0
    
    try:
        for epoch in range(TARGET_EPOCHS):
            model.train()
            epoch_loss = 0.0
            
            for images, labels in train_loader:

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                
                if global_step % 50 == 0:
                    wandb.log({
                        "train/step_loss": loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                    }, step=global_step)
                    
                global_step += 1

            # Validation Loop
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        outputs = model(images)
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
                wandb.run.summary["final_val_acc"] = final_val_acc
                raise optuna.exceptions.TrialPruned()

        total_runtime = time.time() - trial_start_time
        print(f"\nTrial {trial.number} completed in {total_runtime:.2f} seconds.")
        
        wandb.run.summary["state"] = "completed"
        wandb.run.summary["final_val_acc"] = final_val_acc
        wandb.run.summary["total_runtime_seconds"] = total_runtime
        
    except optuna.exceptions.TrialPruned as e:
        raise e
    except Exception as e:
        wandb.run.summary["state"] = "crashed"
        raise e
    finally:
        try:
            del images, labels, outputs, loss
        except NameError:
            pass
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
    journal_file = os.path.join(SHARED_DIR, "optuna_journal.log")
    
    storage = JournalStorage(JournalFileStorage(journal_file))
    
    sampler = optuna.samplers.TPESampler(multivariate=True)
    
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=20,
        n_warmup_steps=10, 
        interval_steps=5
    )
    
    study = optuna.create_study(
        study_name="efficientnet_tpe_sweep",
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction="maximize"
    )
    
    N_TRIALS_PER_NODE = 30
    print(f"Node startup: attempting {N_TRIALS_PER_NODE} trials...")
    
    study.optimize(objective, n_trials=N_TRIALS_PER_NODE)