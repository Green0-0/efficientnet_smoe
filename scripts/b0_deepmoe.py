# AI added the deepmoe portion, also transplanted some code from the old tests

import math
import os
import time
import optuna
import wandb

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ConstantLR
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

from optuna.storages import JournalStorage, JournalFileStorage
from training_utils import get_cosine_schedule_with_warmup, get_dataloaders, train_loop

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.efficientnet import MBConv

class ShallowEmbeddingNet(nn.Module):
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
            nn.Linear(256, latent_dim),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.aux_head = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        pre_soft = self.net(x)
        return self.softmax(pre_soft), self.aux_head(pre_soft)

class GatedMBConvWrapper(nn.Module):
    """
    Takes a pre-trained torchvision MBConv block, rips out its layers, 
    and reconstructs the forward pass to inject a gate BEFORE the SE layer.
    """
    def __init__(self, tv_mbconv):
        super().__init__()
        # Extract standard torchvision layers from the sequential block
        layers = list(tv_mbconv.block.children())

        # Stage 1 has no expansion phase, so it has 3 layers. The rest have 4.
        if len(layers) == 4: 
            self.expand = layers[0]
            self.depthwise = layers[1]
            self.se = layers[2]
            self.project = layers[3]
        else: 
            self.expand = nn.Identity()
            self.depthwise = layers[0]
            self.se = layers[1]
            self.project = layers[2]

        # Preserve the original configuration
        self.use_res_connect = tv_mbconv.use_res_connect
        self.stochastic_depth = tv_mbconv.stochastic_depth
        
        # We need the hidden dimension to size the MoE Linear Layer correctly
        self.hidden_dim = self.depthwise[0].out_channels

    def forward(self, x, gate=None):
        identity = x

        # 1. Expand & Depthwise
        x = self.expand(x)
        x = self.depthwise(x)

        # 2. ---> INJECT DEEPMOE GATE HERE <---
        if gate is not None:
            # Reshape from [Batch, Channels] to [Batch, Channels, 1, 1]
            x = x * gate.view(x.size(0), x.size(1), 1, 1)

        # 3. SE & Project
        x = self.se(x)
        x = self.project(x)

        # 4. Residual Connection
        if self.use_res_connect:
            return identity + self.stochastic_depth(x)
        return x

class TransferDeepMoEEfficientNet(nn.Module):
    def __init__(self, num_classes=1010, latent_dim=32, moe_start_stage=4):
        super().__init__()
        
        # 1. Load the Pre-trained Torchvision Model
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Replace the final classification head
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)
        
        # 2. Setup the MoE Components
        self.embedding_net = ShallowEmbeddingNet(num_classes, latent_dim)
        self.gates = nn.ModuleDict()
        
        # 3. Hijack the blocks to inject the gates
        block_idx = 0
        # features[1] through features[7] contain the sequences of MBConv blocks
        for stage_idx in range(1, 8):
            stage = self.base_model.features[stage_idx]
            
            for i in range(len(stage)):
                module = stage[i]
                if isinstance(module, MBConv):
                    # Wrap the pre-trained block
                    gated_module = GatedMBConvWrapper(module)
                    # Overwrite the original block in the sequential container
                    stage[i] = gated_module
                    
                    # Create the gating linear layer if past the start stage
                    # (Note: torchvision stage_idx starts at 1, so we adjust your start_stage logic)
                    if stage_idx >= moe_start_stage:
                        gate_linear = nn.Linear(latent_dim, gated_module.hidden_dim)
                        nn.init.normal_(gate_linear.weight, std=0.01)
                        nn.init.constant_(gate_linear.bias, 1)
                        self.gates[str(block_idx)] = nn.Sequential(gate_linear, nn.ReLU())
                    
                    block_idx += 1

    def forward(self, x):
        e, aux_logits = self.embedding_net(x)

        l1_loss = 0.0
        active_experts = 0.0
        total_experts = 0.0

        # We must manually iterate through the base_model features to pass the gate tensor
        x = self.base_model.features[0](x) # Stem

        block_idx = 0
        for stage_idx in range(1, 8):
            stage = self.base_model.features[stage_idx]
            for module in stage:
                idx_str = str(block_idx)
                if idx_str in self.gates:
                    gate = self.gates[idx_str](e)
                    x = module(x, gate=gate)

                    if self.training:
                        l1_loss += gate.abs().sum(dim=1).mean()
                    active_experts += (gate > 0).float().sum()
                    total_experts += gate.numel()
                else:
                    x = module(x) # Execute normally without gating
                    active_experts += module.hidden_dim * x.size(0)
                    total_experts += module.hidden_dim * x.size(0)
                
                block_idx += 1

        x = self.base_model.features[8](x) # Head convolution
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.classifier(x) # Final Logits

        active_pct = (active_experts / max(1.0, total_experts)) if total_experts > 0 else torch.tensor(1.0, device=x.device)

        if self.training:
            return x, aux_logits, l1_loss, active_pct
        return x, active_pct

def objective(trial):
    trial_start_time = time.time()
    final_val_acc = 0
    
    BATCH_SIZE = 256
    GRAD_ACCUM_STEPS = trial.suggest_int("grad_accum_steps", 4, 16, 2)
    EPOCHS_HEAD = trial.suggest_int("epochs_head", 0, 5, 1)
    EPOCHS_BODY = 10 - EPOCHS_HEAD

    lr_head_mul = trial.suggest_float("lr_head_mul", 0.0001, 0.01, log=True)
    lr_head2_mul = trial.suggest_float("lr_head2_mul", 0.0001, 0.01, log=True)
    lr_body_mul = trial.suggest_float("lr_body_mul", 0.0001, 0.01, log=True)
    LR_HEAD = lr_head_mul * math.sqrt(GRAD_ACCUM_STEPS) 
    LR_HEAD2 = lr_head2_mul * math.sqrt(GRAD_ACCUM_STEPS) 
    LR_BODY = lr_body_mul * math.sqrt(GRAD_ACCUM_STEPS)  
    
    WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    
    train_loader, val_loader, num_classes = get_dataloaders(BATCH_SIZE)

    run = wandb.init(
        project="efficientnet_b0_TL_baseline",
        group="sweep",
        name=f"device_bs{BATCH_SIZE}_accum{GRAD_ACCUM_STEPS}_lr{LR_HEAD}",
        config={
            "head_LR": LR_HEAD,
            "body_LR": LR_BODY,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "epochs_head": EPOCHS_HEAD,
            "epochs_body": EPOCHS_BODY
        },
        reinit=True,
    )
    
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes) # Reset head for iNaturalist
    head_params = []
    body_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            head_params.append(param)
        else:
            body_params.append(param)

    if EPOCHS_HEAD != 0:
        head_optimizer = torch.optim.AdamW(head_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

        scheduler = ConstantLR(head_optimizer, factor=1, total_iters=1)
        final_val_acc, pruned = train_loop(model, head_optimizer, scheduler, EPOCHS_HEAD, GRAD_ACCUM_STEPS, train_loader, val_loader, trial, 0)
        if pruned:
            total_runtime = time.time() - trial_start_time
            wandb.run.summary["state"] = "pruned"
            wandb.run.summary["total_runtime_seconds"] = total_runtime
            wandb.run.summary["final_val_acc"] = final_val_acc
            wandb.finish()
            del model, head_optimizer, train_loader, val_loader
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
    
    for param in model.parameters():
        param.requires_grad = True

    body_optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": LR_HEAD2},
        {"params": body_params, "lr": LR_BODY},
    ], weight_decay=WEIGHT_DECAY)

    effective_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_body_steps = int(EPOCHS_BODY * effective_steps_per_epoch)
    body_scheduler = get_cosine_schedule_with_warmup(body_optimizer, total_body_steps * 0.1, total_body_steps)
    final_val_acc, pruned = train_loop(model, body_optimizer, body_scheduler, EPOCHS_BODY, GRAD_ACCUM_STEPS, train_loader, val_loader, trial, EPOCHS_HEAD)
    total_runtime = time.time() - trial_start_time
    wandb.run.summary["total_runtime_seconds"] = total_runtime
    wandb.run.summary["final_val_acc"] = final_val_acc
    if pruned:
        wandb.run.summary["state"] = "pruned"
        wandb.finish()
        del model, body_optimizer, train_loader, val_loader
        if EPOCHS_HEAD != 0:
            del head_optimizer
        torch.cuda.empty_cache()
        raise optuna.exceptions.TrialPruned()
    else:
        wandb.run.summary["state"] = "completed"
        wandb.finish()
        del model, body_optimizer, train_loader, val_loader
        if EPOCHS_HEAD != 0:
            del head_optimizer
        torch.cuda.empty_cache()
    return final_val_acc

if __name__ == "__main__":
    SHARED_DIR = "./optuna_slurm_logs"
    os.makedirs(SHARED_DIR, exist_ok=True)
    journal_file = os.path.join(SHARED_DIR, "journal.log")

    storage = JournalStorage(JournalFileStorage(journal_file))

    sampler = optuna.samplers.TPESampler(multivariate=True)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10, n_warmup_steps=8, interval_steps=3
    )

    study = optuna.create_study(
        study_name="b0_baseline_t0",
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction="maximize",
    )

    N_TRIALS_PER_NODE = 10
    print(f"Node startup: attempting {N_TRIALS_PER_NODE} trials...")

    study.optimize(objective, n_trials=N_TRIALS_PER_NODE)