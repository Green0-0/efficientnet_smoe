# AI added the deepmoe portion, also transplanted some code from the old/preliminary tests that worked well (ref: ai_slop/deepmoe_sweep.py)

import math
import os
import time
import optuna
import wandb

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ConstantLR
import torchvision.models as models

from optuna.storages import JournalStorage, JournalFileStorage
from training_utils import get_cosine_schedule_with_warmup, get_dataloaders, train_loop_deepmoe

import torch
import torch.nn as nn
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
    def __init__(self, model_id=0, num_classes=1010, latent_dim=32, moe_start_stage=4, reference_flops=None):
        super().__init__()
        
        # 1. Load the Pre-trained Torchvision Model
        if model_id == 0: # Note: The B1/B2/B3 models are at a slight disadvantage due to the cropped image, but this is fine
            self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        elif model_id == 1:
            self.base_model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        elif model_id == 2:
            self.base_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        elif model_id == 3:
            self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model_type: {model_id}")
        for param in self.base_model.parameters():
            param.requires_grad = False
        
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
        self._profile_flops(image_size=224)
        
        self.reference_flops = reference_flops if reference_flops is not None else self.total_baseline_flops

    def forward(self, x):
        e, aux_logits = self.embedding_net(x)

        l1_loss = 0.0
        active_experts = 0.0
        total_experts = 0.0
        active_body_flops = 0.0
        
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
                    
                    active_channels = (gate > 0).float().sum(dim=1) # Shape: [Batch_Size]
                    active_body_flops += (active_channels * self.flops_per_channel[idx_str]).mean()
                else:
                    x = module(x) # Execute normally without gating
                    active_experts += module.hidden_dim * x.size(0)
                    total_experts += module.hidden_dim * x.size(0)
                    
                    active_body_flops += (module.hidden_dim * self.flops_per_channel[idx_str])
                
                block_idx += 1

        x = self.base_model.features[8](x) # Head convolution
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.classifier(x) # Final Logits

        active_pct = (active_experts / max(1.0, total_experts)) if total_experts > 0 else torch.tensor(1.0, device=x.device)
        
        total_active_flops = active_body_flops + self.static_overhead_flops
        
        flop_retention_pct = total_active_flops / self.reference_flops

        if self.training:
            return x, aux_logits, l1_loss, active_pct, flop_retention_pct
        return x, active_pct, flop_retention_pct
    
    def _profile_flops(self, image_size=224):
        """
        Pushes a dummy tensor through the network to dynamically calculate
        both the static overhead and the per-channel dynamic costs.
        """
        dummy_input = torch.zeros(1, 3, image_size, image_size)
        
        # Helper function to count MACs for standard layers
        def count_macs(module, x):
            if isinstance(module, nn.Conv2d):
                out = module(x)
                macs = out.size(1) * (module.in_channels // module.groups) * \
                    module.kernel_size[0] * module.kernel_size[1] * out.size(2) * out.size(3)
                return out, float(macs)
            elif isinstance(module, nn.Linear):
                out = module(x)
                macs = module.out_features * module.in_features
                return out, float(macs)
            elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.Flatten, nn.Identity)):
                return module(x), 0.0
            else:
                # Activations, BatchNorms, Dropouts cost negligible MACs compared to matmuls
                return module(x), 0.0

        self.static_overhead_flops = 0.0
        self.flops_per_channel = {}
        self.total_baseline_body_flops = 0.0

        # --- 1. Profile the Shallow Embedding Network ---
        embed_x = dummy_input
        for layer in self.embedding_net.net:
            embed_x, macs = count_macs(layer, embed_x)
            self.static_overhead_flops += macs
        _, macs = count_macs(self.embedding_net.aux_head, embed_x)
        self.static_overhead_flops += macs

        # --- 2. Profile the Base Stem ---
        x = dummy_input
        for layer in self.base_model.features[0]: 
            x, macs = count_macs(layer, x)
            self.static_overhead_flops += macs

        # --- 3. Profile the MBConv Body (Dynamic per channel) ---
        block_idx = 0
        for stage_idx in range(1, 8):
            for module in self.base_model.features[stage_idx]:
                # Expand
                H_in, W_in = x.shape[2], x.shape[3]
                x = module.expand(x)
                
                # Depthwise
                x = module.depthwise(x)
                H_out, W_out = x.shape[2], x.shape[3]
                
                # Extract channel configs
                c_in = module.expand[0].in_channels if not isinstance(module.expand, nn.Identity) else module.depthwise[0].in_channels
                k = module.depthwise[0].kernel_size[0]
                c_reduced = module.se.fc1.out_channels
                c_out = module.project[0].out_channels
                
                # Calculate MACs for exactly ONE hidden channel
                macs_expand = c_in * H_in * W_in
                macs_depth = (k * k) * H_out * W_out
                macs_se = 2 * c_reduced
                macs_project = c_out * H_out * W_out
                
                cost_per_channel = float(macs_expand + macs_depth + macs_se + macs_project)
                
                self.flops_per_channel[str(block_idx)] = cost_per_channel
                self.total_baseline_body_flops += (cost_per_channel * module.hidden_dim)
                
                x = module.se(x)
                x = module.project(x)
                block_idx += 1

        # --- 4. Profile the Final Head ---
        for layer in self.base_model.features[8]: # Final Conv2d sequence
            x, macs = count_macs(layer, x)
            self.static_overhead_flops += macs

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Final Classifier
        x, macs = count_macs(self.base_model.classifier, x)
        self.static_overhead_flops += macs

        # Calculate the ultimate baseline computation of the un-pruned network
        self.total_baseline_flops = self.static_overhead_flops + self.total_baseline_body_flops

def objective(trial, b0_reference_flops):
    trial_start_time = time.time()
    final_val_acc = 0
    
    BATCH_SIZE = 256
    GRAD_ACCUM_STEPS = trial.suggest_int("grad_accum_steps", 4, 16, 2)
    EPOCHS_HEAD = trial.suggest_int("epochs_head", 0, 5, 1)
    EPOCHS_BODY = 10 - EPOCHS_HEAD

    model_id = trial.suggest_int("model_id", 0, 2, 1)
    mu = trial.suggest_float("mu", 1e-6, 1, log=True)
    lambda_g = trial.suggest_float("lambda_g", 1e-7, 0.001, log=True)
    lr_head_mul = trial.suggest_float("lr_head_mul", 0.0001, 0.01, log=True)
    lr_head2_mul = trial.suggest_float("lr_head2_mul", 0.0001, 0.01, log=True)
    lr_body_mul = trial.suggest_float("lr_body_mul", 0.0001, 0.01, log=True)
    LR_HEAD = lr_head_mul * math.sqrt(GRAD_ACCUM_STEPS) 
    LR_HEAD2 = lr_head2_mul * math.sqrt(GRAD_ACCUM_STEPS) 
    LR_BODY = lr_body_mul * math.sqrt(GRAD_ACCUM_STEPS)  
    
    WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    
    train_loader, val_loader, num_classes = get_dataloaders(BATCH_SIZE)

    run = wandb.init(
        project="efficientnet_deepmoe",
        group="sweep",
        name=f"id{model_id}_lam{lambda_g:.1e}_lr{LR_HEAD:.1e}",
        config={
            "model_id": model_id,
            "mu": mu,
            "lambda_g": lambda_g,
            "head_LR": LR_HEAD,
            "body_LR": LR_BODY,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "epochs_head": EPOCHS_HEAD,
            "epochs_body": EPOCHS_BODY
        },
        reinit=True,
    )
    
    model = TransferDeepMoEEfficientNet(model_id=model_id, reference_flops=b0_reference_flops)
    
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
        final_val_acc, pruned = train_loop_deepmoe(model, head_optimizer, scheduler, EPOCHS_HEAD, GRAD_ACCUM_STEPS, train_loader, val_loader, trial, 0, lambda_g, mu)
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
    final_score, pruned = train_loop_deepmoe(model, body_optimizer, body_scheduler, EPOCHS_BODY, GRAD_ACCUM_STEPS, train_loader, val_loader, trial, EPOCHS_HEAD, lambda_g, mu)
    total_runtime = time.time() - trial_start_time
    wandb.run.summary["total_runtime_seconds"] = total_runtime
    wandb.run.summary["final_score"] = final_score
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
    return final_score

if __name__ == "__main__":
    SHARED_DIR = "./optuna_slurm_logs"
    os.makedirs(SHARED_DIR, exist_ok=True)
    
    print("Calculating B0 reference FLOPs...")
    dummy_b0 = TransferDeepMoEEfficientNet(model_id=0)
    B0_REFERENCE_FLOPS = dummy_b0.total_baseline_flops
    del dummy_b0
    print(f"B0 Reference FLOPs locked at: {B0_REFERENCE_FLOPS}")
    
    journal_file = os.path.join(SHARED_DIR, "journal.log")

    storage = JournalStorage(JournalFileStorage(journal_file))

    sampler = optuna.samplers.TPESampler(multivariate=True)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10, n_warmup_steps=4, interval_steps=2
    )

    study = optuna.create_study(
        study_name="deepmoe",
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction="maximize",
    )

    N_TRIALS_PER_NODE = 10
    print(f"Node startup: attempting {N_TRIALS_PER_NODE} trials...")

    study.optimize(lambda trial: objective(trial, B0_REFERENCE_FLOPS), n_trials=N_TRIALS_PER_NODE)