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

from scripts.flop_profiler import profile_deepmoe_flops

class ShallowEmbeddingNet(nn.Module):
    def __init__(self, num_classes, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.SiLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, latent_dim),
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
        x = self.expand(x)

        x = self.depthwise(x)
        
        """Notes:
        The gate MUST be after the depthwise and before the SE. 
        This is because:
        1. Placing the gate after expand leads to corruption from the depthwise's batchnorm, which:
            - Causes the zeros from the gate to be lost
            - Pollutes the batch norm stats with artifical zeros
        2. Placing the gate after the SE, before the project makes it difficult to skip depthwise channels because you need the channel feature to determine the attention scales, but you also want to sparsify it
        """
        if gate is not None:
            gate = gate.view(x.size(0), -1, 1, 1)
            x = x * gate

        x = self.se(x)
        x = self.project(x)
        
        if self.use_res_connect:
            return identity + self.stochastic_depth(x)
        return x

class TransferDeepMoEEfficientNet(nn.Module):
    def __init__(self, model_id=0, num_classes=1010, latent_dim=128, moe_start_stage=4, reference_flops=None, relu_init_val=0, relu_init_std=0.1):
        super().__init__()
        
        # 1. Load the Pre-trained Torchvision Model
        if model_id == 0: # Note: The B1/B2/B3 models are at a slight disadvantage due to the cropped image, but this is fine; the purpose is to find the optimal MoE architecture wrt the baseline EfficientNet B0
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
                        nn.init.normal_(gate_linear.weight, std=relu_init_std)
                        nn.init.constant_(gate_linear.bias, relu_init_val)
                        self.gates[str(block_idx)] = nn.Sequential(gate_linear, nn.ReLU())
                    
                    block_idx += 1

        self.reference_flops = reference_flops
        self.static_overhead_flops = 0.0
        self.flops_per_channel = {}

        profiling_stats = profile_deepmoe_flops(self, input_size=(1, 3, 224, 224))
        self.static_overhead_flops = profiling_stats["static"]
        self.flops_per_channel = profiling_stats["per_channel"]

    def forward(self, x):
        track_flops = bool(self.flops_per_channel)
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
                    if track_flops:
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
        
        if track_flops and self.reference_flops is not None:
            flop_retention_pct = total_active_flops / self.reference_flops
        else:
            flop_retention_pct = torch.tensor(1.0, device=x.device)

        if self.training:
            return x, aux_logits, l1_loss, active_pct, flop_retention_pct
        return x, active_pct, flop_retention_pct

def objective(trial, b0_reference_flops):
    trial_start_time = time.time()
    final_score = 0
    
    BATCH_SIZE = 256
    GRAD_ACCUM_STEPS = trial.suggest_int("grad_accum_steps", 4, 16, 2)
    EPOCHS_FINETUNE = trial.suggest_int("epochs_finetune", 0, 5, 1)
    EPOCHS_JOINT = 10 - EPOCHS_FINETUNE # Note: The seperate head finetuning phase in the baseline script converged to epochs head = 0, meaning it was useless, we instead do the two-step finetuning described in the paper.

    model_id = 0 # Note: Due to the large size of the other efficientnet models, they struggle to be more efficient than b0 and their parameter activations go to zero to hit the sparsity req, thus we won't sweep them
    mu = trial.suggest_float("mu", 0, 1)
    relu_init_val = trial.suggest_float("relu_init_val", 0, 1)
    relu_init_std = trial.suggest_float("relu_init_std", 0.001, 1)
    moe_start_stage = trial.suggest_int("moe_start_stage", 1, 5, 1)
    latent_dim = trial.suggest_categorical("latent_dim", [32, 64, 128])
    lambda_g = trial.suggest_float("lambda_g", 1e-5, 1, log=True)

    lr_head_mul = trial.suggest_float("lr_head_mul", 0.0001, 0.1, log=True)
    lr_moe_mul = trial.suggest_float("lr_moe_mul", 0.0001, 0.1, log=True)
    lr_base_mul = trial.suggest_float("lr_base_mul", 0.0001, 0.01, log=True)
    lr_finetune_mul = trial.suggest_float("lr_finetune_mul", 0.0001, 0.01, log=True)

    LR_HEAD = lr_head_mul * math.sqrt(GRAD_ACCUM_STEPS)
    LR_MOE = lr_moe_mul * math.sqrt(GRAD_ACCUM_STEPS)
    LR_BASE = lr_base_mul * math.sqrt(GRAD_ACCUM_STEPS)
    LR_FINETUNE = lr_finetune_mul * math.sqrt(GRAD_ACCUM_STEPS)
    
    WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    
    train_loader, val_loader, num_classes = get_dataloaders(BATCH_SIZE)

    run = wandb.init(
        project="efficientnet_deepmoe",
        group="sweep",
        name=f"lam{lambda_g:.1e}_bodylr{LR_BASE:.1e}",
        config={
            "mu": mu,
            "moe_start_stage": moe_start_stage,
            "lambda_g": lambda_g,
            "moe_LR": LR_MOE,
            "head_LR": LR_HEAD,
            "base_LR": LR_BASE,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "epochs_finetune": EPOCHS_FINETUNE,
            "epochs_joint": EPOCHS_JOINT,
        },
        reinit=True,
    )
    
    model = TransferDeepMoEEfficientNet(
        model_id=model_id, num_classes=num_classes,
        reference_flops=b0_reference_flops,
        moe_start_stage=moe_start_stage, latent_dim=latent_dim,
        relu_init_val=relu_init_val,
        relu_init_std=relu_init_std
    )
    effective_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    joint_steps = int(EPOCHS_JOINT * effective_steps_per_epoch)

    moe_params = list(model.embedding_net.parameters()) + list(model.gates.parameters())
    head_params = []
    base_params = []
    for name, param in model.base_model.named_parameters():
        if param.requires_grad:
            head_params.append(param)
        else:
            base_params.append(param)
        param.requires_grad = True

    shared_optim = torch.optim.AdamW([
        {"params": moe_params, "lr": LR_MOE},
        {"params": head_params, "lr": LR_HEAD},
        {"params": base_params, "lr": LR_BASE},
    ], weight_decay=WEIGHT_DECAY)

    shared_sched = get_cosine_schedule_with_warmup(
        shared_optim, 
        int(joint_steps * 0.1), 
        joint_steps
    )
    final_score, pruned = train_loop_deepmoe(model, shared_optim, shared_sched, EPOCHS_JOINT, GRAD_ACCUM_STEPS, train_loader, val_loader, trial, 0, lambda_g, mu)

    del shared_optim
    if pruned:
        wandb.run.summary.update({"state": "pruned", "final_score": final_score, "total_runtime_seconds": time.time() - trial_start_time})
        wandb.finish()
        del model, train_loader, val_loader
        torch.cuda.empty_cache()
        raise optuna.exceptions.TrialPruned()
    
    if EPOCHS_FINETUNE > 0:
        for param in model.embedding_net.parameters():
            param.requires_grad = False
        for param in model.gates.parameters():
            param.requires_grad = False

        finetune_optim = torch.optim.AdamW([
            {"params": head_params, "lr": LR_FINETUNE},
            {"params": base_params, "lr": LR_FINETUNE},
        ], weight_decay=WEIGHT_DECAY)

        finetune_steps = int(EPOCHS_FINETUNE * effective_steps_per_epoch)
        finetune_sched = get_cosine_schedule_with_warmup(
            finetune_optim, 
            int(finetune_steps * 0.1), 
            finetune_steps
        )

        final_score, pruned = train_loop_deepmoe(model, finetune_optim, finetune_sched, EPOCHS_FINETUNE, GRAD_ACCUM_STEPS, train_loader, val_loader, trial, EPOCHS_JOINT, 0.0, 0.0, freeze_routing=True)
        
        del finetune_optim
        if pruned:
            wandb.run.summary.update({"state": "pruned", "final_score": final_score, "total_runtime_seconds": time.time() - trial_start_time})
            wandb.finish()
            del model, train_loader, val_loader
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        
    wandb.run.summary.update({
        "state": "completed", 
        "final_score": final_score, 
        "total_runtime_seconds": time.time() - trial_start_time
    })
    wandb.finish()
    
    del model, train_loader, val_loader
    torch.cuda.empty_cache()
    return final_score

if __name__ == "__main__":
    SHARED_DIR = "./optuna_slurm_logs"
    os.makedirs(SHARED_DIR, exist_ok=True)
    
    print("Loading baseline EfficientNet-B0...")
    baseline_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # 2. Run the profiler
    print("Profiling MACs...")
    b0_stats = profile_deepmoe_flops(baseline_b0, input_size=(1, 3, 224, 224))

    # 3. Extract the reference FLOPs
    B0_REFERENCE_FLOPS = b0_stats['total']

    print("-" * 40)
    print(f"B0 Total FLOPs (MACs): {B0_REFERENCE_FLOPS:,.0f}")
    print(f"B0 Static Overhead:    {b0_stats['static']:,.0f}")
    print("-" * 40)
    del baseline_b0
    
    journal_file = os.path.join(SHARED_DIR, "journal_deepmoe.log")

    storage = JournalStorage(JournalFileStorage(journal_file))

    sampler = optuna.samplers.TPESampler(multivariate=True)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10, n_warmup_steps=4, interval_steps=2 # Will be done in parallel on many gpus
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