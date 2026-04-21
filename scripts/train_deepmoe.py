# AI added the deepmoe portion, also transplanted some code from the old/preliminary tests that worked well (ref: ai_slop/deepmoe_sweep.py)
from huggingface_hub import PyTorchModelHubMixin, ModelCard, ModelCardData

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

from flop_profiler import profile_deepmoe_flops

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

class TransferDeepMoEEfficientNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, effnet_version=0, num_classes=1010, latent_dim=128, moe_start_stage=4, reference_flops=None, relu_init_val=0, relu_init_std=0.1):
        super().__init__()
        
        # 1. Load the Pre-trained Torchvision Model
        if effnet_version == 0: # Note: The B1/B2/B3 models are at a slight disadvantage due to the cropped image, but this is fine; the purpose is to find the optimal MoE architecture wrt the baseline EfficientNet B0
            self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        elif effnet_version == 1:
            self.base_model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        elif effnet_version == 2:
            self.base_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        elif effnet_version == 3:
            self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model_type: {effnet_version}")
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

    def forward(self, x, return_gates=False):
        track_flops = bool(self.flops_per_channel)
        e, aux_logits = self.embedding_net(x)

        l1_loss = 0.0
        active_experts = 0.0
        total_experts = 0.0
        active_body_flops = 0.0
        
        if return_gates:
            collected_gates = []

        # We must manually iterate through the base_model features to pass the gate tensor
        x = self.base_model.features[0](x) # Stem

        block_idx = 0
        for stage_idx in range(1, 8):
            stage = self.base_model.features[stage_idx]
            for module in stage:
                idx_str = str(block_idx)
                if idx_str in self.gates:
                    gate = self.gates[idx_str](e)

                    if return_gates:
                        collected_gates.append(gate.detach().cpu())

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
                    if track_flops:
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
        if return_gates:
            return x, aux_logits, collected_gates
        if self.training:
            return x, aux_logits, l1_loss, active_pct, flop_retention_pct
        return x, active_pct, flop_retention_pct

def train(model_hf_id, b0_reference_flops, GRAD_ACCUM_STEPS, EPOCHS_FINETUNE, mu, relu_init_val, relu_init_std, moe_start_stage, latent_dim, lambda_g, lr_head_mul, lr_moe_mul, lr_base_mul, lr_finetune_mul, WEIGHT_DECAY):
    trial_start_time = time.time()
    final_score = 0
    
    BATCH_SIZE = 256
    EPOCHS_JOINT = 10 - EPOCHS_FINETUNE # Note: The seperate head finetuning phase in the baseline script converged to epochs head = 0, meaning it was useless, we instead do the two-step finetuning described in the paper.

    effnet_version = 0 # Note: Due to the large size of the other efficientnet models, they struggle to be more efficient than b0 and their parameter activations go to zero to hit the sparsity req, thus we won't sweep them

    LR_HEAD = lr_head_mul * math.sqrt(GRAD_ACCUM_STEPS)
    LR_MOE = lr_moe_mul * math.sqrt(GRAD_ACCUM_STEPS)
    LR_BASE = lr_base_mul * math.sqrt(GRAD_ACCUM_STEPS)
    LR_FINETUNE = lr_finetune_mul * math.sqrt(GRAD_ACCUM_STEPS)
        
    train_loader, _, test_loader, num_classes = get_dataloaders(BATCH_SIZE)

    run = wandb.init(
        project="efficientnet_deepmoe_run",
        group="sweep",
        name=f"lam{lambda_g:.1e}_bodylr{LR_BASE:.1e}",
        config={
            "mu": mu,
            "relu_init_val": relu_init_val,
            "relu_init_std": relu_init_std,
            "moe_start_stage": moe_start_stage,
            "latent_dim": latent_dim,
            "lambda_g": lambda_g,
            "moe_LR": LR_MOE,
            "head_LR": LR_HEAD,
            "base_LR": LR_BASE,
            "finetune_LR": LR_FINETUNE,
            "weight_decay":WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "epochs_finetune": EPOCHS_FINETUNE,
            "epochs_joint": EPOCHS_JOINT,
        },
        reinit=True,
    )
    
    model = TransferDeepMoEEfficientNet(
        effnet_version=effnet_version, num_classes=num_classes,
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
    final_score, _, final_active_pct, final_flop_pct = train_loop_deepmoe(model, shared_optim, shared_sched, EPOCHS_JOINT, GRAD_ACCUM_STEPS, train_loader, test_loader, None, 0, lambda_g, mu)

    del shared_optim
    
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

        final_score, _, final_active_pct, final_flop_pct = train_loop_deepmoe(model, finetune_optim, finetune_sched, EPOCHS_FINETUNE, GRAD_ACCUM_STEPS, train_loader, test_loader, None, EPOCHS_JOINT, 0.0, 0.0, freeze_routing=True)
        
        del finetune_optim
        
    # Calculate runtime early for the README
    total_runtime = time.time() - trial_start_time

    if model_hf_id is not None and isinstance(model_hf_id, str):
        print(f"Pushing model to Hugging Face Hub: {model_hf_id}")
        model.push_to_hub(model_hf_id)
        
        # --- Generate and Push the Model Card (README.md) ---
        print("Pushing Model Card with DeepMoE training stats...")
        
        card_data = ModelCardData(
            language="en",
            tags=["image-classification", "pytorch", "efficientnet", "mixture-of-experts", "deepmoe"],
            datasets=["inaturalist2019"],
        )
        
        readme_text = f"""
---
{card_data.to_yaml()}
---

# DeepMoE EfficientNet-B0 fine-tuned on iNaturalist 2019

This model is a Mixture-of-Experts (DeepMoE) variant of EfficientNet-B0, fine-tuned on the iNaturalist 2019 dataset to optimize both accuracy and computational efficiency (FLOP reduction).

## Training Results
- **Final Score (Acc/FLOPs composite)**: {final_score:.4f}
- **Expert Activation Ratio**: {final_active_pct * 100:.1f}%
- **FLOPs Usage**: {final_flop_pct * 100:.1f}% *(compared to baseline B0)*
- **Baseline B0 Reference FLOPs**: {b0_reference_flops:,.0f}
- **Total Runtime**: {total_runtime:.2f} seconds

## Hyperparameters
- **Batch Size**: {BATCH_SIZE}
- **Gradient Accumulation Steps**: {GRAD_ACCUM_STEPS}
- **Weight Decay**: {WEIGHT_DECAY}

### Epochs
- **Total Epochs**: {EPOCHS_JOINT + EPOCHS_FINETUNE}
  - Joint Training Epochs: {EPOCHS_JOINT}
  - Routing-Frozen Finetuning Epochs: {EPOCHS_FINETUNE}

### DeepMoE Architecture & Routing
- **MoE Start Stage**: {moe_start_stage}
- **Latent Dimension**: {latent_dim}
- **Sparsity Penalty ($\\lambda_g$)**: {lambda_g}
- **Target Sparsity ($\\mu$)**: {mu}
- **ReLU Init (Val / Std)**: {relu_init_val} / {relu_init_std}

### Learning Rates
- **MoE Routing Parameters**: {LR_MOE:.2e}
- **Classification Head**: {LR_HEAD:.2e}
- **Base Model (Body)**: {LR_BASE:.2e}
- **Finetune Phase (Frozen Routing)**: {LR_FINETUNE:.2e}

*Training was tracked using [Weights & Biases](https://wandb.ai).*
"""
        card = ModelCard(readme_text)
        card.push_to_hub(model_hf_id)
        
    wandb.run.summary.update({
        "state": "completed", 
        "final_score": final_score, 
        "final_active_pct": final_active_pct,
        "final_flop_pct": final_flop_pct,
        "total_runtime_seconds": total_runtime
    })
    wandb.finish()
    
    # Cleanup 
    del model, train_loader, test_loader
    torch.cuda.empty_cache()
    
    return final_score, final_active_pct, final_flop_pct

if __name__ == "__main__":
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
    
    lambda_g_and_moe_lr = [0.0015, 0.004]
    lambda_g_and_moe_lr = [0.0007, 0.005]
    lambda_g_and_moe_lr = [0.0003, 0.02]
    lambda_g_and_moe_lr = [0.00015, 0.05]
    lambda_g_and_moe_lr = [0.00007, 0.06]
    train(f"G-reen/effnet_b0_iNat2019_deepmoe_lambda{lambda_g_and_moe_lr[0]}", B0_REFERENCE_FLOPS, 4, 0, 0.5, 1, 1, 1, 32, lambda_g_and_moe_lr[0], 0.01, lambda_g_and_moe_lr[1], 0.001, 0, 0.005)