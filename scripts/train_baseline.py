# AI minimally assisted with some syntax and bug checking
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
from torchvision.models import EfficientNet_B0_Weights

from training_utils import get_cosine_schedule_with_warmup, get_dataloaders, train_loop

class HubWrapper(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)

def train(model_hf_id, GRAD_ACCUM_STEPS, EPOCHS_HEAD, lr_head_mul, lr_head2_mul, lr_body_mul, WEIGHT_DECAY):
    trial_start_time = time.time()
    final_val_acc = 0
    
    BATCH_SIZE = 256
    EPOCHS_BODY = 10 - EPOCHS_HEAD

    LR_HEAD = lr_head_mul * math.sqrt(GRAD_ACCUM_STEPS) 
    LR_HEAD2 = lr_head2_mul * math.sqrt(GRAD_ACCUM_STEPS) 
    LR_BODY = lr_body_mul * math.sqrt(GRAD_ACCUM_STEPS)  
    
    
    train_loader, _, test_loader, num_classes, _ = get_dataloaders(BATCH_SIZE)

    run = wandb.init(
        project="efficientnet_b0_TL_run",
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
        final_val_acc, _ = train_loop(model, head_optimizer, scheduler, EPOCHS_HEAD, GRAD_ACCUM_STEPS, train_loader, test_loader, None, 0)
    
    for param in model.parameters():
        param.requires_grad = True

    body_optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": LR_HEAD2},
        {"params": body_params, "lr": LR_BODY},
    ], weight_decay=WEIGHT_DECAY)

    effective_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_body_steps = int(EPOCHS_BODY * effective_steps_per_epoch)
    body_scheduler = get_cosine_schedule_with_warmup(body_optimizer, total_body_steps * 0.1, total_body_steps)
    final_val_acc, _ = train_loop(model, body_optimizer, body_scheduler, EPOCHS_BODY, GRAD_ACCUM_STEPS, train_loader, test_loader, None, EPOCHS_HEAD)
    total_runtime = time.time() - trial_start_time
    
    if model_hf_id is not None and isinstance(model_hf_id, str):
        print(f"Pushing model to Hugging Face Hub: {model_hf_id}")
        hub_model = HubWrapper(model)
        hub_model.push_to_hub(model_hf_id)
        
        # --- NEW: Generate and Push the Model Card (README.md) ---
        print("Pushing Model Card with training stats...")
        
        # 1. Define standard metadata for the top of the README
        card_data = ModelCardData(
            language="en",
            tags=["image-classification", "pytorch", "efficientnet"],
            datasets=["inaturalist2019"], # Add your specific dataset tag
            metrics=["accuracy"]
        )
        
        # 2. Build the markdown content using f-strings to inject your variables
        readme_text = f"""
---
{card_data.to_yaml()}
---

# EfficientNet-B0 fine-tuned on iNaturalist 2019

This model is a fine-tuned EfficientNet-B0 on the iNaturalist 2019 dataset, trained using PyTorch.

## Training Results
- **Final Validation Accuracy**: {final_val_acc:.4f}
- **Total Runtime**: {total_runtime:.2f} seconds

## Hyperparameters
- **Batch Size**: {BATCH_SIZE}
- **Gradient Accumulation Steps**: {GRAD_ACCUM_STEPS}
- **Total Epochs**: {EPOCHS_HEAD + EPOCHS_BODY}
  - Head-only Epochs: {EPOCHS_HEAD}
  - Full-body Epochs: {EPOCHS_BODY}
- **Weight Decay**: {WEIGHT_DECAY}
- **Learning Rates**:
  - Head Phase: {LR_HEAD}
  - Body Phase (Head Params): {LR_HEAD2}
  - Body Phase (Body Params): {LR_BODY}

*Training was tracked using [Weights & Biases](https://wandb.ai).*
"""
        
        # 3. Instantiate the card and push it to the same repo
        card = ModelCard(readme_text)
        card.push_to_hub(model_hf_id)
        
    wandb.run.summary["total_runtime_seconds"] = total_runtime
    wandb.run.summary["final_val_acc"] = final_val_acc
    wandb.run.summary["state"] = "completed"
    wandb.finish()
    del model, body_optimizer, train_loader, val_loader
    if EPOCHS_HEAD != 0:
        del head_optimizer
    torch.cuda.empty_cache()
    return final_val_acc

if __name__ == "__main__":
    train("G-reen/effnet_b0_iNat2019_baseline", 4, 0, 0, 0.005, 0.001, 0.01)