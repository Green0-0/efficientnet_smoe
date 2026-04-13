# AI minimally assisted with some syntax and bug checking

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
    journal_file = os.path.join(SHARED_DIR, "optuna_journal.log")

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

    N_TRIALS_PER_NODE = 7
    print(f"Node startup: attempting {N_TRIALS_PER_NODE} trials...")

    study.optimize(objective, n_trials=N_TRIALS_PER_NODE)