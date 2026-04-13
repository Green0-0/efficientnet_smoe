# Written by Gemini

import os
import wandb
import optuna

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import math
import torch
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        progress = min(1.0, progress)
        
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)

def get_dataloaders(batch_size):
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_dir = "/usr/project/xtmp/inaturalistdata_store"

    # Note: Torchvision does not seem to supply a 2019 val/train split, so we are manually creating one
    # Also, we use a tiny train dataset to speed up training on a smaller subset of the data
    full_train_dataset = datasets.INaturalist(
        root=data_dir,
        version="2019",
        transform=train_transforms,
        download=False,
    )
    full_val_dataset = datasets.INaturalist(
        root=data_dir,
        version="2019",
        transform=val_transforms,
        download=False,
    )

    num_classes = len(full_train_dataset.all_categories)

    # Do stratified split for train (45%), val (5%), test (50%), making sure every class gets the same proportion so no class disappears
    seed = 42
    targets = [cat_id for cat_id, fname in full_train_dataset.index]
    all_indices = list(range(len(full_train_dataset)))

    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.50,
        stratify=targets,
        random_state=seed
    )
    train_val_targets = [targets[i] for i in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.10,
        stratify=train_val_targets,
        random_state=seed
    )

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)
    # test_dataset = Subset(full_val_dataset, test_indices)

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    num_workers = max(1, slurm_cpus - 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(2, num_workers),
        pin_memory=True,
    )
    
    return train_loader, val_loader, num_classes

def train_loop(model, optimizer, scheduler, epochs, grad_accum_steps, train_loader, val_loader, optuna_trial, optuna_trial_start_epoch):
    device = "cuda" # Note: An a5000 gpu or higher is expected, or the script will not work!
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    global_step = 0

    try:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss = loss / grad_accum_steps

                loss.backward()
                
                if (i+1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                if global_step % 50 == 0:
                    wandb.log(
                        {
                            "train/step_loss": loss.item() * grad_accum_steps,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                        }
                    )
                global_step += 1
            
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            final_val_acc = 100.0 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)

            wandb.log(
                {
                    "val/accuracy": final_val_acc,
                    "val/loss": avg_val_loss,
                    "epoch": epoch + optuna_trial_start_epoch,
                }
            )
            
            if optuna_trial != None:
                optuna_trial.report(final_val_acc, epoch + optuna_trial_start_epoch)
                if optuna_trial.should_prune():
                    return final_val_acc, True
    except Exception as e:
        wandb.run.summary["state"] = "crashed"
        raise e
    return final_val_acc, False