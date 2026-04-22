import torch
import torchvision.transforms as transforms
from huggingface_hub import PyTorchModelHubMixin

import torchvision.models as models
import torch.nn as nn

from training_utils import get_dataloaders

class TransferBaseline(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes: int):
        super().__init__()
        # Takes in baseline EfficinetB0 and replaces head
        # Allows for loading pretrained hugging face model
        backbone = models.efficientnet_b0()
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.classifier[1] = nn.Linear(
            backbone.classifier[1].in_features, num_classes
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


def evaluate(model, dataloader, device, max_batches=None):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for i, (x, y, super_y) in enumerate(dataloader): 
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            predicted_class = torch.argmax(logits, dim=1) # (batch_size,)

            correct += (predicted_class == y).sum().item()
            total += y.size(0)

    top1_acc = 100.0 * correct / total
    return top1_acc

if __name__ == "__main__":
    BATCH_SIZE = 256

    print("Loading Test Dataset")
    _, _, test_loader, num_classes, _ = get_dataloaders(BATCH_SIZE)

    repo_id = "G-reen/effnet_b0_iNat2019_baseline"
    print(f"Loading model from {repo_id}...")

    model = TransferBaseline.from_pretrained(repo_id, num_classes=num_classes)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    print("Evaluating Model")
    top1_acc = evaluate(model, test_loader, device)
    
    print(f"Top 1 Accuracy: {top1_acc:.4f}%")
