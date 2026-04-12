import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

from pathlib import Path

import torch
from torchvision import datasets, transforms

model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
