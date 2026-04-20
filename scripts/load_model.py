import torch
import torchvision.transforms as transforms
from PIL import Image

from train_deepmoe import TransferDeepMoEEfficientNet 

repo_id = "G-reen/effnet_b0_iNat2019_deepmoe_lambda7e-05"

print(f"Loading model from {repo_id}...")
model = TransferDeepMoEEfficientNet.from_pretrained(repo_id)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dummy_image = Image.new("RGB", (224, 224)) 
input_tensor = transform(dummy_image).unsqueeze(0).to(device)

with torch.no_grad():
    logits, active_pct, flop_pct = model(input_tensor)
    
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

print(f"Predicted Class ID: {predicted_class}")
print(f"Expert Activation:  {active_pct.item() * 100:.2f}%")
print(f"FLOP Retention:     {flop_pct.item() * 100:.2f}%")