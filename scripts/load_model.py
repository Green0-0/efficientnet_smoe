import torch
import torchvision.transforms as transforms
from PIL import Image

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F

from train_deepmoe import TransferDeepMoEEfficientNet 
from training_utils import get_dataloaders
import os
from sklearn.decomposition import PCA

def extract_routing(model, dataloader, device, max_batches=None):
    model.eval()
    
    all_routes = []
    all_super = []

    correct = 0
    total = 0
    active_pct_sum = float(0)
    flop_pct_sum = float(0)
    num_batches = 0

    with torch.no_grad():
        for i, (x, y, super_y) in enumerate(dataloader): 
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            
            logits, active_pct, flop_pct, gates = model(x, return_gates=True)  # forward once to populate gates
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)

            correct += (predicted_classes == y).sum().item()
            total += y.size(0)
            active_pct_sum += active_pct.item()
            flop_pct_sum += flop_pct.item()
            num_batches += 1

            route = torch.cat(gates, dim=1)  # [B, total_channels]
            
            all_routes.append(route.cpu())
            all_super.append(super_y.view(-1).cpu())

    top1_acc = 100.0 * correct / total
    avg_active_pct = active_pct_sum / num_batches * 100
    avg_flop_pct = flop_pct_sum / num_batches * 100
    return (
        torch.cat(all_routes),
        torch.cat(all_super),
        top1_acc,
        avg_active_pct,
        avg_flop_pct
    )

def tSNE_visualization(routes, super_labels, id_to_super_name, save_path, perplexity=30):
    plt.figure()
    tsne = TSNE(n_components=2, perplexity=perplexity)
    # Could add PCA if noisy
    routes_np = routes.cpu().numpy()
    routes_np = PCA(n_components=50).fit_transform(routes_np)
    emb = tsne.fit_transform(routes_np)

    # plt.scatter(emb[:,0], emb[:,1], c=super_labels.cpu().numpy(), cmap='tab10', s=5)
    unique_labels = torch.unique(super_labels).cpu().numpy()

    for u in unique_labels:
        idx = super_labels == u
        label_name = id_to_super_name.get(int(u), f"Class {u}")
        plt.scatter(
            emb[idx, 0],
            emb[idx, 1],
            s=5,
            label=label_name
        )
    plt.title("Routing Patterns (t-SNE)")
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# High cosine similarity between super labels on multiple runs
def cosine_sim_visualization(mean_routes, save_path):
    plt.figure()
    mean_routes = F.normalize(mean_routes, dim=1)
    sim = F.cosine_similarity(
        mean_routes.unsqueeze(1),
        mean_routes.unsqueeze(0),
        dim=-1
    )
    plt.imshow(sim.numpy(), cmap='viridis')
    plt.colorbar()
    plt.title("Routing Similarity Between Super-Categories")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# The following is not entirely correctly implemented and should be discarded
# Essentially, would need slices of layers of model to implement crrectly
def heatmap_visualization(mean_routes,save_path):
    plt.figure()
    plt.imshow(mean_routes.numpy(), aspect='auto')
    plt.colorbar()

    plt.ylabel("Super-category")
    plt.xlabel("Channels (concatenated layers)")
    plt.title("DeepMoE Routing Heatmap")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    repo_id = "G-reen/effnet_b0_iNat2019_deepmoe_lambda7e-05"

    print(f"Loading model from {repo_id}...")
    model = TransferDeepMoEEfficientNet.from_pretrained(repo_id)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    BATCH_SIZE = 256

    print("Loading Test Dataset")
    _, _, test_loader, _, id_to_super_name = get_dataloaders(BATCH_SIZE)

    print("Extracting Routes")
    # routes, super_labels, top1_acc, avg_active_pct, avg_flop_pct = extract_routing(model, test_loader, device, max_batches=3)
    routes, super_labels, top1_acc, avg_active_pct, avg_flop_pct = extract_routing(model, test_loader, device)
    
    print(f"Top 1 Accuracy: {top1_acc:.4f}%")
    print(f"Expert Activation: {avg_active_pct:.4f}%")
    print(f"FLOP activation: {avg_flop_pct:.4f}%")

    OUTPUT_DIR = "visualizations"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Attempting tSNE Visualization between Super Labels")
    tSNE_visualization(routes, super_labels, id_to_super_name, save_path=os.path.join(OUTPUT_DIR, "tsne_routes_test.png"), perplexity=30)

    # unique = torch.unique(super_labels)

    # mean_routes = []

    # for u in unique:
    #     mask = (super_labels == u)
    #     mean_route = routes[mask].mean(dim=0)
    #     mean_routes.append(mean_route)

    # mean_routes = torch.stack(mean_routes)  # [num_super, total_channels]
    # # Mean routes are per super label

    # print("Attempting Cosine Similarity Comparison between Super Labels")
    # cosine_sim_visualization(mean_routes, save_path=os.path.join(OUTPUT_DIR, "cos_sim_test.png"))

    # print("Attempting Heatmap Visualization between Super Labels")
    # heatmap_visualization(mean_routes, save_path=os.path.join(OUTPUT_DIR, "heatmaps_test.png"))
    
# Original testing:
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# dummy_image = Image.new("RGB", (224, 224)) 
# input_tensor = transform(dummy_image).unsqueeze(0).to(device)

# with torch.no_grad():
#     logits, active_pct, flop_pct = model(input_tensor)
    
#     probabilities = torch.nn.functional.softmax(logits, dim=-1)
#     predicted_class = torch.argmax(probabilities, dim=-1).item()

# print(f"Predicted Class ID: {predicted_class}")
# print(f"Expert Activation:  {active_pct.item() * 100:.2f}%")
# print(f"FLOP Retention:     {flop_pct.item() * 100:.2f}%")