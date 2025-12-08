import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os

# ✅ IMPORT YOUR MODULES
from _imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from _density import compute_density_probability
from _consistency import ConsistencyCalculator
from selection import select_samples

# ==========================================
# 1. Architecture Wrapper
# ==========================================
class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_CIFAR, self).__init__()
        self.net = models.resnet18(weights=None)
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity() 
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x, return_features=False):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = self.net.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.net.fc(features)
        if return_features: return features, logits
        return logits

def extract_all_features(model, dataset, device):
    """Extracts features from the ENTIRE dataset."""
    print(f"   -> Extracting features from all {len(dataset)} images...")
    model.eval()
    # Shuffle FALSE to keep alignment with targets
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    all_feats = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            feats, _ = model(inputs, return_features=True)
            all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0).numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.7, help="Selection Ratio")
    parser.add_argument("--dataset", type=str, default="imbalance_cifar10", help="imbalance_cifar10 or imbalance_cifar100")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"✅ Generating Paper-Style Plot for {args.dataset} (Ratio: {args.ratio})")

    # 1. Setup Data
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    
    if args.dataset == "imbalance_cifar10":
        dataset = IMBALANCECIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
        num_classes = 10
    else:
        dataset = IMBALANCECIFAR100(root=args.data_dir, train=True, download=True, transform=transform)
        num_classes = 100

    model = ResNet18_CIFAR(num_classes=num_classes).to(device)
    
    # 2. Extract Features
    features = extract_all_features(model, dataset, device)
    all_targets = np.array(dataset.targets)

    # 3. Calculate Selection Logic
    print("   -> Calculating Selection Indices...")
    p_rho = compute_density_probability(features)
    calc = ConsistencyCalculator(device=device)
    p_con = calc.calculate(dataset)
    
    # Dynamic Indices
    ours_indices_list = select_samples(dataset, p_rho, p_con, args.ratio)
    mask_ours = np.zeros(len(dataset), dtype=bool)
    mask_ours[ours_indices_list] = True

    # Random Indices
    total_len = len(dataset)
    keep_len = int(total_len * args.ratio)
    rand_idxs = torch.randperm(total_len)[:keep_len].tolist()
    mask_random = np.zeros(len(dataset), dtype=bool)
    mask_random[rand_idxs] = True

    # 4. Subsampling for Visualization (CRITICAL FIX FOR RED DOTS)
    # We sample randomly from the WHOLE dataset to ensure we get both Head (start) and Tail (end)
    print("   -> Subsampling 4,000 points for clean plotting...")
    limit = 4000
    if len(dataset) < limit: limit = len(dataset)
        
    vis_indices = np.random.choice(len(dataset), limit, replace=False)
    
    vis_features = features[vis_indices]
    vis_targets = all_targets[vis_indices]
    vis_mask_ours = mask_ours[vis_indices]
    vis_mask_random = mask_random[vis_indices]

    # 5. Run t-SNE
    print("   -> Running t-SNE...")
    tsne = TSNE(
            n_components=2, 
            perplexity=50,       # <--- increased from default 30
            n_iter=1000,         # <--- ensure convergence
            init='pca', 
            learning_rate='auto',
            random_state=42
        )    
    X_2d = tsne.fit_transform(vis_features)

    # 6. Plotting (Paper Style)
    print("   -> Generating Plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Define Head vs Tail
    # Tail = Bottom 50% of classes
    is_tail = vis_targets >= (num_classes // 2)

    def plot_subplot(ax, selection_mask, title):
        # 1. Background: DISCARDED (Gray Fade)
        # We plot ALL points first as gray, then overlay the selected ones.
        # This ensures "holes" are visible.
        ax.scatter(X_2d[~selection_mask, 0], X_2d[~selection_mask, 1], 
                   c='#e0e0e0', s=15, alpha=0.3, label='Discarded') 

        # 2. Foreground: SELECTED HEAD (Blue Dots)
        mask_head = selection_mask & (~is_tail)
        ax.scatter(X_2d[mask_head, 0], X_2d[mask_head, 1], 
                   c='#4169E1', s=10, alpha=0.6, label='Selected (Head)') # Royal Blue

        # 3. Foreground: SELECTED TAIL (Red Crosses)
        # Plot these LAST so they sit on top of everything
        mask_tail = selection_mask & is_tail
        ax.scatter(X_2d[mask_tail, 0], X_2d[mask_tail, 1], 
                   c='#DC143C', marker='x', s=40, linewidth=1.5, alpha=0.9, label='Selected (Tail)') # Crimson

        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Custom Legend
        if mask_tail.sum() > 0:
            ax.legend(loc='upper right', frameon=True, fontsize=10)
        
        ax.grid(False) # Clean look
        ax.set_xticks([])
        ax.set_yticks([])
        # Add border box
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

    plot_subplot(ax1, vis_mask_random, f"(a) Random Selection (r={args.ratio})")
    plot_subplot(ax2, vis_mask_ours, f"(b) Dynamic Selection (r={args.ratio})")

    plt.tight_layout()
    save_path = f"paper_plot_{args.dataset}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Done! Saved to '{save_path}'")

if __name__ == "__main__":
    main()