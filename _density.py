import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors

def extract_features(model, dataset, device="cuda", batch_size=256, layer_name="avgpool"):
    """
    Extracts features from the dataset.
    automatically registers a hook to capture output from the specified layer name (e.g., 'avgpool').
    This ensures we get Features (512-dim), not Logits.
    """
    model.eval()
    model.to(device)

    # 1. Setup Hook to capture features
    features_list = []
    
    def hook_fn(module, input, output):
        # Flatten the output (N, 512, 1, 1) -> (N, 512)
        features_list.append(output.flatten(1).cpu())

    # Find the layer to hook
    hook_handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            hook_handle = module.register_forward_hook(hook_fn)
            print(f"[Density] Hook registered on layer: {name}")
            break
            
    if hook_handle is None:
        # Fallback: assume model is just a feature extractor or try the penultimate layer
        print(f"[Density] Warning: Layer '{layer_name}' not found. Using model output directly.")

    # 2. DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print("[Density] Extracting features...")
    
    # 3. Inference Loop
    dummy_features_list = [] # Used only if hook fails
    
    with torch.no_grad():
        # --- FIX STARTS HERE ---
        # Old: for images, _ in loader:  <-- CRASHES with 3 items
        # New: Robust way to grab just the image
        for batch in loader:
            images = batch[0] 
            
            images = images.to(device)
            output = model(images)
            if hook_handle is None:
                dummy_features_list.append(output.cpu())
        # --- FIX ENDS HERE ---

    # 4. Clean up
    if hook_handle:
        hook_handle.remove()
        features = torch.cat(features_list, dim=0).numpy()
    else:
        features = torch.cat(dummy_features_list, dim=0).numpy()

    print(f"[Density] Extracted features shape: {features.shape}")
    return features


def compute_density_probability(features: np.ndarray, k: int = 50) -> np.ndarray:
    """
    Calculates Density probability based on k-Nearest Neighbors distance.
    Formula: Higher Distance = Rare Sample = High Probability (to keep)
    """
    print("[Density] Computing neighbor distances...")
    
    # Safety: Ensure k isn't larger than dataset
    n_samples = features.shape[0]
    k = min(k, n_samples - 1)
    
    if k < 1:
        return np.ones(n_samples)

    # Step 1: kNN
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn_model.fit(features)

    # Step 2: Get distances
    dists, _ = nn_model.kneighbors(features)      
    neighbor_dists = dists[:, 1:] # remove self-distance (index 0)

    # Step 3: Mean distance (Rho)
    rho = neighbor_dists.mean(axis=1)       
    
    # Step 4: Normalize to [0, 1]
    # High distance (outlier/rare) -> High score
    p_rho = (rho - rho.min()) / (rho.max() - rho.min() + 1e-12)

    print(f"[Density] p_rho range: {p_rho.min():.3f} -> {p_rho.max():.3f}")
    return p_rho
