import numpy as np
import torch
from torch.utils.data import Subset

def get_targets_safe(dataset):
    """
    Safely retrieves targets/labels from a dataset instance.
    Handles: torchvision datasets, ImageFolder, Tensors, and Lists.
    """
    # 1. Try standard attribute names
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        # Fallback: Warning and slow iteration (only if absolutely necessary)
        print("[Selection] Warning: Dataset has no .targets or .labels. Iterating to find classes (Slow)...")
        targets = [y for _, y in dataset]

    # 2. Convert to Numpy
    if torch.is_tensor(targets):
        return targets.cpu().numpy()
    elif isinstance(targets, list):
        return np.array(targets)
    elif isinstance(targets, np.ndarray):
        return targets
    else:
        raise ValueError(f"[Selection] Could not process targets of type {type(targets)}")

def select_samples(dataset, p_rho, p_con, selection_ratio=0.5):
    """
    Combines probabilities and selects the top samples per class.
    
    Args:
        dataset: The dataset object
        p_rho: Density probability (from density.py)
        p_con: Consistency probability (from consistency.py)
        selection_ratio: Percentage of data to keep (0.0 to 1.0)
        
    Returns:
        list[int]: The global indices of the selected samples.
    """
    
    # 1. Formula 3: Joint Distribution
    if len(p_rho) != len(p_con):
        raise ValueError(f"Shape mismatch! p_rho: {len(p_rho)}, p_con: {len(p_con)}")

    # Ensure inputs are numpy arrays
    p_rho = np.array(p_rho)
    p_con = np.array(p_con)

    p_sel = p_rho * p_con
    print(f"[Selection] Joint Probability calculated. Range: {p_sel.min():.4f} -> {p_sel.max():.4f}")

    # 2. Safe Target Extraction (Generic Fix)
    targets = get_targets_safe(dataset)
    
    # Ensure targets match probability length (Sanity Check)
    if len(targets) != len(p_sel):
        # This handles cases where dataset might have been subsetted previously
        print(f"[Selection] Warning: Dataset size ({len(dataset)}) != Scores size ({len(p_sel)}).")
        # Depending on your pipeline, you might want to raise an error here
        # raise ValueError("Dataset and Score size mismatch")

    # 3. Per-Class Selection
    classes = np.unique(targets)
    print(f"[Selection] Selecting top {selection_ratio*100:.1f}% samples for {len(classes)} classes...")
    
    selected_indices = []

    for c in classes:
        # Find indices for this class
        class_idxs = np.where(targets == c)[0]
        
        # Get scores for this class
        class_scores = p_sel[class_idxs]
        
        # Calculate how many to keep
        k = max(1, int(len(class_idxs) * selection_ratio))
        
        # Sort descending (Best scores first)
        local_sorted_args = np.argsort(-class_scores) 
        
        # Take top k
        keep_local = local_sorted_args[:k]
        
        # Convert back to global dataset indices
        keep_global = class_idxs[keep_local]
        
        selected_indices.extend(keep_global.tolist())

    print(f"[Selection] Final subset: {len(selected_indices)} / {len(dataset)} samples "
          f"({len(selected_indices)/len(dataset):.2%})")
    
    return selected_indices