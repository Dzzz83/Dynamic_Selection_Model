import numpy as np
import torch
from torch.utils.data import DataLoader

def get_targets_safe(dataset):
    """
    Safely retrieves targets/labels from a dataset instance.
    Handles tuple returns of size 2 or 3.
    """
    # 1. Fast Path: Attributes
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)

    # 2. Slow Path: Iteration
    print("[Selection] Warning: Iterating dataset to find targets (Slow)...")
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=4)
    all_targets = []
    
    for batch in loader:
        # === UPDATE: Handle (Image, Label, Index) ===
        # batch[1] is the Label
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            targets = batch[1]
        else:
             raise ValueError(f"Unexpected batch format: {type(batch)}")
             
        all_targets.extend(targets.tolist())

    return np.array(all_targets)

def select_samples(dataset, p_rho, p_con, selection_ratio=0.5):
    """
    Combines probabilities and selects the top samples per class.
    """
    # 1. Formula 3: Joint Distribution
    if len(p_rho) != len(p_con):
        # Allow broadcasting if one is scalar, otherwise error
        pass 

    p_rho = np.array(p_rho)
    p_con = np.array(p_con)
    p_sel = p_rho * p_con
    
    # 2. Safe Target Extraction
    targets = get_targets_safe(dataset)
    
    # 3. Per-Class Selection
    classes = np.unique(targets)
    selected_indices = []

    for c in classes:
        class_idxs = np.where(targets == c)[0]
        class_scores = p_sel[class_idxs]
        
        # Calculate k (number of samples to keep)
        k = max(1, int(len(class_idxs) * selection_ratio))
        
        # Sort descending (Best scores first)
        local_sorted_args = np.argsort(-class_scores) 
        keep_local = local_sorted_args[:k]
        
        # Map back to global indices
        keep_global = class_idxs[keep_local]
        selected_indices.extend(keep_global.tolist())

    print(f"[Selection] Selected {len(selected_indices)} / {len(dataset)} samples.")
    return selected_indices