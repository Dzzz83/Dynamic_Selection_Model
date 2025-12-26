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
    Handles both static float ratios and class-wise dictionary ratios.
    """
    # 1. Formula 3: Joint Distribution
    if len(p_rho) != len(p_con):
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
        
        # --- FIX: Handle Dictionary vs Float Ratio ---
        if isinstance(selection_ratio, dict):
            # Look up the specific ratio for this class 'c'
            # Fallback to 0.5 if class key is missing for safety
            r = selection_ratio.get(c, 0.5)
        else:
            # Use the static float value
            r = selection_ratio
        # ---------------------------------------------
        
        # Calculate k (number of samples to keep)
        k = max(1, int(len(class_idxs) * r))
        
        # Sort descending (Best scores first)
        local_sorted_args = np.argsort(-class_scores) 
        keep_local = local_sorted_args[:k]
        
        # Map back to global indices
        keep_global = class_idxs[keep_local]
        selected_indices.extend(keep_global.tolist())

    print(f"[Selection] Selected {len(selected_indices)} / {len(dataset)} samples.")
    return selected_indices

def get_class_adaptive_ratios(labels, base_ratio=0.7, max_ratio=1.0):
    """
    Calculates a dynamic selection ratio for each class based on sample count.
    
    Args:
        labels (torch.Tensor or list): The full list/tensor of training labels.
        base_ratio (float): The ratio for the majority class (e.g., 0.7).
        max_ratio (float): The ratio for the minority class (e.g., 1.0).
        
    Returns:
        dict: A dictionary mapping {class_index: selection_ratio}
    """
    # 1. Count samples per class
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
        
    classes, counts = np.unique(labels, return_counts=True)
    n_max = np.max(counts) # Count of majority class
    n_min = np.min(counts) # Count of minority class
    
    # Safety check for perfectly balanced data
    if n_max == n_min:
        return {c: base_ratio for c in classes}

    ratios = {}
    for cls, count in zip(classes, counts):
        # Linear Interpolation Formula:
        # If count is high (close to n_max) -> Ratio is low (base_ratio)
        # If count is low (close to n_min)  -> Ratio is high (max_ratio)
        
        # Calculate how "minority" this class is (0.0 = Majority, 1.0 = Minority)
        minority_factor = (n_max - count) / (n_max - n_min)
        
        # Scale the ratio
        r = base_ratio + (max_ratio - base_ratio) * minority_factor
        ratios[cls] = round(r, 4)
        
    return ratios