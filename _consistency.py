import torch
import numpy as np
import clip
from torch.utils.data import DataLoader
import sys

class ConsistencyCalculator:
    """
    Calculate semantic consistency scores using CLIP.
    Formula 2: | CosineSimilarity(Image_Features, Text_Features) |
    """

    def __init__(self, class_names=None, device=None, model_name="ViT-B/32"):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[CLIP] Loading {model_name} model on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self.class_names = class_names

    def _resolve_class_names(self, dataset):
        """Attempts to find class names from the dataset if not provided."""
        if self.class_names is not None:
            return self.class_names
        
        if hasattr(dataset, "classes") and dataset.classes:
            print("[CLIP] Found class names in dataset attributes.")
            return dataset.classes
        
        raise ValueError(
            "Class names could not be determined. "
            "Please pass 'class_names' list to the constructor."
        )

    def encode_images(self, dataset, batch_size=128):
        """Encodes images using a DataLoader to handle any dataset type."""
        all_features = []
        
        # Create a loader that applies CLIP's preprocessing
        # We assume the dataset returns (image, label)
        # We need to override the transform to match CLIP's expectation
        
        # Save original transform to restore later
        original_transform = getattr(dataset, 'transform', None)
        dataset.transform = self.preprocess 

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"[CLIP] Encoding {len(dataset)} images...")
        
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                features = self.model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features.cpu())
        
        # Restore original transform
        dataset.transform = original_transform
        
        return torch.cat(all_features, dim=0)

    def encode_texts(self, labels, class_names, batch_size=128):
        """Encodes text prompts: 'a photo of a {class_name}'"""
        # Convert integer labels to text prompts
        unique_labels = torch.unique(torch.tensor(labels)).tolist()
        text_features_map = {}

        print("[CLIP] Encoding text prompts...")
        with torch.no_grad():
            # Optimize: only encode unique class names once
            for label_idx in unique_labels:
                class_name = class_names[label_idx]
                prompt = f"a photo of a {class_name}"
                token = clip.tokenize([prompt]).to(self.device)
                feat = self.model.encode_text(token)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                text_features_map[label_idx] = feat.cpu()

        # Map back to full dataset size
        # This creates a tensor of shape (N, 512) aligned with images
        aligned_text_feats = []
        for l in labels:
            aligned_text_feats.append(text_features_map[l])
            
        return torch.cat(aligned_text_feats, dim=0)

    def calculate(self, dataset, batch_size=128):
        class_names = self._resolve_class_names(dataset)
        print(f"[CLIP] Using {len(class_names)} classes: {class_names[:5]}...")

        # 1. Encode Images
        img_feats = self.encode_images(dataset, batch_size)

        # 2. Get Targets (generic handling)
        if hasattr(dataset, "targets"):
            labels = dataset.targets
        else:
            # Fallback for datasets without .targets (iterating)
            labels = []
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for _, batch_labels in loader:
                labels.extend(batch_labels.tolist())
        
        # 3. Encode Texts
        txt_feats = self.encode_texts(labels, class_names, batch_size)

        # 4. Formula 2: Absolute Cosine Similarity
        print("[CLIP] Calculating similarity...")
        scores = torch.nn.functional.cosine_similarity(img_feats, txt_feats)
        scores = scores.abs().numpy()
        
        # Normalize to [0, 1]
        if scores.max() > scores.min():
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores_norm = np.ones_like(scores)

        print(f"[CLIP] Score range: {scores_norm.min():.4f} -> {scores_norm.max():.4f}")
        return scores_norm

if __name__ == "__main__":
    pass