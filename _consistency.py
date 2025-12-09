import torch
import numpy as np
import clip
from torch.utils.data import DataLoader
from tqdm import tqdm

class ConsistencyCalculator:
    def __init__(self, class_names=None, device=None, model_name="ViT-B/32"):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CLIP] Loading {model_name} model on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.class_names = class_names

    def _resolve_class_names(self, dataset):
        if self.class_names is not None:
            return self.class_names
        if hasattr(dataset, "classes") and dataset.classes:
            return dataset.classes
        raise ValueError("Class names could not be determined.")

    def encode_images(self, dataset, batch_size=128):
        # Save original transform
        original_transform = getattr(dataset, 'transform', None)
        dataset.transform = self.preprocess 

        # Use 8 workers for speed
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        all_features = []
        print(f"[CLIP] Encoding {len(dataset)} images...")
        
        with torch.no_grad():
            # === UPDATE: Handle (Image, Label, Index) ===
            # 'images' grabs the 1st item. '*rest' grabs the label and index.
            for images, *rest in tqdm(loader):
                images = images.to(self.device)
                features = self.model.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features.cpu())
        
        dataset.transform = original_transform
        return torch.cat(all_features, dim=0)

    def encode_texts(self, labels, class_names, batch_size=128):
        unique_labels = torch.unique(torch.tensor(labels)).tolist()
        text_features_map = {}

        print("[CLIP] Encoding text prompts...")
        with torch.no_grad():
            for label_idx in unique_labels:
                class_name = class_names[label_idx]
                prompt = f"a photo of a {class_name}"
                token = clip.tokenize([prompt]).to(self.device)
                feat = self.model.encode_text(token)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                text_features_map[label_idx] = feat.cpu()

        aligned_text_feats = []
        for l in labels:
            aligned_text_feats.append(text_features_map[l])
            
        return torch.cat(aligned_text_feats, dim=0)

    def calculate(self, dataset, batch_size=128):
        class_names = self._resolve_class_names(dataset)
        
        # 1. Encode Images
        img_feats = self.encode_images(dataset, batch_size)

        # 2. Get Targets
        if hasattr(dataset, "targets"):
            labels = dataset.targets
        else:
            labels = []
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            # === UPDATE: Handle 3-item tuple here too ===
            for batch in loader:
                # batch[1] is the Label (since we fixed _cifar10.py)
                labels.extend(batch[1].tolist())
        
        # 3. Encode Texts
        txt_feats = self.encode_texts(labels, class_names, batch_size)

        # 4. Calculate Scores
        print("[CLIP] Calculating similarity...")
        scores = torch.nn.functional.cosine_similarity(img_feats, txt_feats)
        scores = scores.abs().numpy()
        
        if scores.max() > scores.min():
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores_norm = np.ones_like(scores)

        return scores_norm