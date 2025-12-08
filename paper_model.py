import argparse
import time
import os
import logging
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from dataloader import create_dataloader 
from _density import compute_density_probability
from _consistency import ConsistencyCalculator
from selection import select_samples
from _imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from _augmentation import get_Trival_Augmentation, get_week_augmentation


# Modified Resnet 18
class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = models.resnet18(weights=None)
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x, return_features=False):
        # Standard ResNet18 forward
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

        if return_features:
            return features, logits
        return logits

# === HELPER TO HANDLE VARIABLE TUPLE SIZES ===
def unpack_batch(batch, device):
    """
    Automatically identifies Inputs (4D) and Targets (Labels).
    Handles cases: 
    1. (Image, Label)
    2. (Image, Label, Index)
    3. (Index, Image, Label)
    """
    inputs, targets = None, None
    
    # 1. Move everything to device first
    batch = [b.to(device, non_blocking=True) for b in batch]

    # 2. Find the Image (4D Tensor)
    for item in batch:
        if item.dim() == 4:
            inputs = item
            break
            
    # 3. Find the Target
    # If 3 items, and 1st is Index (1D), then 2nd is Image (4D), 3rd is Target (1D)
    if len(batch) == 3:
        if batch[0].dim() == 1 and batch[1].dim() == 4:
            # Case: (Index, Image, Label)
            targets = batch[2]
        else:
            # Case: (Image, Label, Index) -> Standard
            targets = batch[1]
    elif len(batch) == 2:
        # Case: (Image, Label)
        targets = batch[1]
    
    if inputs is None:
        raise ValueError(f"Could not find 4D Image Tensor in batch of shapes: {[b.shape for b in batch]}")
        
    return inputs, targets


def validate(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            # Smart Unpacking
            inputs, targets = unpack_batch(batch, device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return running_loss / total, correct / total

def train_one_epoch(model, loader, optimizer, device, criterion, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in loader:
        # Smart Unpacking
        inputs, targets = unpack_batch(batch, device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total

def extract_features(model, dataset, device):
    """Extracts features using the current dataset transform."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    all_feats = []
    with torch.no_grad():
        for batch in loader:
            # Manual unpacking for features (we only need the image)
            batch = [b.to(device) for b in batch]
            inputs = None
            for item in batch:
                if item.dim() == 4:
                    inputs = item
                    break
            
            features, _ = model(inputs, return_features=True)
            all_feats.append(features.cpu())
    return torch.cat(all_feats, dim=0).numpy()

def setup_logger(args):
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ratio_str = f"r{args.ratio}" if args.ratio is not None else "Baseline"
    log_filename = f"results/train_{args.dataset}_{args.augmentation}_{ratio_str}_{timestamp}.log"
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_filename)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_filename

# 3. Main Training Pipeline
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001) 
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--num-workers", type=int, default=16)
    
    # Overrides
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--augmentation", type=str, default=None)
    parser.add_argument("--ratio", type=float, default=None)

    args = parser.parse_args()

    # Experiment Grid (32 models)
    datasets = ["cifar10","cifar100","imbalance_cifar10","imbalance_cifar100"] 
    augmentations = ["trivial", "weak"] 
    select_ratios = [None, 0.5, 0.7, 0.9]

    # Handle Overrides
    if args.dataset: datasets = [args.dataset]
    if args.augmentation: augmentations = [args.augmentation]
    if args.ratio is not None: select_ratios = [args.ratio]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    exp_id = 0

    for dataset_name in datasets:
        for augmentation in augmentations:
            for select_ratio in select_ratios:
                exp_id += 1
                args.dataset = dataset_name
                args.augmentation = augmentation
                args.ratio = select_ratio
                
                logger, logfile = setup_logger(args)
                logger.info(f"STARTING EXP {exp_id}: {dataset_name} | {augmentation} | Ratio: {select_ratio}")

                # define Transforms
                mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                
                clean_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
                
                if augmentation == "trivial":
                    train_transform, _ = get_Trival_Augmentation()
                else:
                    train_transform, _ = get_week_augmentation()

                # create Dataset Instance
                if dataset_name == "cifar10":
                    from _cifar10 import cifar10_dataset
                    dataset = cifar10_dataset(root=args.data_dir, train=True, download=True, transform=clean_transform)
                    val_dataset = cifar10_dataset(root=args.data_dir, train=False, download=True, transform=clean_transform)
                    num_classes = 10

                elif dataset_name == "cifar100":
                    from _cifar100 import cifar100_dataset
                    dataset = cifar100_dataset(root=args.data_dir, train=True, download=True, transform=clean_transform)
                    val_dataset = cifar100_dataset(root=args.data_dir, train=False, download=True, transform=clean_transform)
                    num_classes = 100

                elif dataset_name == "imbalance_cifar10":
                    dataset = IMBALANCECIFAR10(root=args.data_dir, train=True, download=True, transform=clean_transform)
                    val_dataset = IMBALANCECIFAR10(root=args.data_dir, train=False, download=True, transform=clean_transform)
                    num_classes = 10

                elif dataset_name == "imbalance_cifar100":
                    dataset = IMBALANCECIFAR100(root=args.data_dir, train=True, download=True, transform=clean_transform)
                    val_dataset = IMBALANCECIFAR100(root=args.data_dir, train=False, download=True, transform=clean_transform)
                    num_classes = 100

                else:
                    raise ValueError("Unknown dataset")

                # check select ratio to perform dynamic selection
                p_con = np.ones(len(dataset)) 
                if select_ratio is not None:
                    # Set transform to clean for CLIP
                    dataset.transform = clean_transform 
                    logger.info("Calculating CLIP Consistency Scores...")
                    consistency_calc = ConsistencyCalculator(device=device)
                    p_con = consistency_calc.calculate(dataset)
                    del consistency_calc
                    torch.cuda.empty_cache()

                model = ResNet18_CIFAR(num_classes=num_classes).to(device)
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                criterion = nn.CrossEntropyLoss()
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
                scaler = torch.amp.GradScaler('cuda')

                # Validation Loader
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

                # Training Loop
                current_indices = None 
                
                for epoch in range(1, args.epochs + 1):
                    t0 = time.time()
                    
                    # 1. Dynamic Selection Phase (Clean Data)
                    if select_ratio is not None:
                        # calculate density from model feature vector every 5 epochs
                        if epoch == 1 or epoch % 5 == 0:
                            dataset.transform = clean_transform
                            logger.info(f"[Epoch {epoch}] Updating Selection...")
                            # get feature vectors from model
                            feats = extract_features(model, dataset, device)
                            # compute density
                            p_rho = compute_density_probability(feats)
                            current_indices = select_samples(dataset, p_rho, p_con, select_ratio)
                            logger.info(f"Selected {len(current_indices)} samples.")

                    # 2. Training Phase (Augmented Data)
                    dataset.transform = train_transform
                    
                    
                    train_loader, _ = create_dataloader(
                        train_dataset=dataset,
                        val_dataset=val_dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        select_indices=current_indices 
                    )

                    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion, scaler)
                    val_loss, val_acc = validate(model, val_loader, device, criterion)
                    
                    scheduler.step()
                    elapsed = time.time() - t0
                    
                    msg = (f"[EXP {exp_id}] Ep {epoch:03d} | "
                           f"TrLoss: {train_loss:.4f} Acc: {(train_acc*100):.2f}% | "
                           f"ValLoss: {val_loss:.4f} Acc: {(val_acc*100):.2f}% | Time: {elapsed:.1f}s")
                    print(msg)
                    logger.info(msg)

                logger.info("Experiment Finished.")

if __name__ == "__main__":
    main()