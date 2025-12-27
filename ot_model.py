import argparse
import time
import os
import logging
from datetime import datetime
import numpy as np
import lava

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from dataloader import create_dataloader 
from _density import compute_density_probability
from _consistency import ConsistencyCalculator
from selection import select_samples, get_class_adaptive_ratios, get_targets_safe
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

def validate(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        # CLEAN: Use *rest to ignore indices if they exist
        for inputs, targets, *rest in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

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
    
    # CLEAN: Use *rest to ignore indices if they exist
    for inputs, targets, *rest in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

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
        # CLEAN: We only need inputs, ignore everything else
        for inputs, *rest in loader:
            inputs = inputs.to(device)
            features, _ = model(inputs, return_features=True)
            all_feats.append(features.cpu())
    return torch.cat(all_feats, dim=0).numpy()

def setup_logger(args, exp_name):
    os.makedirs("results", exist_ok=True)
    
    # Filename matches exp_name for easy skipping
    log_filename = f"results/{exp_name}.log"
    
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

def run_lava_selection(model, train_dataset, val_dataset):
    """
    Runs LAVA Optimal Transport to calculate the value of each training sample.
    """
   
    train_loader_lava = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    val_loader_lava = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    
    loaders = {'train': train_loader_lava, 'test': val_loader_lava}
    

    shuffle_ind = [] 
    
    print(">>> [LAVA] Computing Transport Cost (This may take a moment)...")
 
    dual_sol, _ = lava.compute_dual(
        model, 
        loaders['train'], 
        loaders['test'], 
        len(train_dataset), 
        shuffle_ind, 
        resize=32
    )
    
    return dual_sol

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=200)
    
    # === OPTIMIZED DEFAULTS ===
    parser.add_argument("--batch-size", type=int, default=128) 
    parser.add_argument("--num-workers", type=int, default=16) 
    parser.add_argument("--lr", type=float, default=0.001) 
    parser.add_argument("--device", default="cuda:1")
    # ==========================
    
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--augmentation", type=str, default=None)
    parser.add_argument("--ratio", type=float, default=None)

    args = parser.parse_args()

    # ... (Experiment Grid Setup - Same as before) ...
    # [Keep lines 180-210 from your code: datasets, augmentations, loops...]
    
    datasets = ["cifar10","cifar100","imbalance_cifar10","imbalance_cifar100"] 
    augmentations = ["trivial", "weak"] 
    select_ratios = [None, 0.7, 0.8, 0.9]

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
                
                ratio_str = f"r{select_ratio}" if select_ratio is not None else "Baseline"
                exp_name = f"train_{dataset_name}_{augmentation}_{ratio_str}"
                
                potential_log = f"results/{exp_name}.log"
                if os.path.exists(potential_log):
                    with open(potential_log, 'r') as f:
                        if "Experiment Finished" in f.read():
                            print(f"[SKIP] Experiment {exp_name} already completed.")
                            continue

                logger, logfile = setup_logger(args, exp_name)
                logger.info(f"STARTING EXP {exp_id}: {exp_name}")

                mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                clean_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
                
                if augmentation == "trivial":
                    train_transform, _ = get_Trival_Augmentation()
                else:
                    train_transform, _ = get_week_augmentation()

                if dataset_name == "cifar10":
                    from _cifar10 import cifar10_dataset
                    dataset = cifar10_dataset(root=args.data_dir, train=True, download=True, transform=clean_transform)
                    val_dataset = cifar10_dataset(root=args.data_dir, train=False, download=True, transform=clean_transform)
                    num_classes = 10
                elif dataset_name == "imbalance_cifar10":
                    dataset = IMBALANCECIFAR10(root=args.data_dir, train=True, download=True, transform=clean_transform)
                    val_dataset = IMBALANCECIFAR10(root=args.data_dir, train=False, download=True, transform=clean_transform)
                    num_classes = 10

                elif dataset_name == "imbalance_cifar100":
                    dataset = IMBALANCECIFAR100(root=args.data_dir, train=True, download=True, transform=clean_transform)
                    val_dataset = IMBALANCECIFAR100(root=args.data_dir, train=False, download=True, transform=clean_transform)
                    num_classes = 100

                model = ResNet18_CIFAR(num_classes=num_classes).to(device)
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                criterion = nn.CrossEntropyLoss()
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
                scaler = torch.amp.GradScaler('cuda')

                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
                )

                # Initialize Train Loader (Initial state)
                dataset.transform = train_transform
                current_indices = None
                
                train_loader, _ = create_dataloader(
                    train_dataset=dataset, val_dataset=val_dataset,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    select_indices=current_indices 
                )


                for epoch in range(1, args.epochs + 1):
                    t0 = time.time()
                    
                # 1. Dynamic Selection Phase
                if select_ratio is not None:
                    # Update selection every 5 epochs (LAVA is slow, don't run every epoch)
                    if epoch == 1 or epoch % 5 == 0:
                        logger.info(f"[Epoch {epoch}] Updating Selection with LAVA...")
                        
                        # A. Calculate how many samples to keep
                        num_total = len(dataset)
                        num_keep = int(num_total * select_ratio)

                        # B. Switch to Clean Transform
                        dataset.transform = clean_transform
                        
                        # [LAVA] Computing Transport Cost
                        lava_scores = run_lava_selection(model, dataset, val_dataset)

                        # --- ROBUST FIX START ---
                        # 1. Handle if lava_scores is a LIST (e.g., per-class scores)
                        if isinstance(lava_scores, list):
                            processed_list = []
                            for s in lava_scores:
                                # Ensure it's numpy
                                if torch.is_tensor(s):
                                    s = s.detach().cpu().numpy()
                                else:
                                    s = np.array(s)
                                
                                # Flatten to 1D immediately (Fixes the (1, N) vs (1, M) mismatch)
                                processed_list.append(s.flatten())
                            
                            # Now safe to concatenate 1D arrays of different lengths
                            lava_scores = np.concatenate(processed_list)
                        
                        # 2. Handle if lava_scores is a single Tensor
                        elif torch.is_tensor(lava_scores):
                            lava_scores = lava_scores.detach().cpu().numpy().flatten()
                            
                        # 3. Handle if lava_scores is already a single Numpy array
                        else:
                            lava_scores = np.array(lava_scores).flatten()
                        # --- ROBUST FIX END ---

                        # C. Sort and Select
                        sorted_indices = np.argsort(lava_scores)[::-1]
                        current_indices = sorted_indices[:num_keep]
                        
                        logger.info(f"[LAVA] Selected {len(current_indices)} / {num_total} samples.")
                        
                        # D. Switch back to Train Transform
                        dataset.transform = train_transform

                        # Update Train Loader
                        train_loader, _ = create_dataloader(
                            train_dataset=dataset, val_dataset=val_dataset,
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            select_indices=current_indices 
                        )
                    
                    # 2. Training Phase
                    # (Note: We use the existing train_loader, we don't recreate it if not needed)
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
