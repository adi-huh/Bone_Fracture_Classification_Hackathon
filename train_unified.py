"""
train_unified.py — Train a single HybridCNNViT on 4 classes
=============================================================

Classes:
  0 = fractured       (bone X-ray)
  1 = not fractured    (bone X-ray)
  2 = NORMAL           (chest X-ray)
  3 = PNEUMONIA        (chest X-ray)

Both datasets are merged into one training/val/test set and the
same HybridCNNViT architecture learns to classify all four.

Usage:
    python train_unified.py
"""

import os
import copy
import csv
import time
import yaml
import random
import numpy as np
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, classification_report

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from model import build_model

# ── Reproducibility ────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Device ─────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Transforms ─────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_train_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.95, 1.05),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.5,
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_eval_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ── Dataset ────────────────────────────────────────────────────────
class UnifiedXrayDataset(Dataset):
    def __init__(self, image_paths, labels, class_names, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read {self.image_paths[idx]}")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, self.labels[idx]


# ── Scan folder for images ─────────────────────────────────────────
def scan_class_folder(root_dir, class_name, label_idx):
    """Scan root_dir/class_name/ for images and return (paths, labels)."""
    paths, labels = [], []
    cls_dir = os.path.join(root_dir, class_name)
    if not os.path.isdir(cls_dir):
        print(f"  [warn] Directory not found: {cls_dir}")
        return paths, labels
    for fname in sorted(os.listdir(cls_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            paths.append(os.path.join(cls_dir, fname))
            labels.append(label_idx)
    return paths, labels


# ── Build unified dataloaders ──────────────────────────────────────
def build_dataloaders(config):
    """
    Merge bone and chest datasets into unified train/val/test loaders.

    Unified class mapping:
      0 = fractured
      1 = not fractured
      2 = NORMAL        (chest)
      3 = PNEUMONIA      (chest)
    """
    bone_path = config["bone_dataset_path"]
    chest_path = config["chest_dataset_path"]
    image_size = config.get("image_size", 224)
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 2)

    CLASS_NAMES = ["fractured", "not fractured", "NORMAL", "PNEUMONIA"]

    # ── Bone dataset ───────────────────────────────────────────
    bone_train_dir = os.path.join(bone_path, "train")
    bone_val_dir = os.path.join(bone_path, "val")
    bone_test_dir = os.path.join(bone_path, "test")

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    for split_dir, sp, sl in [
        (bone_train_dir, train_paths, train_labels),
        (bone_val_dir, val_paths, val_labels),
        (bone_test_dir, test_paths, test_labels),
    ]:
        for cls_name, cls_idx in [("fractured", 0), ("not fractured", 1)]:
            p, l = scan_class_folder(split_dir, cls_name, cls_idx)
            sp.extend(p)
            sl.extend(l)

    print(f"[data] Bone — Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")

    # ── Chest dataset ──────────────────────────────────────────
    # Try direct path first, then nested chest_xray/chest_xray
    for chest_base in [chest_path, os.path.join(chest_path, "chest_xray")]:
        chest_train = os.path.join(chest_base, "train")
        if os.path.isdir(chest_train):
            chest_path_resolved = chest_base
            break
    else:
        chest_path_resolved = chest_path

    chest_train_dir = os.path.join(chest_path_resolved, "train")
    chest_val_dir = os.path.join(chest_path_resolved, "val")
    chest_test_dir = os.path.join(chest_path_resolved, "test")

    for split_dir, sp, sl in [
        (chest_train_dir, train_paths, train_labels),
        (chest_val_dir, val_paths, val_labels),
        (chest_test_dir, test_paths, test_labels),
    ]:
        for cls_name, cls_idx in [("NORMAL", 2), ("PNEUMONIA", 3)]:
            p, l = scan_class_folder(split_dir, cls_name, cls_idx)
            sp.extend(p)
            sl.extend(l)

    print(f"[data] Chest — added to splits")
    print(f"[data] UNIFIED — Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")

    # Print per-class counts
    for split_name, labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        c = Counter(labels)
        parts = [f"{CLASS_NAMES[i]}: {c.get(i, 0)}" for i in range(4)]
        print(f"  {split_name:5s}: {len(labels):>6d}  |  {' | '.join(parts)}")

    # Build datasets
    train_ds = UnifiedXrayDataset(train_paths, train_labels, CLASS_NAMES, get_train_transforms(image_size))
    val_ds = UnifiedXrayDataset(val_paths, val_labels, CLASS_NAMES, get_eval_transforms(image_size))
    test_ds = UnifiedXrayDataset(test_paths, test_labels, CLASS_NAMES, get_eval_transforms(image_size))

    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=use_pin)

    return train_loader, val_loader, test_loader, CLASS_NAMES


# ── Training helpers ───────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, f1, all_preds, all_labels


# ── Main training ──────────────────────────────────────────────────
def train_unified(config_path="config_unified.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device = get_device()
    print(f"[train] Device: {device}")

    # Data
    train_loader, val_loader, test_loader, class_names = build_dataloaders(config)
    num_classes = len(class_names)
    config["num_classes"] = num_classes
    config["class_names"] = class_names

    # Model (4 classes)
    model = build_model(config).to(device)

    # Class weights for imbalanced data
    train_labels = train_loader.dataset.labels
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    weights = torch.tensor(
        [total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)
    print(f"[train] Class weights: {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    total_epochs = config.get("epochs", 15)
    patience = config.get("early_stopping_patience", 10)
    freeze_epochs = config.get("freeze_epochs", 7)
    finetune_epochs = total_epochs - freeze_epochs
    base_lr = config.get("learning_rate", 1e-4)

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "unified_model.pth")

    csv_path = "unified_training_log.csv"
    csv_columns = ["epoch", "phase", "train_loss", "val_loss",
                   "train_accuracy", "val_accuracy", "val_f1", "learning_rate"]
    csv_rows = []

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    wait = 0

    start_time = time.time()

    # ── Phase 1: Frozen backbones ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Frozen backbones (epochs 1–{freeze_epochs})")
    print(f"{'='*60}")
    model.freeze_backbones()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr, weight_decay=config.get("weight_decay", 1e-2),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=freeze_epochs)

    for ep in range(1, freeze_epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        csv_rows.append({
            "epoch": ep, "phase": "frozen",
            "train_loss": f"{tr_loss:.4f}", "val_loss": f"{vl_loss:.4f}",
            "train_accuracy": f"{tr_acc:.2f}", "val_accuracy": f"{vl_acc:.2f}",
            "val_f1": f"{vl_f1:.4f}", "learning_rate": f"{lr:.6f}",
        })

        print(f"[P1] Epoch {ep:2d}/{total_epochs}  "
              f"TrLoss={tr_loss:.4f}  TrAcc={tr_acc:.2f}%  "
              f"VlLoss={vl_loss:.4f}  VlAcc={vl_acc:.2f}%  "
              f"VlF1={vl_f1:.4f}  LR={lr:.6f}")

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_val_acc = vl_acc
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch": ep,
                "model_state_dict": best_state,
                "val_accuracy": best_val_acc,
                "val_f1": best_val_f1,
                "class_names": class_names,
                "config": config,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (F1={best_val_f1:.4f})")

    # ── Phase 2: Full fine-tuning ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Full fine-tuning (epochs {freeze_epochs+1}–{total_epochs})")
    print(f"{'='*60}")
    model.unfreeze_backbones()
    wait = 0

    optimizer = AdamW(
        model.parameters(),
        lr=base_lr * 0.1,
        weight_decay=config.get("weight_decay", 1e-2),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs)

    for ep_local in range(1, finetune_epochs + 1):
        ep_global = freeze_epochs + ep_local
        lr = optimizer.param_groups[0]["lr"]

        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        csv_rows.append({
            "epoch": ep_global, "phase": "finetune",
            "train_loss": f"{tr_loss:.4f}", "val_loss": f"{vl_loss:.4f}",
            "train_accuracy": f"{tr_acc:.2f}", "val_accuracy": f"{vl_acc:.2f}",
            "val_f1": f"{vl_f1:.4f}", "learning_rate": f"{lr:.6f}",
        })

        print(f"[P2] Epoch {ep_global:2d}/{total_epochs}  "
              f"TrLoss={tr_loss:.4f}  TrAcc={tr_acc:.2f}%  "
              f"VlLoss={vl_loss:.4f}  VlAcc={vl_acc:.2f}%  "
              f"VlF1={vl_f1:.4f}  LR={lr:.6f}")

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_val_acc = vl_acc
            best_epoch = ep_global
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            torch.save({
                "epoch": ep_global,
                "model_state_dict": best_state,
                "val_accuracy": best_val_acc,
                "val_f1": best_val_f1,
                "class_names": class_names,
                "config": config,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (F1={best_val_f1:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print(f"  ✗ Early stopping at epoch {ep_global}")
                break

    training_time = time.time() - start_time

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Final test evaluation ──────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL TEST EVALUATION")
    print(f"{'='*60}")
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\n  Test Accuracy: {test_acc:.2f}%")
    print(f"  Test F1 (macro): {test_f1:.4f}")
    print(f"\n{classification_report(test_labels, test_preds, target_names=class_names)}")

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[train] Best Val Acc: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"[train] Training time: {training_time/60:.1f} min")
    print(f"[train] Checkpoint: {ckpt_path}")
    print(f"[train] Training log: {csv_path}")
    print("\n✅ Training complete!")

    return model, class_names


if __name__ == "__main__":
    train_unified()
