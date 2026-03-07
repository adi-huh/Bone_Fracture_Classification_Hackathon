"""
train.py — Training loop with early stopping & cross-validation
================================================================

• Multi-class CrossEntropyLoss
• AdamW optimiser  (lr=1e-4, weight_decay=1e-2)
• CosineAnnealingLR scheduler
• Early stopping on validation F1 (patience=10)
• Epoch-by-epoch CSV log  → model_performance_analysis.csv
• Best checkpoint saved    → checkpoints/model.pth
• 5-fold cross-validation  → run_cross_validation()
"""

import os
import csv
import copy
import time
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

from data_loader import load_config, set_seed, get_dataloaders, get_cv_dataloaders
from model import build_model

# ------------------------------------------------------------------ #
#  Device helper                                                      #
# ------------------------------------------------------------------ #

def get_device(config: dict) -> torch.device:
    requested = config.get("device", "auto")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


# ------------------------------------------------------------------ #
#  Single-epoch helpers                                               #
# ------------------------------------------------------------------ #

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
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
    running_loss = 0.0
    correct = 0
    total = 0
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
    return avg_loss, accuracy, f1


# ------------------------------------------------------------------ #
#  Main training routine                                              #
# ------------------------------------------------------------------ #

def train(config: dict | None = None, config_path: str = "config.yaml",
          resume: bool = False):
    """
    Two-phase training run:
      Phase 1 — Freeze CNN+ViT backbones, train fusion+classifier (fast).
      Phase 2 — Unfreeze all, fine-tune end-to-end with lower LR.

    If resume=True, loads the last checkpoint and continues training from where
    it left off (skipping already-completed epochs).

    Returns (model, history_dict, training_time_sec).
    """
    if config is None:
        config = load_config(config_path)

    set_seed(config.get("seed", 42))
    device = get_device(config)
    print(f"[train] Device: {device}")

    # Data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(config)
    num_classes = len(class_names)
    config["num_classes"] = num_classes
    config["class_names"] = class_names

    # Model
    model = build_model(config).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    total_epochs = config.get("epochs", 50)
    patience = config.get("early_stopping_patience", 10)
    ckpt_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Phase configuration ---- #
    freeze_epochs = config.get("freeze_epochs", 10)     # Phase 1 epochs
    finetune_epochs = total_epochs - freeze_epochs       # Phase 2 epochs
    base_lr = config.get("learning_rate", 1e-4)

    # CSV logger
    csv_path = "model_performance_analysis.csv"
    csv_columns = [
        "epoch", "train_loss", "val_loss",
        "train_accuracy", "val_accuracy",
        "overfitting_gap", "learning_rate",
    ]
    csv_rows = []

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    best_state = None
    wait = 0
    global_epoch = 0
    resume_epoch = 0          # epoch to resume from (0 = start fresh)

    # ---- Resume from checkpoint ---- #
    ckpt_path = os.path.join(ckpt_dir, "model.pth")
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        resume_epoch = ckpt["epoch"]
        best_val_f1 = ckpt.get("val_f1", 0.0)
        best_val_acc = ckpt.get("val_accuracy", 0.0)
        best_epoch = resume_epoch
        best_state = copy.deepcopy(model.state_dict())
        print(f"[train] ✓ Resumed from checkpoint epoch {resume_epoch} "
              f"(F1={best_val_f1:.4f}, Acc={best_val_acc:.2f}%)")

    start_time = time.time()

    # ============================================================ #
    #  PHASE 1: Frozen backbones — fast convergence of head         #
    # ============================================================ #
    p1_start = resume_epoch + 1 if resume_epoch < freeze_epochs else freeze_epochs + 1
    if p1_start <= freeze_epochs:
        print(f"\n{'='*60}")
        print(f"  PHASE 1: Frozen backbones (epochs {p1_start}–{freeze_epochs})")
        print(f"{'='*60}")
        model.freeze_backbones()

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr,
            weight_decay=config.get("weight_decay", 1e-2),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=freeze_epochs)
        # Step scheduler to match resumed epoch
        for _ in range(resume_epoch):
            scheduler.step()

        for ep in range(p1_start, freeze_epochs + 1):
            global_epoch = ep
            lr = optimizer.param_groups[0]["lr"]

            tr_loss, tr_acc, tr_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            vl_loss, vl_acc, vl_f1 = evaluate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            gap = tr_acc - vl_acc
            row = {
                "epoch": global_epoch,
                "train_loss": f"{tr_loss:.4f}",
                "val_loss": f"{vl_loss:.4f}",
                "train_accuracy": f"{tr_acc:.2f}",
                "val_accuracy": f"{vl_acc:.2f}",
                "overfitting_gap": f"{gap:.2f}",
                "learning_rate": f"{lr:.6f}",
            }
            csv_rows.append(row)

            print(
                f"[P1] Epoch {global_epoch:3d}/{total_epochs}  "
                f"TrLoss={tr_loss:.4f}  TrAcc={tr_acc:.2f}%  "
                f"VlLoss={vl_loss:.4f}  VlAcc={vl_acc:.2f}%  "
                f"VlF1={vl_f1:.4f}  LR={lr:.6f}"
            )

            if vl_f1 > best_val_f1:
                best_val_f1 = vl_f1
                best_val_acc = vl_acc
                best_epoch = global_epoch
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "epoch": global_epoch,
                        "model_state_dict": best_state,
                        "val_accuracy": best_val_acc,
                        "val_f1": best_val_f1,
                        "config": config,
                    },
                    os.path.join(ckpt_dir, "model.pth"),
                )
                print(f"  ✓ Saved best checkpoint (F1={best_val_f1:.4f})")
    else:
        print(f"\n[train] Phase 1 already completed (resume_epoch={resume_epoch})")
        global_epoch = freeze_epochs

    # ============================================================ #
    #  PHASE 2: Unfreeze all — end-to-end fine-tuning               #
    # ============================================================ #
    p2_start_epoch = max(freeze_epochs + 1, resume_epoch + 1)
    p2_local_start = p2_start_epoch - freeze_epochs   # 1-based within Phase 2
    if p2_local_start <= finetune_epochs:
        print(f"\n{'='*60}")
        print(f"  PHASE 2: Full fine-tuning (epochs {p2_start_epoch}–{total_epochs})")
        print(f"{'='*60}")
        model.unfreeze_backbones()
        wait = 0  # reset early stopping counter

        optimizer = AdamW(
            model.parameters(),
            lr=base_lr * 0.1,   # 10× lower LR for fine-tuning
            weight_decay=config.get("weight_decay", 1e-2),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs)
        # Step scheduler to match resumed position within Phase 2
        for _ in range(p2_local_start - 1):
            scheduler.step()

        for ep in range(p2_local_start, finetune_epochs + 1):
            global_epoch = freeze_epochs + ep
            lr = optimizer.param_groups[0]["lr"]

            tr_loss, tr_acc, tr_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            vl_loss, vl_acc, vl_f1 = evaluate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            gap = tr_acc - vl_acc
            row = {
                "epoch": global_epoch,
                "train_loss": f"{tr_loss:.4f}",
                "val_loss": f"{vl_loss:.4f}",
                "train_accuracy": f"{tr_acc:.2f}",
                "val_accuracy": f"{vl_acc:.2f}",
                "overfitting_gap": f"{gap:.2f}",
                "learning_rate": f"{lr:.6f}",
            }
            csv_rows.append(row)

            print(
                f"[P2] Epoch {global_epoch:3d}/{total_epochs}  "
                f"TrLoss={tr_loss:.4f}  TrAcc={tr_acc:.2f}%  "
                f"VlLoss={vl_loss:.4f}  VlAcc={vl_acc:.2f}%  "
                f"VlF1={vl_f1:.4f}  LR={lr:.6f}"
            )

            if vl_f1 > best_val_f1:
                best_val_f1 = vl_f1
                best_val_acc = vl_acc
                best_epoch = global_epoch
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
                torch.save(
                    {
                        "epoch": global_epoch,
                        "model_state_dict": best_state,
                        "val_accuracy": best_val_acc,
                        "val_f1": best_val_f1,
                        "config": config,
                    },
                    os.path.join(ckpt_dir, "model.pth"),
                )
                print(f"  ✓ Saved best checkpoint (F1={best_val_f1:.4f})")
            else:
                wait += 1
                if wait >= patience:
                    print(f"  ✗ Early stopping at epoch {global_epoch} (patience={patience})")
                    break

    training_time = time.time() - start_time

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Write epoch log CSV ---- #
    _write_performance_csv(csv_path, csv_columns, csv_rows,
                           best_val_acc, best_epoch, training_time, config)

    print(f"\n[train] Best val acc: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"[train] Training time: {training_time/60:.1f} min")
    print(f"[train] Checkpoint:    {ckpt_dir}/model.pth")
    print(f"[train] Epoch log:     {csv_path}")

    history = {
        "csv_rows": csv_rows,
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "training_time": training_time,
    }
    return model, history, training_time


# ------------------------------------------------------------------ #
#  Cross-validation                                                   #
# ------------------------------------------------------------------ #

def run_cross_validation(config: dict | None = None, config_path: str = "config.yaml"):
    """
    5-fold stratified cross-validation.

    Returns
    -------
    fold_results : list[dict]
        Per-fold val accuracy and F1.
    mean_acc, std_acc, mean_f1, std_f1
    """
    if config is None:
        config = load_config(config_path)

    set_seed(config.get("seed", 42))
    device = get_device(config)
    freeze_epochs = config.get("freeze_epochs", 10)
    finetune_epochs = 15  # limited fine-tuning for CV speed
    patience = config.get("early_stopping_patience", 7)
    base_lr = config.get("learning_rate", 1e-4)

    fold_results = []

    for fold, tr_loader, vl_loader, class_names in get_cv_dataloaders(config):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold + 1}")
        print(f"{'='*60}")

        config["num_classes"] = len(class_names)
        model = build_model(config).to(device)
        criterion = nn.CrossEntropyLoss()

        # --- Phase 1: frozen backbones ---
        model.freeze_backbones()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr,
            weight_decay=config.get("weight_decay", 1e-2),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=freeze_epochs)

        best_f1 = 0.0
        best_acc = 0.0

        for epoch in range(1, freeze_epochs + 1):
            train_one_epoch(model, tr_loader, criterion, optimizer, device)
            vl_loss, vl_acc, vl_f1 = evaluate(model, vl_loader, criterion, device)
            scheduler.step()
            if vl_f1 > best_f1:
                best_f1 = vl_f1
                best_acc = vl_acc
            print(f"  [F{fold+1}][P1] Epoch {epoch}/{freeze_epochs}  "
                  f"VlAcc={vl_acc:.2f}%  VlF1={vl_f1:.4f}")

        # --- Phase 2: unfrozen fine-tuning ---
        model.unfreeze_backbones()
        optimizer = AdamW(model.parameters(), lr=base_lr * 0.1,
                          weight_decay=config.get("weight_decay", 1e-2))
        scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs)
        wait = 0

        for epoch in range(1, finetune_epochs + 1):
            train_one_epoch(model, tr_loader, criterion, optimizer, device)
            vl_loss, vl_acc, vl_f1 = evaluate(model, vl_loader, criterion, device)
            scheduler.step()
            if vl_f1 > best_f1:
                best_f1 = vl_f1
                best_acc = vl_acc
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
            print(f"  [F{fold+1}][P2] Epoch {epoch}/{finetune_epochs}  "
                  f"VlAcc={vl_acc:.2f}%  VlF1={vl_f1:.4f}")

        fold_results.append({"fold": fold + 1, "val_accuracy": best_acc, "val_f1": best_f1})
        print(f"  Fold {fold + 1} — Best Val Acc: {best_acc:.2f}%  F1: {best_f1:.4f}")

    accs = [r["val_accuracy"] for r in fold_results]
    f1s  = [r["val_f1"]       for r in fold_results]
    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_f1,  std_f1  = np.mean(f1s),  np.std(f1s)

    print(f"\n[CV] Accuracy: {mean_acc:.2f} ± {std_acc:.2f}")
    print(f"[CV] F1:       {mean_f1:.4f} ± {std_f1:.4f}")

    # Save CV results to JSON for evaluate.py
    import json as _json
    cv_out = {
        "fold_results": fold_results,
        "mean_acc": float(mean_acc),
        "std_acc": float(std_acc),
        "mean_f1": float(mean_f1),
        "std_f1": float(std_f1),
    }
    cv_json_path = "cv_results.json"
    with open(cv_json_path, "w") as _f:
        _json.dump(cv_out, _f, indent=2)
    print(f"[CV] Saved → {cv_json_path}")

    return fold_results, mean_acc, std_acc, mean_f1, std_f1


# ------------------------------------------------------------------ #
#  CSV helper                                                         #
# ------------------------------------------------------------------ #

def _write_performance_csv(path, columns, rows,
                           best_val_acc, best_epoch, training_time, config):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

        # Blank row
        f.write("\n")
        # Generalization metrics summary
        f.write("# GENERALIZATION METRICS\n")
        gaps = [float(r["overfitting_gap"]) for r in rows]
        f.write(f"Max Overfitting Gap,{max(gaps):.2f}\n")
        f.write(f"Best Val Accuracy,{best_val_acc:.2f}\n")
        f.write(f"Best Epoch,{best_epoch}\n")
        f.write(f"Training Time (s),{training_time:.1f}\n")
        # Placeholders — filled after evaluate.py or CV run
        f.write("Test Accuracy,TBD (run evaluate.py)\n")
        f.write("Train/Test Accuracy Delta,TBD\n")
        f.write("CV Mean ± Std,TBD (run train.py --cv)\n")


# ------------------------------------------------------------------ #
#  CLI entry point                                                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train bone fracture classifier")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--cv", action="store_true", help="Run 5-fold cross-validation instead of normal training")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.cv:
        fold_results, mean_acc, std_acc, mean_f1, std_f1 = run_cross_validation(config)
    else:
        model, history, t = train(config, resume=args.resume)
        print("\n✅  Training complete.  Run `python evaluate.py` to compute test metrics.")
