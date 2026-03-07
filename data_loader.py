"""
data_loader.py — Dataset loading, augmentation, and stratified splits
======================================================================

Loads PNG/JPEG grayscale X-ray images for multi-class bone fracture
classification.  Supports two common Kaggle directory layouts:

  1. Pre-split:   dataset_path/{train,val,test}/{class_name}/*.png
  2. Flat:        dataset_path/{class_name}/*.png   (will be split on the fly)

Augmentation (training only):
  • Random horizontal flip
  • Random rotation ±15°
  • Random brightness / contrast
  • CLAHE contrast enhancement (great for X-rays)

All images are resized to 224×224 and normalised with ImageNet stats so they
are ready for an ImageNet-pretrained backbone.
"""

import os
import random
import yaml
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ------------------------------------------------------------------ #
#  Reproducibility helpers                                            #
# ------------------------------------------------------------------ #

def set_seed(seed: int = 42):
    """Fix every random seed we can reach."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  Transforms                                                         #
# ------------------------------------------------------------------ #

# ImageNet statistics (single-channel images are replicated to 3 channels)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224):
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


def get_eval_transforms(image_size: int = 224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ------------------------------------------------------------------ #
#  Dataset                                                            #
# ------------------------------------------------------------------ #

class BoneFractureDataset(Dataset):
    """
    Generic image-folder dataset for bone fracture classification.

    Parameters
    ----------
    image_paths : list[str]
        Absolute paths to image files.
    labels : list[int]
        Integer class labels.
    class_names : list[str]
        Human-readable class names (index == label).
    transform : albumentations.Compose | None
        Albumentations pipeline.
    """

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
            # Fallback: try grayscale → convert to 3-ch
            img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read {self.image_paths[idx]}")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        label = self.labels[idx]
        return img, label


# ------------------------------------------------------------------ #
#  Directory scanning helpers                                         #
# ------------------------------------------------------------------ #

def _scan_class_folder(root_dir, class_names):
    """Return (paths, labels) for images found under root_dir/{class}/."""
    paths, labels = [], []
    for idx, cname in enumerate(class_names):
        cls_dir = os.path.join(root_dir, cname)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(idx)
    return paths, labels


def _discover_class_names(dataset_path: str, config_names: list | None):
    """
    Try to discover class names from the folder structure.
    Falls back to the list given in config.yaml.
    """
    # Check if the dataset has a train/ subfolder
    for sub in ("train", "Train", "TRAIN"):
        train_dir = os.path.join(dataset_path, sub)
        if os.path.isdir(train_dir):
            names = sorted(
                d for d in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, d))
            )
            if names:
                return names

    # Flat layout — immediate sub-folders are classes
    names = sorted(
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
        and d.lower() not in ("train", "val", "test", "validation")
    )
    if names:
        return names

    if config_names:
        return config_names

    raise RuntimeError(
        f"Could not discover class names from {dataset_path}. "
        "Please list them in config.yaml → class_names."
    )


# ------------------------------------------------------------------ #
#  Public API: build DataLoaders                                      #
# ------------------------------------------------------------------ #

def get_dataloaders(config: dict | None = None, config_path: str = "config.yaml"):
    """
    Build train / val / test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, class_names
    """
    if config is None:
        config = load_config(config_path)

    set_seed(config.get("seed", 42))

    dataset_path = config["dataset_path"]
    image_size   = config.get("image_size", 224)
    batch_size   = config.get("batch_size", 32)
    num_workers  = config.get("num_workers", 0)

    class_names = _discover_class_names(
        dataset_path, config.get("class_names")
    )
    num_classes = len(class_names)

    print(f"[data_loader] Discovered {num_classes} classes: {class_names}")

    # ------ Try pre-split layout first ------
    train_dir = None
    for sub in ("train", "Train", "TRAIN"):
        p = os.path.join(dataset_path, sub)
        if os.path.isdir(p):
            train_dir = p
            break

    if train_dir is not None:
        # Look for val / test dirs
        val_dir, test_dir = None, None
        for sub in ("val", "Val", "VAL", "validation", "Validation"):
            p = os.path.join(dataset_path, sub)
            if os.path.isdir(p):
                val_dir = p
                break
        for sub in ("test", "Test", "TEST"):
            p = os.path.join(dataset_path, sub)
            if os.path.isdir(p):
                test_dir = p
                break

        train_paths, train_labels = _scan_class_folder(train_dir, class_names)
        if val_dir:
            val_paths, val_labels = _scan_class_folder(val_dir, class_names)
        else:
            val_paths, val_labels = [], []
        if test_dir:
            test_paths, test_labels = _scan_class_folder(test_dir, class_names)
        else:
            test_paths, test_labels = [], []

        # If val or test are missing, split from train
        if not val_paths or not test_paths:
            all_paths = train_paths + val_paths + test_paths
            all_labels = train_labels + val_labels + test_labels
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = (
                _stratified_split(all_paths, all_labels, config)
            )
    else:
        # Flat layout — split manually
        all_paths, all_labels = _scan_class_folder(dataset_path, class_names)
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = (
            _stratified_split(all_paths, all_labels, config)
        )

    # Print stats
    _print_split_stats("Train", train_labels, class_names)
    _print_split_stats("Val",   val_labels,   class_names)
    _print_split_stats("Test",  test_labels,  class_names)

    # Build datasets
    train_ds = BoneFractureDataset(
        train_paths, train_labels, class_names, get_train_transforms(image_size)
    )
    val_ds = BoneFractureDataset(
        val_paths, val_labels, class_names, get_eval_transforms(image_size)
    )
    test_ds = BoneFractureDataset(
        test_paths, test_labels, class_names, get_eval_transforms(image_size)
    )

    # pin_memory only helps on CUDA, not MPS
    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin,
    )

    return train_loader, val_loader, test_loader, class_names


# ------------------------------------------------------------------ #
#  Cross-validation splits                                            #
# ------------------------------------------------------------------ #

def get_cv_dataloaders(config: dict | None = None, config_path: str = "config.yaml"):
    """
    Yield (fold, train_loader, val_loader) for k-fold stratified CV.

    The *test* set is held out entirely and not used here.
    """
    if config is None:
        config = load_config(config_path)

    set_seed(config.get("seed", 42))
    dataset_path = config["dataset_path"]
    image_size   = config.get("image_size", 224)
    batch_size   = config.get("batch_size", 32)
    num_workers  = config.get("num_workers", 0)
    n_folds      = config.get("cv_folds", 5)

    class_names = _discover_class_names(dataset_path, config.get("class_names"))

    # Gather ALL images (train+val; test is excluded)
    all_paths, all_labels = [], []
    for sub in ("train", "Train", "TRAIN", "val", "Val", "VAL", "validation"):
        p = os.path.join(dataset_path, sub)
        if os.path.isdir(p):
            pp, ll = _scan_class_folder(p, class_names)
            all_paths.extend(pp)
            all_labels.extend(ll)

    if not all_paths:
        # Flat layout — use everything except test
        pp, ll = _scan_class_folder(dataset_path, class_names)
        # Hold out 20 % as test
        rest_p, _, rest_l, _ = train_test_split(
            pp, ll, test_size=0.2, stratify=ll, random_state=42
        )
        all_paths, all_labels = rest_p, rest_l

    all_labels_np = np.array(all_labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels_np)):
        tr_paths  = [all_paths[i] for i in train_idx]
        tr_labels = [all_labels[i] for i in train_idx]
        vl_paths  = [all_paths[i] for i in val_idx]
        vl_labels = [all_labels[i] for i in val_idx]

        tr_ds = BoneFractureDataset(tr_paths, tr_labels, class_names, get_train_transforms(image_size))
        vl_ds = BoneFractureDataset(vl_paths, vl_labels, class_names, get_eval_transforms(image_size))

        use_pin = torch.cuda.is_available()
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=use_pin, drop_last=True)
        vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=use_pin)

        yield fold, tr_loader, vl_loader, class_names


# ------------------------------------------------------------------ #
#  Internal helpers                                                   #
# ------------------------------------------------------------------ #

def _stratified_split(paths, labels, config):
    """70 / 10 / 20 stratified split."""
    train_r = config.get("train_ratio", 0.70)
    val_r   = config.get("val_ratio",   0.10)
    # test_r is the remainder

    # First split: train+val vs test (20 %)
    trval_p, test_p, trval_l, test_l = train_test_split(
        paths, labels,
        test_size=1.0 - train_r - val_r,
        stratify=labels,
        random_state=42,
    )
    # Second split: train vs val
    relative_val = val_r / (train_r + val_r)
    train_p, val_p, train_l, val_l = train_test_split(
        trval_p, trval_l,
        test_size=relative_val,
        stratify=trval_l,
        random_state=42,
    )
    return train_p, train_l, val_p, val_l, test_p, test_l


def _print_split_stats(name, labels, class_names):
    from collections import Counter
    c = Counter(labels)
    total = len(labels)
    print(f"  {name:5s}: {total:>6d} images  ", end="")
    for idx, cn in enumerate(class_names):
        print(f"| {cn}: {c.get(idx, 0)}", end=" ")
    print()
