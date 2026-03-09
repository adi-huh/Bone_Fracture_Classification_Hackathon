#!/usr/bin/env python3
"""
deploy_to_hf.py — Deploy the Bone Fracture Classifier to Hugging Face Spaces
==============================================================================

Creates a Hugging Face Space with all files needed to run the app.
Uses the huggingface_hub API to upload model, code, and assets.

Usage:
    python deploy_to_hf.py
"""

import os
import sys
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
HF_USERNAME = "Adi3003"                              # your HF username
SPACE_NAME  = "Bone-Fracture-Classifier"             # Space name
REPO_ID     = f"{HF_USERNAME}/{SPACE_NAME}"
ROOT        = Path(__file__).parent
DEPLOY_DIR  = ROOT / "hf_space_deploy"               # staging directory

# ── Files to include ───────────────────────────────────────────────
NEEDED_FILES = {
    # Source code
    "app.py":          ROOT / "app.py",
    "model.py":        ROOT / "model.py",
    "data_loader.py":  ROOT / "data_loader.py",
    "config.yaml":     ROOT / "config.yaml",
    # Model checkpoint
    "checkpoints/model.pth": ROOT / "checkpoints" / "model.pth",
    # Result images
    "results/confusion_matrix.png":                           ROOT / "results" / "confusion_matrix.png",
    "results/explainability/fractured_explainability.png":     ROOT / "results" / "explainability" / "fractured_explainability.png",
    "results/explainability/not fractured_explainability.png": ROOT / "results" / "explainability" / "not fractured_explainability.png",
    "results/explainability/fractured_gradcam.png":           ROOT / "results" / "explainability" / "fractured_gradcam.png",
    "results/explainability/not fractured_gradcam.png":       ROOT / "results" / "explainability" / "not fractured_gradcam.png",
    "results/explainability/fractured_attention.png":          ROOT / "results" / "explainability" / "fractured_attention.png",
    "results/explainability/not fractured_attention.png":      ROOT / "results" / "explainability" / "not fractured_attention.png",
}


def create_space_requirements():
    """Create requirements.txt for HF Space (CPU-only)."""
    return """torch
torchvision
timm
albumentations
opencv-python-headless
numpy
Pillow
gradio
pyyaml
scikit-learn
"""


def create_space_readme():
    """Create README.md for HF Space with YAML front-matter."""
    return """---
title: Bone Fracture Classifier
emoji: 🦴
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.9.0"
app_file: app.py
pinned: true
license: mit
short_description: "CNN+ViT bone fracture classifier with Grad-CAM"
---

# 🦴 Bone Fracture Classification System

**Hybrid CNN + Vision Transformer with Cross-Attention Fusion**

KBG Hackathon 2025 — Kamand Bioengineering Group, IIT Mandi

## Model Performance
| Metric | Value |
|--------|-------|
| Test Accuracy | **98.22%** |
| Macro F1-Score | **0.9822** |
| AUC-ROC | **0.9970** |
| Inference Speed | 5.67 ms/image |

## Architecture
- **CNN**: EfficientNet-B3 (local features, d=1536)
- **ViT**: ViT-Small-Patch16-224 (global attention, d=384)
- **Fusion**: Cross-Attention (d=512)
- **Training**: 2-Phase Transfer Learning (ImageNet pretrained only)

## Usage
Upload a bone X-ray image to get:
- Fracture classification with confidence scores
- Grad-CAM heatmap showing model focus regions

## Links
- [GitHub Repository](https://github.com/adi-huh/Bone_Fracture_Classification_Hackathon)
"""


def main():
    print("=" * 60)
    print("Deploying to Hugging Face Spaces".center(60))
    print(f"Repository: {REPO_ID}".center(60))
    print("=" * 60)

    # 1. Check all files exist
    print("\n[1/4] Checking files...")
    missing = []
    for name, path in NEEDED_FILES.items():
        if not path.exists():
            missing.append(str(path))
            print(f"  ❌ MISSING: {name} -> {path}")
        else:
            size = path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {name} ({size:.1f} MB)")

    if missing:
        print(f"\n⚠️  {len(missing)} file(s) missing. They will be skipped.")

    # 2. Create staging directory
    print("\n[2/4] Staging files...")
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    # Copy files to staging
    import shutil
    for name, path in NEEDED_FILES.items():
        if path.exists():
            dest = DEPLOY_DIR / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            print(f"  → {name}")

    # Create Space-specific files
    (DEPLOY_DIR / "requirements.txt").write_text(create_space_requirements())
    print("  → requirements.txt (Space)")
    (DEPLOY_DIR / "README.md").write_text(create_space_readme())
    print("  → README.md (Space)")

    # 3. Create/get the Space
    print("\n[3/4] Creating Hugging Face Space...")
    api = HfApi()

    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False,
        )
        print(f"  ✅ Space ready: https://huggingface.co/spaces/{REPO_ID}")
    except Exception as e:
        print(f"  ⚠️  Note: {e}")
        print("  Continuing with upload...")

    # 4. Upload everything
    print("\n[4/4] Uploading files to Space...")
    try:
        upload_folder(
            folder_path=str(DEPLOY_DIR),
            repo_id=REPO_ID,
            repo_type="space",
            commit_message="Deploy Bone Fracture Classifier — KBG Hackathon 2025",
        )
        print(f"\n{'=' * 60}")
        print("✅ DEPLOYMENT SUCCESSFUL!".center(60))
        print(f"{'=' * 60}")
        print(f"\n🔗 Live URL: https://huggingface.co/spaces/{REPO_ID}")
        print(f"   (May take 2-5 minutes to build on first deploy)\n")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Run: huggingface-cli login")
        print("  2. Enter your HF token from https://huggingface.co/settings/tokens")
        print(f"  3. Re-run: python {__file__}")
        sys.exit(1)

    # Cleanup staging
    shutil.rmtree(DEPLOY_DIR)
    print("🧹 Staging directory cleaned up.")


if __name__ == "__main__":
    main()
