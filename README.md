---
title: HybridMedicalVision
emoji: 🦴
colorFrom: blue
colorTo: green
license: MIT license
---

# Bone Fracture Classification — Hybrid CNN + ViT

A **Hybrid CNN + Vision Transformer** system for bone fracture classification from X-ray radiographs, built for the **KBG Hackathon 2025** (Kamand Bioengineering Group, IIT Mandi).

## Classification Task

| # | Class | Description |
|---|-------|-------------|
| 0 | Fractured | X-ray showing a bone fracture |
| 1 | Not Fractured | Normal bone X-ray |

## Model Architecture

```
Input (B, 3, 224, 224)
   ├── CNN Backbone (EfficientNet-B3, ImageNet pretrained)
   │     → local feature extraction (d=1536)
   └── ViT Head (vit_small_patch16_224, ImageNet pretrained)
         → global attention features (d=384)
         │
    Cross-Attention Fusion → (d=512)
         │
    Linear → ReLU → Dropout(0.3) → Linear(2)
```

**Important**: Only ImageNet pre-trained weights are used. No model is pre-trained on fracture or bone X-ray data.

## Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **98.22%** |
| **Macro F1** | **0.9822** |
| **AUC-ROC** | **0.9970** |
| Val Accuracy (best) | 99.16% |
| Inference Speed | 5.67 ms/image |
| Model Size | 130 MB |

## Project Structure

```
config.yaml                    ← All hyperparameters
data_loader.py                 ← Dataset loading, augmentation, splits
model.py                       ← Hybrid CNN + ViT architecture
train.py                       ← Training loop, cross-validation
evaluate.py                    ← Metrics computation, CSV export
explainability.py              ← Grad-CAM and attention maps
requirements.txt               ← Python dependencies
README.md                      ← This file
final_results.csv              ← (auto-generated) test-set metrics
model_performance_analysis.csv ← (auto-generated) epoch logs
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

Download the [Bone Fracture Binary Classification Dataset](https://www.kaggle.com/) and place it under the path specified in `config.yaml → dataset_path`.

Expected folder layout:

```
dataset_path/
  train/
    fractured/
    not fractured/
  val/
    fractured/
    not fractured/
  test/
    fractured/
    not fractured/
```

### 3. Train

```bash
python train.py --config config.yaml
```

### 4. Cross-validation (optional)

```bash
python train.py --config config.yaml --cv
```

### 5. Evaluate on test set

```bash
python evaluate.py --config config.yaml
```

This produces `final_results.csv` and updates `model_performance_analysis.csv`.

### 6. Explainability

```bash
python explainability.py --config config.yaml
```

Saves Grad-CAM and attention-map overlays to `results/explainability/`.

## Training Details

| Setting | Value |
|---------|-------|
| Loss | CrossEntropyLoss |
| Optimiser | AdamW (lr=1e-4, weight_decay=1e-2) |
| Scheduler | CosineAnnealingLR |
| Phase 1 | Frozen backbones, 10 epochs, lr=1e-4 |
| Phase 2 | Full fine-tuning, up to 40 epochs, lr=1e-5, early stopping (patience=10) |
| Image size | 224 × 224 |
| Batch size | 16 |
| Augmentation | HFlip, Rotation ±15°, Brightness/Contrast, CLAHE |
| Seed | 42 (fully reproducible) |

**Resume training** from a checkpoint:

```bash
python train.py --config config.yaml --resume
```

## Metrics Reported

- Overall Accuracy
- Per-class Precision, Recall, F1
- Macro F1-Score
- AUC-ROC (binary)
- Confusion Matrix
- Training time, inference time per image (ms), model size (MB)
- 5-fold cross-validation mean ± std

## Medical Disclaimer

This system is provided for **educational and hackathon purposes only**.  It is **NOT a medical device**.  Always consult qualified healthcare professionals for clinical decisions.

## Author

**Aditya Rai**

## License

MIT License — see LICENSE file for details.

---

*KBG Hackathon 2025 — IIT Mandi*
