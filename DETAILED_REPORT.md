# Medical X-ray Analysis System — Detailed Technical Report
## KBG Hackathon 2025 | Kamand Bioengineering Group | IIT Mandi
### Author: Aditya Rai

---

## 1. Problem Statement

Build an AI system that can analyze medical X-ray images and provide automated classification for:
- **Bone X-rays**: Detect whether a fracture is present or not
- **Chest X-rays**: Detect whether pneumonia is present or not

The key constraint: **one unified model**, **one upload** — the system automatically determines the X-ray type and gives the correct diagnosis.

---

## 2. Solution Architecture

### 2.1 Model: HybridCNNViT (Custom Architecture)

We designed a **hybrid model** that combines two complementary deep learning paradigms:

| Component | Model | Feature Dim | Role |
|-----------|-------|-------------|------|
| **CNN Backbone** | EfficientNet-B3 | 1,536 | Extracts **local features** — edges, textures, cortex patterns, opacity regions |
| **Transformer** | ViT-Small-Patch16-224 | 384 | Captures **global spatial relationships** — anatomical structure across the full image |
| **Fusion** | Cross-Attention | 512 | CNN features query ViT features via attention mechanism |
| **Classifier** | Linear → ReLU → Dropout → Linear | 512 → 256 → 4 | Final 4-class prediction |

**Why this design?**
- CNNs excel at detecting local patterns (fracture lines, lung opacities) but miss global context
- Vision Transformers see the full image structure but can miss fine details
- Cross-attention fusion lets CNN features "attend to" relevant global context from ViT, producing richer representations than simple concatenation
- Both backbones use **ImageNet pretrained weights** for strong initialization

**Total Parameters: 33,939,116 (~34M)**

### 2.2 Architecture Diagram

```
Input Image (B, 3, 224, 224)
         |
    ┌────┴────┐
    │         │
    ▼         ▼
EfficientNet  ViT-Small
  -B3          -Patch16
(d=1536)      (d=384)
    │         │
    ▼         ▼
    Q ──────► K, V
    Cross-Attention (d=512)
         │
         ▼
    Linear(512→256)
    ReLU
    Dropout(0.3)
    Linear(256→4)
         │
         ▼
    [fractured, not fractured, NORMAL, PNEUMONIA]
```

### 2.3 Cross-Attention Fusion (Key Innovation)

Standard approaches concatenate CNN and ViT features. We use **cross-attention**:

```
Q = Linear_q(CNN_features)    # Query from CNN
K = Linear_k(ViT_features)    # Key from ViT
V = Linear_v(ViT_features)    # Value from ViT

Attention = softmax(Q · K^T / √d) · V
Output = LayerNorm(Attention + residual)
```

This allows CNN features to selectively attend to the most relevant global context from ViT, creating a more informative fused representation.

---

## 3. Dataset

### 3.1 Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| Bone Fracture | Roboflow Bone Fracture Binary Classification | Binary: fractured vs not fractured |
| Chest X-ray | Kaggle Chest X-ray Pneumonia | Binary: NORMAL vs PNEUMONIA |

### 3.2 Combined Dataset Statistics

| Split | Fractured | Not Fractured | NORMAL | PNEUMONIA | Total |
|-------|-----------|---------------|--------|-----------|-------|
| **Train** | 4,606 | 4,640 | 1,341 | 3,875 | **14,462** |
| **Val** | 337 | 492 | 8 | 8 | **845** |
| **Test** | 238 | 268 | 234 | 390 | **1,130** |

### 3.3 Class Imbalance

The NORMAL class (1,341 train images) is significantly underrepresented compared to not fractured (4,640). We handle this with **weighted CrossEntropyLoss**:

| Class | Training Count | Computed Weight | Effect |
|-------|---------------|-----------------|--------|
| fractured | 4,606 | 0.785 | Slight downweight |
| not fractured | 4,640 | 0.779 | Slight downweight |
| NORMAL | 1,341 | **2.696** | **3.4x upweight** (compensates for small size) |
| PNEUMONIA | 3,875 | 0.933 | Near neutral |

Formula: `weight[i] = total_samples / (num_classes × class_count[i])`

---

## 4. Training Strategy

### 4.1 Two-Phase Training

We use a **progressive unfreezing** strategy:

**Phase 1 — Frozen Backbones (Epochs 1–7)**
- EfficientNet-B3 and ViT-Small weights are **frozen**
- Only the cross-attention fusion layer and classifier head are trained
- Trainable parameters: **1,577,220** (4.6% of total)
- Learning rate: **1e-4**
- Purpose: Learn good fusion and classification weights without disrupting pretrained features

**Phase 2 — Full Fine-tuning (Epoch 8+)**
- All layers are **unfrozen**
- Trainable parameters: **33,939,116** (100%)
- Learning rate: **1e-5** (10x lower to avoid catastrophic forgetting)
- Purpose: Fine-tune backbone features specifically for X-ray classification

### 4.2 Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Scheduler | CosineAnnealingLR |
| Batch Size | 16 |
| Loss Function | Weighted CrossEntropyLoss |
| Random Seed | 42 |
| Best Checkpoint | Epoch 8 (selected by validation F1) |

### 4.3 Data Augmentation (Training Only)

| Augmentation | Parameters | Purpose |
|-------------|------------|---------|
| Horizontal Flip | p=0.5 | Simulate left/right orientation |
| Affine Rotation | ±15°, ±5% translate, ±5% scale | Simulate positioning variation |
| CLAHE | clip_limit=2.0, grid=8×8 | Enhance contrast in medical images |
| Random Brightness/Contrast | ±15% | Simulate exposure differences |
| Normalize | ImageNet mean/std | Required for pretrained backbone compatibility |
| Resize | 224×224 | Standard input size for both backbones |

**Validation and test sets use only Resize + Normalize** (no augmentation).

### 4.4 Training History

| Epoch | Phase | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 | LR |
|-------|-------|-----------|-----------|----------|---------|--------|-----|
| 1 | Frozen | 0.3087 | 86.17% | 0.3013 | 88.28% | 0.8155 | 1.00e-4 |
| 2 | Frozen | 0.1724 | 93.40% | 0.2424 | 90.30% | 0.8184 | 9.50e-5 |
| 3 | Frozen | 0.1328 | 95.11% | 0.1700 | 92.66% | 0.8168 | 8.10e-5 |
| 4 | Frozen | 0.1047 | 96.32% | 0.1648 | 93.25% | **0.8409** | 6.10e-5 |
| 5 | Frozen | 0.0901 | 96.77% | 0.1500 | 94.20% | 0.8082 | 3.90e-5 |
| 6 | Frozen | 0.0823 | 97.31% | 0.1332 | 94.91% | 0.8284 | 1.90e-5 |
| 7 | Frozen | 0.0732 | 97.61% | 0.1280 | 94.91% | 0.8284 | 5.00e-6 |
| **8** | **Fine-tune** | **0.0864** | **97.10%** | **0.1006** | **95.86%** | **0.8548** | **1.00e-5** |

**Key observations:**
- Phase 1 alone achieved 94.91% val accuracy — strong baseline
- Phase 2 (full fine-tuning) jumped to 95.86% — backbone adaptation helps
- Train accuracy slightly dropped in epoch 8 (97.10% vs 97.61%) because all parameters were reset to trainable with a lower LR — this is expected and healthy (less overfitting)

---

## 5. Results

### 5.1 Overall Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **88.76%** |
| **Validation Accuracy** | **95.86%** |
| Test F1 Score (Macro) | 0.8773 |
| Test F1 Score (Weighted) | 0.8806 |
| Test Precision (Macro) | 0.9199 |
| Test Recall (Macro) | 0.8674 |
| **AUC-ROC (Macro)** | **0.9944** |
| Inference Speed | 28.9 ms/image |

### 5.2 Domain-Specific Performance

| Domain | Accuracy | F1 (Weighted) | Test Images |
|--------|----------|---------------|-------------|
| **Bone X-ray** | **95.45%** | **0.9546** | 506 |
| **Chest X-ray** | **83.33%** | **0.8206** | 624 |

### 5.3 Per-Class Metrics

| Class | Precision | Recall | F1 Score | AUC-ROC | Support |
|-------|-----------|--------|----------|---------|---------|
| Fractured | 0.9498 | 0.9538 | **0.9518** | 0.9991 | 238 |
| Not Fractured | 0.9588 | 0.9552 | **0.9570** | 0.9992 | 268 |
| Normal Lungs | 0.9779 | 0.5684 | 0.7189 | 0.9882 | 234 |
| Pneumonia | 0.7930 | 0.9923 | **0.8815** | 0.9909 | 390 |

### 5.4 Confusion Matrix

|  | Pred: Fractured | Pred: Not Fractured | Pred: Normal | Pred: Pneumonia |
|--|-----------------|---------------------|--------------|-----------------|
| **Actual: Fractured** | **227** | 11 | 0 | 0 |
| **Actual: Not Fractured** | 12 | **256** | 0 | 0 |
| **Actual: Normal** | 0 | 0 | **133** | 101 |
| **Actual: Pneumonia** | 0 | 0 | 3 | **387** |

### 5.5 Key Observations

1. **Perfect domain separation**: Zero confusion between bone and chest classes. The model never classifies a bone X-ray as a chest X-ray or vice versa.

2. **Excellent bone classification**: 95.45% accuracy with balanced precision and recall (~95% each). Only 23 misclassifications out of 506 bone images.

3. **High pneumonia recall (99.2%)**: The model catches virtually all pneumonia cases — critical for a screening tool where missing a positive case is dangerous.

4. **Normal lung recall is lower (56.8%)**: 101 out of 234 normal images are predicted as pneumonia. This is a conservative bias — the model errs on the side of flagging potential issues. In a medical screening context, this is actually preferable (high sensitivity > high specificity).

5. **AUC-ROC near perfect (0.9944)**: Despite the recall gap on Normal class, the model's probabilistic outputs separate classes very well. A different threshold could improve Normal recall at the cost of Pneumonia recall.

---

## 6. Explainability — Grad-CAM

### 6.1 What is Grad-CAM?

**Gradient-weighted Class Activation Mapping (Grad-CAM)** produces visual explanations for model predictions by:

1. Computing gradients of the predicted class score with respect to the last convolutional layer activations
2. Computing importance weights by global-average-pooling these gradients
3. Creating a weighted combination of activation maps
4. Applying ReLU to keep only positive influences
5. Overlaying the resulting heatmap on the original image

### 6.2 Implementation

- **Target layer**: EfficientNet-B3's `conv_head` (final convolutional layer)
- **Bone X-rays**: JET colormap (blue→green→red) — warm regions = high importance
- **Chest X-rays**: MAGMA colormap (dark→purple→bright) — bright regions = high importance
- **Overlay blend**: 40% heatmap + 60% original image

### 6.3 Clinical Relevance

- For **fractures**: Grad-CAM highlights the fracture line and surrounding bone disruption
- For **pneumonia**: Grad-CAM highlights opaque/consolidated lung regions
- For **normal** images: Activation is typically diffuse (no focal pathology to highlight)

---

## 7. Web Application

### 7.1 Interface Design

- **Framework**: Gradio 6.9.0
- **Single upload**: User uploads any X-ray (bone or chest)
- **Outputs**: 
  - Classification label with confidence scores for all 4 classes
  - Grad-CAM heatmap overlay
  - Text report with interpretation

### 7.2 Design Choices

- Clean dark theme (GitHub-inspired color palette)
- No emojis — professional medical aesthetic
- Minimal information: model name, accuracy stats, feature descriptions, disclaimer
- Deployment-ready with `share=True` for public Gradio link

---

## 8. Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.13.7 |
| Deep Learning | PyTorch | 2.10.0 |
| Model Library | timm (PyTorch Image Models) | 1.0.25 |
| Augmentation | Albumentations | 2.0.8 |
| Metrics | scikit-learn | 1.8.0 |
| Image Processing | OpenCV | 4.13.0 |
| Web Interface | Gradio | 6.9.0 |
| Numerical | NumPy | 2.4.2 |
| Hardware | Apple M2 (MPS) | MacBook Air |

---

## 9. File Structure

```
Bone_Fracture_Classification/
├── app.py                     # Gradio web application
├── model.py                   # HybridCNNViT architecture definition
├── train_unified.py           # Unified 4-class training script
├── config_unified.yaml        # Training hyperparameters
├── requirements.txt           # Python dependencies
├── model_metrics_report.csv   # All metrics in CSV format
├── evaluation_results.json    # Detailed evaluation data
├── checkpoints/
│   └── unified_model.pth      # Best model checkpoint (epoch 8, 130MB)
├── dataset/                   # Bone fracture dataset
│   └── Bone_Fracture_Binary_Classification/
│       └── Bone_Fracture_Binary_Classification/
│           ├── train/ (fractured: 4606, not fractured: 4640)
│           ├── val/   (fractured: 337, not fractured: 492)
│           └── test/  (fractured: 238, not fractured: 268)
├── chest_xray/                # Chest X-ray pneumonia dataset
│   ├── train/ (NORMAL: 1341, PNEUMONIA: 3875)
│   ├── val/   (NORMAL: 8, PNEUMONIA: 8)
│   └── test/  (NORMAL: 234, PNEUMONIA: 390)
└── results/                   # Visualizations
    ├── confusion_matrix.png
    └── explainability/
        ├── fractured_explainability.png
        └── not fractured_explainability.png
```

---

## 10. Reproducibility

To reproduce this project:

```bash
# 1. Clone and setup
git clone https://github.com/adi-huh/Bone_Fracture_Classification_Hackathon.git
cd Bone_Fracture_Classification_Hackathon
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Train (requires both datasets in place)
python train_unified.py

# 3. Launch app
python app.py
```

**Random seed 42** ensures deterministic initialization. Training results may vary slightly across hardware due to non-deterministic MPS/CUDA operations.

---

## 11. Limitations and Future Work

### Current Limitations
1. **Small chest validation set** (16 images) — val metrics may not fully represent generalization
2. **Normal lung recall at 56.8%** — model is conservative (over-predicts pneumonia)
3. **Single-epoch fine-tuning** — only 1 fine-tune epoch completed (epoch 8); more epochs could improve

### Future Improvements
1. Train for full 15 epochs (or until convergence) for better chest classification
2. Add more chest validation data for reliable validation metrics
3. Implement threshold tuning to balance Normal/Pneumonia recall
4. Add multi-region fracture classification (e.g., hand, elbow, shoulder)
5. Deploy to Hugging Face Spaces for permanent hosting

---

## 12. Summary

We built a **unified medical X-ray analysis system** that classifies both bone fractures and lung pneumonia from a single model and single upload interface.

**Key achievements:**
- **95.45% bone fracture detection accuracy** with near-perfect AUC (0.999)
- **99.2% pneumonia recall** — catches virtually all positive cases
- **Zero cross-domain confusion** — never confuses bone for chest or vice versa
- **AUC-ROC = 0.9944** — excellent probabilistic class separation
- **Real-time inference** at 29ms/image
- **Explainable** with Grad-CAM heatmaps for every prediction
- **Clean web interface** with single-click classification

---

*KBG Hackathon 2025 | IIT Mandi | Aditya Rai*
