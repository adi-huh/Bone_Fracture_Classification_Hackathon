#!/usr/bin/env python3
"""
Generate presentation.pdf — 8-slide hackathon presentation.
KBG Hackathon 2025, IIT Mandi — Bone Fracture Classification
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import textwrap

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "presentation.pdf")
CM_PATH = os.path.join(ROOT, "results", "confusion_matrix.png")
GRADCAM_FRAC = os.path.join(ROOT, "results", "explainability", "fractured_gradcam.png")
GRADCAM_NOFRAC = os.path.join(ROOT, "results", "explainability", "not fractured_gradcam.png")
ATTN_FRAC = os.path.join(ROOT, "results", "explainability", "fractured_attention.png")
EXPLAIN_FRAC = os.path.join(ROOT, "results", "explainability", "fractured_explainability.png")
EXPLAIN_NOFRAC = os.path.join(ROOT, "results", "explainability", "not fractured_explainability.png")

# ── Colours ────────────────────────────────────────────────────────────────
BG       = "#0d1117"
CARD_BG  = "#161b22"
ACCENT   = "#58a6ff"
ACCENT2  = "#3fb950"
ACCENT3  = "#f78166"
TEXT     = "#e6edf3"
SUBTEXT  = "#8b949e"
WHITE    = "#ffffff"

# ── Helper ─────────────────────────────────────────────────────────────────

def new_slide(pdf, title, subtitle=None):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.set_facecolor(BG); ax.axis("off")
    # Top accent bar
    ax.add_patch(mpatches.FancyBboxPatch((0, 8.55), 16, 0.45, boxstyle="square", facecolor=ACCENT, edgecolor="none"))
    ax.text(0.5, 8.78, title, fontsize=26, fontweight="bold", color=WHITE, va="center", fontfamily="sans-serif")
    if subtitle:
        ax.text(15.5, 8.78, subtitle, fontsize=12, color=SUBTEXT, va="center", ha="right", fontfamily="sans-serif")
    return fig, ax

def add_card(ax, x, y, w, h, text_lines, title=None, title_color=ACCENT):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                          facecolor=CARD_BG, edgecolor="#30363d", linewidth=1.2))
    ty = y + h - 0.35
    if title:
        ax.text(x + 0.25, ty, title, fontsize=14, fontweight="bold", color=title_color, va="top", fontfamily="sans-serif")
        ty -= 0.45
    for line in text_lines:
        ax.text(x + 0.25, ty, line, fontsize=11, color=TEXT, va="top", fontfamily="sans-serif")
        ty -= 0.35
    return ty

def bullet(ax, x, y, items, fontsize=11, color=TEXT, spacing=0.38, bullet_char="•"):
    for item in items:
        ax.text(x, y, f"{bullet_char} {item}", fontsize=fontsize, color=color, va="top", fontfamily="sans-serif")
        y -= spacing
    return y


# ══════════════════════════════════════════════════════════════════════════
#  SLIDE 1 — Title
# ══════════════════════════════════════════════════════════════════════════
def slide_1(pdf):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.set_facecolor(BG); ax.axis("off")

    # Accent bar at bottom
    ax.add_patch(mpatches.FancyBboxPatch((0, 0), 16, 0.15, boxstyle="square", facecolor=ACCENT, edgecolor="none"))

    # Emoji / icon
    ax.text(8, 7.2, "⚕", fontsize=64, ha="center", va="center", fontfamily="sans-serif")

    # Title
    ax.text(8, 5.9, "Bone Fracture Classification", fontsize=36, fontweight="bold",
            color=WHITE, ha="center", va="center", fontfamily="sans-serif")
    ax.text(8, 5.2, "Hybrid CNN + Vision Transformer", fontsize=22, color=ACCENT,
            ha="center", va="center", fontfamily="sans-serif")

    # Separator
    ax.plot([5, 11], [4.6, 4.6], color=ACCENT, linewidth=2.5)

    # Hackathon info
    ax.text(8, 4.0, "KBG Hackathon 2025  •  IIT Mandi", fontsize=16, color=SUBTEXT,
            ha="center", va="center", fontfamily="sans-serif")
    ax.text(8, 3.4, "Kamand Bioengineering Group", fontsize=14, color=SUBTEXT,
            ha="center", va="center", fontfamily="sans-serif")

    # Problem statement box
    add_card(ax, 3.5, 1.0, 9, 1.8, [
        "Develop an AI system for binary bone fracture classification",
        "from X-ray radiographs to assist radiologists in faster,",
        "more accurate fracture identification."
    ], title="Problem Statement", title_color=ACCENT3)

    # Team
    ax.text(8, 0.5, "Team: Aditya Rai  |  Domain: Medical Image Analysis", fontsize=12,
            color=SUBTEXT, ha="center", va="center", fontfamily="sans-serif")

    pdf.savefig(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  SLIDE 2 — Literature Review & Approach Justification
# ══════════════════════════════════════════════════════════════════════════
def slide_2(pdf):
    fig, ax = new_slide(pdf, "Literature Review & Approach Justification", "Slide 2/8")

    # Left card: Literature
    add_card(ax, 0.3, 4.3, 7.4, 4.0, [], title="Literature Review", title_color=ACCENT)
    bullet(ax, 0.6, 7.5, [
        "CNNs (ResNet, EfficientNet) dominate fracture detection",
        "  → strong local feature extraction from texture/edges",
        "ViTs capture long-range spatial dependencies via attention",
        "  → whole-bone context for subtle / hairline fractures",
        "Hybrid CNN+ViT outperforms single-backbone approaches",
        "  → e.g., TransMed, MedViT show complementary strengths",
        "Cross-attention fusion > simple concatenation",
        "  → allows bidirectional feature interaction",
    ], fontsize=10.5, spacing=0.40)

    # Right card: Our Approach
    add_card(ax, 8.0, 4.3, 7.7, 4.0, [], title="Our Approach", title_color=ACCENT2)
    bullet(ax, 8.3, 7.5, [
        "EfficientNet-B3 (CNN): efficient local features (d=1536)",
        "ViT-Small-Patch16-224: global self-attention (d=384)",
        "Cross-Attention Fusion into shared d=512 space",
        "2-Phase Transfer Learning:",
        "  Phase 1 → frozen backbones (learn fusion head)",
        "  Phase 2 → full fine-tuning with low LR",
        "Only ImageNet pre-trained weights (no medical pretrain)",
        "Designed for clinical decision support",
    ], fontsize=10.5, spacing=0.40)

    # Bottom card: Why this combination?
    add_card(ax, 0.3, 0.3, 15.4, 3.7, [], title="Why Hybrid CNN + ViT?", title_color=ACCENT3)

    ax.text(1.0, 3.2, "CNN  ", fontsize=14, fontweight="bold", color=ACCENT, va="top", fontfamily="sans-serif")
    ax.text(2.5, 3.2, "Texture, edges, bone cortex patterns → local features", fontsize=11, color=TEXT, va="top")

    ax.text(1.0, 2.6, "ViT  ", fontsize=14, fontweight="bold", color=ACCENT2, va="top", fontfamily="sans-serif")
    ax.text(2.5, 2.6, "Whole-bone geometry, fracture line continuity → global context", fontsize=11, color=TEXT, va="top")

    ax.text(1.0, 2.0, "Fusion", fontsize=14, fontweight="bold", color=ACCENT3, va="top", fontfamily="sans-serif")
    ax.text(2.5, 2.0, "Cross-attention lets CNN & ViT features attend to each other → richer representation", fontsize=11, color=TEXT, va="top")

    ax.text(1.0, 1.2, "Result", fontsize=14, fontweight="bold", color=WHITE, va="top", fontfamily="sans-serif")
    ax.text(2.5, 1.2, "98.22% test accuracy  |  0.9822 Macro F1  |  0.9970 AUC-ROC", fontsize=13, fontweight="bold", color=ACCENT2, va="top")

    pdf.savefig(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  SLIDE 3 — Data Preprocessing & Strategy
# ══════════════════════════════════════════════════════════════════════════
def slide_3(pdf):
    fig, ax = new_slide(pdf, "Data Preprocessing & Strategy", "Slide 3/8")

    # Dataset info
    add_card(ax, 0.3, 5.0, 7.4, 3.3, [], title="Dataset Overview", title_color=ACCENT)
    bullet(ax, 0.6, 7.5, [
        "Bone Fracture Binary Classification (Kaggle)",
        "Total images: 10,581",
        "  → Train: 9,246  |  Val: 829  |  Test: 506",
        "Classes: Fractured / Not Fractured",
        "Format: PNG/JPEG, variable resolution",
        "Pre-split into train/val/test folders",
    ], fontsize=10.5, spacing=0.38)

    # Preprocessing
    add_card(ax, 8.0, 5.0, 7.7, 3.3, [], title="Preprocessing Pipeline", title_color=ACCENT2)
    bullet(ax, 8.3, 7.5, [
        "Resize to 224×224 pixels",
        "ImageNet normalization (μ, σ per channel)",
        "Pixel values → [0, 1] → normalized",
        "Consistent 3-channel RGB conversion",
        "Reproducible with seed=42",
        "num_workers=2, pin_memory on CUDA",
    ], fontsize=10.5, spacing=0.38)

    # Augmentation
    add_card(ax, 0.3, 0.3, 15.4, 4.4, [], title="Data Augmentation Strategy (Training Only)", title_color=ACCENT3)

    augmentations = [
        ("Horizontal Flip", "p=0.5", "Simulate L/R orientation differences"),
        ("Affine Rotation", "±15°", "Account for positioning variations"),
        ("Brightness/Contrast", "±20%", "Handle exposure differences across X-ray machines"),
        ("CLAHE", "clip=2.0", "Enhance local contrast in bone structures"),
        ("Normalization", "ImageNet μ,σ", "Standardize for pretrained backbone input"),
    ]
    headers_y = 4.0
    ax.text(1.0, headers_y, "Augmentation", fontsize=12, fontweight="bold", color=ACCENT, va="top")
    ax.text(5.5, headers_y, "Parameters", fontsize=12, fontweight="bold", color=ACCENT, va="top")
    ax.text(9.0, headers_y, "Rationale", fontsize=12, fontweight="bold", color=ACCENT, va="top")
    ax.plot([0.8, 15.2], [headers_y - 0.2, headers_y - 0.2], color="#30363d", linewidth=0.8)

    ty = headers_y - 0.45
    for name, param, reason in augmentations:
        ax.text(1.0, ty, name, fontsize=11, color=TEXT, va="top")
        ax.text(5.5, ty, param, fontsize=11, color=SUBTEXT, va="top")
        ax.text(9.0, ty, reason, fontsize=11, color=TEXT, va="top")
        ty -= 0.38

    ax.text(1.0, ty - 0.2, "Validation & Test: only resize + normalize (no augmentation)", fontsize=11,
            color=ACCENT3, va="top", style="italic")

    pdf.savefig(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  SLIDE 4 — Model Architecture
# ══════════════════════════════════════════════════════════════════════════
def slide_4(pdf):
    fig, ax = new_slide(pdf, "Model Architecture — HybridCNNViT", "Slide 4/8")

    # Architecture diagram (hand-drawn style with boxes and arrows)
    def draw_box(ax, cx, cy, w, h, label, color, sublabel=None):
        ax.add_patch(mpatches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                     boxstyle="round,pad=0.1", facecolor=color, edgecolor=WHITE, linewidth=1.5, alpha=0.85))
        ax.text(cx, cy + 0.1 if sublabel else cy, label, fontsize=11, fontweight="bold",
                color=WHITE, ha="center", va="center", fontfamily="sans-serif")
        if sublabel:
            ax.text(cx, cy - 0.25, sublabel, fontsize=9, color="#ccc", ha="center", va="center")

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color=SUBTEXT, lw=1.5))

    # Input
    draw_box(ax, 8, 7.5, 3.5, 0.8, "Input (B, 3, 224, 224)", "#30363d")

    # Two branches
    arrow(ax, 6.5, 7.1, 4.5, 6.5)
    arrow(ax, 9.5, 7.1, 11.5, 6.5)

    # CNN Branch
    draw_box(ax, 4.5, 6.0, 4.0, 0.9, "EfficientNet-B3", "#1f6feb", "CNN backbone (d=1536)")
    arrow(ax, 4.5, 5.55, 4.5, 5.0)
    draw_box(ax, 4.5, 4.5, 3.5, 0.8, "AdaptiveAvgPool → Proj", "#1f6feb")

    # ViT Branch
    draw_box(ax, 11.5, 6.0, 4.0, 0.9, "ViT-Small-Patch16", "#238636", "ViT backbone (d=384)")
    arrow(ax, 11.5, 5.55, 11.5, 5.0)
    draw_box(ax, 11.5, 4.5, 3.5, 0.8, "CLS token → Proj", "#238636")

    # Arrows to fusion
    arrow(ax, 4.5, 4.1, 7.0, 3.4)
    arrow(ax, 11.5, 4.1, 9.0, 3.4)

    # Fusion
    draw_box(ax, 8, 3.0, 5.0, 0.9, "Cross-Attention Fusion", ACCENT3, "MultiheadAttention (d=512, 4 heads)")
    arrow(ax, 8, 2.55, 8, 2.0)

    # Classifier
    draw_box(ax, 8, 1.6, 5.5, 0.7, "Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→2)", "#8b5cf6")
    arrow(ax, 8, 1.25, 8, 0.8)

    # Output
    draw_box(ax, 8, 0.5, 3.0, 0.6, "Fractured / Not Fractured", ACCENT2)

    # Stats on the right side
    ax.text(0.4, 2.3, "Key Design Choices:", fontsize=11, fontweight="bold", color=ACCENT, va="top")
    stats = [
        "✦ Cross-attention > concat fusion",
        "✦ EfficientNet-B3 balances FLOPs & accuracy",
        "✦ ViT-Small: 22M params, 16×16 patches",
        "✦ Total: 33.9M params (unfrozen)",
        "✦ Phase 1 frozen: only 1.58M trainable",
    ]
    ty = 1.9
    for s in stats:
        ax.text(0.4, ty, s, fontsize=9, color=TEXT, va="top", fontfamily="sans-serif")
        ty -= 0.30

    pdf.savefig(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  SLIDE 5 — Training Details
# ══════════════════════════════════════════════════════════════════════════
def slide_5(pdf):
    fig, ax = new_slide(pdf, "Training Details", "Slide 5/8")

    # Phase 1
    add_card(ax, 0.3, 4.8, 7.4, 3.5, [], title="Phase 1 — Frozen Backbones", title_color=ACCENT)
    bullet(ax, 0.6, 7.5, [
        "Epochs: 10 (of total 50 budget)",
        "Trainable params: 1.58M (fusion + classifier)",
        "Learning rate: 1e-4 (AdamW)",
        "Scheduler: CosineAnnealingLR",
        "Purpose: learn good fusion weights first",
        "Converged to 97.95% val accuracy",
    ], fontsize=10.5, spacing=0.38)

    # Phase 2
    add_card(ax, 8.0, 4.8, 7.7, 3.5, [], title="Phase 2 — Full Fine-tuning", title_color=ACCENT2)
    bullet(ax, 8.3, 7.5, [
        "Epochs: 11–15 (early stopped at epoch 15)",
        "Trainable params: 33.9M (all layers)",
        "Learning rate: 1e-5 (10× lower)",
        "Scheduler: CosineAnnealingLR (reset)",
        "Early stopping: patience=10 on val F1",
        "Best: epoch 13 → 99.16% val acc, 0.9913 F1",
    ], fontsize=10.5, spacing=0.38)

    # Training config table
    add_card(ax, 0.3, 0.3, 15.4, 4.2, [], title="Hyperparameters & Configuration", title_color=ACCENT3)

    configs = [
        ("Loss Function", "CrossEntropyLoss"),
        ("Optimizer", "AdamW (weight_decay=1e-2)"),
        ("Batch Size", "16"),
        ("Image Size", "224 × 224"),
        ("Augmentation", "HFlip, Rotation ±15°, Brightness, CLAHE"),
        ("Device", "Apple M2 (MPS backend)"),
        ("Training Time", "~45 minutes total"),
        ("Seed", "42 (fully reproducible)"),
    ]

    # Two columns
    left_configs = configs[:4]
    right_configs = configs[4:]

    ty = 3.7
    for k, v in left_configs:
        ax.text(0.7, ty, k + ":", fontsize=10.5, fontweight="bold", color=ACCENT, va="top")
        ax.text(3.8, ty, v, fontsize=10.5, color=TEXT, va="top")
        ty -= 0.38

    ty = 3.7
    for k, v in right_configs:
        ax.text(8.3, ty, k + ":", fontsize=10.5, fontweight="bold", color=ACCENT, va="top")
        ax.text(11.5, ty, v, fontsize=10.5, color=TEXT, va="top")
        ty -= 0.38

    # Training curve mini-chart
    epochs_p1 = list(range(1, 11))
    val_acc_p1 = [91.44, 93.97, 94.57, 96.02, 96.26, 96.62, 97.47, 97.83, 98.07, 97.95]
    train_acc_p1 = [87.09, 95.94, 97.30, 98.26, 98.59, 98.88, 99.03, 99.23, 99.33, 99.53]
    epochs_p2 = list(range(11, 16))
    val_acc_p2 = [98.31, 98.31, 99.16, 98.67, 98.91]
    train_acc_p2 = [98.93, 99.56, 99.74, 99.82, 99.82]

    # Tiny chart inset
    ax2 = fig.add_axes([0.57, 0.08, 0.24, 0.35])
    ax2.set_facecolor(CARD_BG)
    ax2.plot(epochs_p1, train_acc_p1, '-o', color=ACCENT, markersize=3, label='Train Acc', linewidth=1.5)
    ax2.plot(epochs_p1, val_acc_p1, '-s', color=ACCENT2, markersize=3, label='Val Acc', linewidth=1.5)
    ax2.plot(epochs_p2, train_acc_p2, '-o', color=ACCENT, markersize=3, linewidth=1.5)
    ax2.plot(epochs_p2, val_acc_p2, '-s', color=ACCENT2, markersize=3, linewidth=1.5)
    ax2.axvline(x=10.5, color=ACCENT3, linestyle='--', linewidth=1, alpha=0.7)
    ax2.text(10.5, 88, "Phase\n  2→", fontsize=7, color=ACCENT3, ha="center")
    ax2.set_xlabel("Epoch", fontsize=8, color=SUBTEXT)
    ax2.set_ylabel("Accuracy (%)", fontsize=8, color=SUBTEXT)
    ax2.set_title("Training Curves", fontsize=9, color=TEXT, fontweight="bold")
    ax2.tick_params(labelsize=7, colors=SUBTEXT)
    ax2.legend(fontsize=7, loc="lower right")
    ax2.set_ylim(85, 100.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for spine in ax2.spines.values():
        spine.set_color("#30363d")
    ax2.set_facecolor(CARD_BG)

    pdf.savefig(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  SLIDE 6 — Results & Performance Metrics
# ══════════════════════════════════════════════════════════════════════════
def slide_6(pdf):
    fig, ax = new_slide(pdf, "Results & Performance Metrics", "Slide 6/8")

    # Main metrics cards
    metrics_data = [
        ("Test Accuracy", "98.22%", ACCENT),
        ("Macro F1-Score", "0.9822", ACCENT2),
        ("AUC-ROC", "0.9970", ACCENT3),
    ]
    for i, (label, value, color) in enumerate(metrics_data):
        cx = 1.5 + i * 5.0
        ax.add_patch(mpatches.FancyBboxPatch((cx, 6.3), 4.2, 2.0, boxstyle="round,pad=0.15",
                                              facecolor=CARD_BG, edgecolor=color, linewidth=2.5))
        ax.text(cx + 2.1, 7.7, value, fontsize=28, fontweight="bold", color=color,
                ha="center", va="center", fontfamily="sans-serif")
        ax.text(cx + 2.1, 6.8, label, fontsize=13, color=SUBTEXT,
                ha="center", va="center", fontfamily="sans-serif")

    # Per-class metrics table
    add_card(ax, 0.3, 2.7, 8.2, 3.3, [], title="Per-Class Metrics", title_color=ACCENT)

    # Table header
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    hx = [0.7, 2.7, 4.2, 5.5, 7.0]
    hy = 5.2
    for h, x in zip(headers, hx):
        ax.text(x, hy, h, fontsize=10.5, fontweight="bold", color=ACCENT, va="top")
    ax.plot([0.5, 8.2], [hy - 0.2, hy - 0.2], color="#30363d", linewidth=0.8)

    # Rows
    rows = [
        ["Fractured",    "0.9791", "0.9832", "0.9811", "238"],
        ["Not Fractured","0.9850", "0.9813", "0.9832", "268"],
    ]
    ty = hy - 0.45
    for row in rows:
        for val, x in zip(row, hx):
            ax.text(x, ty, val, fontsize=10.5, color=TEXT, va="top")
        ty -= 0.38

    # Macro row
    ax.plot([0.5, 8.2], [ty + 0.1, ty + 0.1], color="#30363d", linewidth=0.8)
    ax.text(0.7, ty - 0.1, "Macro Avg", fontsize=10.5, fontweight="bold", color=ACCENT2, va="top")
    ax.text(2.7, ty - 0.1, "0.9820", fontsize=10.5, color=TEXT, va="top")
    ax.text(4.2, ty - 0.1, "0.9823", fontsize=10.5, color=TEXT, va="top")
    ax.text(5.5, ty - 0.1, "0.9822", fontsize=10.5, color=TEXT, va="top")
    ax.text(7.0, ty - 0.1, "506", fontsize=10.5, color=TEXT, va="top")

    # Confusion matrix image
    if os.path.exists(CM_PATH):
        ax_cm = fig.add_axes([0.56, 0.15, 0.40, 0.38])
        img = mpimg.imread(CM_PATH)
        ax_cm.imshow(img)
        ax_cm.axis("off")
        ax_cm.set_title("Confusion Matrix", fontsize=11, color=TEXT, fontweight="bold", pad=5)
    else:
        add_card(ax, 8.8, 2.7, 6.9, 3.3, [
            "Confusion Matrix:",
            "",
            "              Pred Frac    Pred Not Frac",
            "True Frac        234            4",
            "True Not Frac      5          263",
            "",
            "Only 9 misclassifications out of 506!",
        ], title="Confusion Matrix", title_color=ACCENT2)

    # Bottom info
    add_card(ax, 0.3, 0.3, 15.4, 2.1, [], title="Additional Metrics", title_color=ACCENT3)
    extra = [
        ("Training Time", "~45 min (Apple M2 MPS)"),
        ("Inference Speed", "5.67 ms/image"),
        ("Model Size", "130 MB"),
        ("Best Val F1", "0.9913 (epoch 13)"),
    ]
    ex = 0.7
    for k, v in extra:
        ax.text(ex, 1.5, k, fontsize=10, fontweight="bold", color=ACCENT, va="top")
        ax.text(ex, 1.1, v, fontsize=10.5, color=TEXT, va="top")
        ex += 3.8

    pdf.savefig(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  SLIDE 7 — Explainability
# ══════════════════════════════════════════════════════════════════════════
def slide_7(pdf):
    fig, ax = new_slide(pdf, "Explainability — Grad-CAM & Attention Maps", "Slide 7/8")

    ax.text(0.5, 7.9, "CNN Grad-CAM highlights fracture regions  |  ViT Attention shows global focus patterns",
            fontsize=12, color=SUBTEXT, va="top", style="italic")

    # Load and display explainability images
    images_info = [
        (EXPLAIN_FRAC, "Fractured — Full Analysis"),
        (EXPLAIN_NOFRAC, "Not Fractured — Full Analysis"),
    ]

    available_imgs = [(p, t) for p, t in images_info if os.path.exists(p)]

    if len(available_imgs) >= 2:
        # Two large explainability panels
        ax1 = fig.add_axes([0.03, 0.10, 0.46, 0.72])
        img1 = mpimg.imread(available_imgs[0][0])
        ax1.imshow(img1)
        ax1.axis("off")
        ax1.set_title(available_imgs[0][1], fontsize=12, color=TEXT, fontweight="bold", pad=8)

        ax2 = fig.add_axes([0.52, 0.10, 0.46, 0.72])
        img2 = mpimg.imread(available_imgs[1][0])
        ax2.imshow(img2)
        ax2.axis("off")
        ax2.set_title(available_imgs[1][1], fontsize=12, color=TEXT, fontweight="bold", pad=8)
    elif len(available_imgs) == 1:
        ax1 = fig.add_axes([0.15, 0.10, 0.70, 0.72])
        img1 = mpimg.imread(available_imgs[0][0])
        ax1.imshow(img1)
        ax1.axis("off")
        ax1.set_title(available_imgs[0][1], fontsize=12, color=TEXT, fontweight="bold", pad=8)
    else:
        # No images available — describe techniques
        add_card(ax, 0.5, 1.0, 15, 6.5, [], title="Explainability Techniques Applied")
        bullet(ax, 1.0, 6.5, [
            "Grad-CAM: Gradient-weighted Class Activation Mapping",
            "  → Highlights CNN regions most relevant to classification",
            "  → Shows fracture line localization capability",
            "",
            "Attention Rollout: ViT attention layer aggregation",
            "  → Shows which image patches the transformer attends to",
            "  → Reveals global spatial reasoning patterns",
            "",
            "Both techniques confirm the model focuses on",
            "clinically relevant bone regions for its decisions.",
        ], fontsize=12, spacing=0.45)

    pdf.savefig(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  SLIDE 8 — Challenges, Limitations, Future Improvements
# ══════════════════════════════════════════════════════════════════════════
def slide_8(pdf):
    fig, ax = new_slide(pdf, "Challenges, Limitations & Future Work", "Slide 8/8")

    # Challenges
    add_card(ax, 0.3, 5.3, 7.4, 3.0, [], title="Challenges Faced", title_color=ACCENT3)
    bullet(ax, 0.6, 7.5, [
        "Large model (33.9M params) on limited hardware",
        "MPS backend lacks full CUDA feature parity",
        "Some corrupted JPEG images in dataset",
        "Balancing CNN and ViT learning rates",
        "Cross-attention fusion hyperparameter tuning",
        "2-phase training required careful LR scheduling",
    ], fontsize=10.5, spacing=0.36)

    # Limitations
    add_card(ax, 8.0, 5.3, 7.7, 3.0, [], title="Current Limitations", title_color=ACCENT)
    bullet(ax, 8.3, 7.5, [
        "Binary classification only (fracture / no fracture)",
        "No fracture type sub-classification",
        "Single anatomical region generalization unknown",
        "Not validated on clinical-grade DICOM images",
        "Limited external dataset validation",
        "No uncertainty quantification",
    ], fontsize=10.5, spacing=0.36)

    # Future work
    add_card(ax, 0.3, 0.3, 15.4, 4.7, [], title="Future Improvements", title_color=ACCENT2)

    future_items = [
        ("Multi-class Extension", "Expand to 7+ fracture types (simple, comminuted, spiral, stress, etc.)"),
        ("Uncertainty Estimation", "Monte Carlo Dropout or ensemble for confidence scores"),
        ("Multi-Region Support", "Train on diverse anatomical regions (hand, wrist, hip, spine)"),
        ("DICOM Integration", "Direct DICOM reading with windowing for clinical deployment"),
        ("Model Compression", "Knowledge distillation → lighter model for edge devices"),
        ("Vision Mamba", "Explore state-space models (Vim) for linear-complexity alternatives"),
        ("Clinical Validation", "IRB-approved prospective study with radiologist comparison"),
    ]

    ty = 4.2
    for title, desc in future_items:
        ax.text(0.7, ty, f"→ {title}:", fontsize=10.5, fontweight="bold", color=ACCENT2, va="top")
        ax.text(5.2, ty, desc, fontsize=10.5, color=TEXT, va="top")
        ty -= 0.45

    # Thank you note
    ax.text(8, 0.05, "Thank you!  Questions?", fontsize=14, fontweight="bold",
            color=ACCENT, ha="center", va="bottom", style="italic")

    pdf.savefig(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    with PdfPages(OUT) as pdf:
        slide_1(pdf)
        slide_2(pdf)
        slide_3(pdf)
        slide_4(pdf)
        slide_5(pdf)
        slide_6(pdf)
        slide_7(pdf)
        slide_8(pdf)
    print(f"✅  Presentation saved to: {OUT}")
    print(f"    8 slides, ready for submission.")

if __name__ == "__main__":
    main()
