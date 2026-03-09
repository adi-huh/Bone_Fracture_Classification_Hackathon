#!/usr/bin/env python3
"""
app.py — Gradio Web App for Bone Fracture Classification
=========================================================
KBG Hackathon 2025 — IIT Mandi
Hybrid CNN + Vision Transformer with Grad-CAM Explainability

Deploy to Hugging Face Spaces:
    gradio deploy
"""

import os
import numpy as np
import cv2
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import build_model
from data_loader import load_config

# ── Paths ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")
CHECKPOINT = os.path.join(ROOT, "checkpoints", "model.pth")
CM_IMG = os.path.join(ROOT, "results", "confusion_matrix.png")
EXPLAIN_FRAC = os.path.join(ROOT, "results", "explainability", "fractured_explainability.png")
EXPLAIN_NOFRAC = os.path.join(ROOT, "results", "explainability", "not fractured_explainability.png")


# ── Device ─────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model loader (cached) ─────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def load_model():
    device = get_device()
    config = load_config(CONFIG_PATH)

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

    # Infer classes from checkpoint
    if "class_names" in ckpt:
        class_names = ckpt["class_names"]
    else:
        last_w = ckpt["model_state_dict"]["classifier.3.weight"]
        n = last_w.shape[0]
        class_names = ["fractured", "not fractured"] if n == 2 else [f"class_{i}" for i in range(n)]

    config["num_classes"] = len(class_names)

    mdl = build_model(config)
    mdl.load_state_dict(ckpt["model_state_dict"])
    mdl.to(device)
    mdl.eval()

    print(f"[app] Model loaded | Device: {device} | Classes: {class_names}")
    return mdl, device, class_names, config


# ── Preprocessing ──────────────────────────────────────────────────
def get_transform(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ── Grad-CAM ──────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None

        target = self._find_target_layer()
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def _find_target_layer(self):
        if hasattr(self.model.cnn, "conv_head"):
            return self.model.cnn.conv_head
        if hasattr(self.model.cnn, "layer4"):
            return self.model.cnn.layer4[-1]
        last_conv = None
        for _, m in self.model.cnn.named_modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv:
            return last_conv
        raise RuntimeError("No conv layer found for Grad-CAM")

    def generate(self, tensor, cls_idx=None):
        self.model.zero_grad()
        out = self.model(tensor)
        if cls_idx is None:
            cls_idx = out.argmax(dim=1).item()
        out[0, cls_idx].backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return None

        w = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((w * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().detach().numpy()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


# ── Prediction ─────────────────────────────────────────────────────
def predict(image):
    if image is None:
        return {}, None, ""

    model, device, class_names, config = load_model()
    grad_cam = GradCAM(model, device)
    img_size = config.get("image_size", 224)
    transform = get_transform(img_size)

    img_np = np.array(image)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    orig_h, orig_w = img_np.shape[:2]
    aug = transform(image=img_np)
    tensor = aug["image"].unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    with torch.enable_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        cam = grad_cam.generate(tensor, cls_idx=pred_idx)

    # Confidences
    confidences = {class_names[i]: float(probs[0, i].item()) for i in range(len(class_names))}

    # Grad-CAM overlay
    if cam is not None:
        cam_r = cv2.resize(cam, (orig_w, orig_h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.clip(np.float32(heatmap) * 0.4 + np.float32(img_np) * 0.6, 0, 255).astype(np.uint8)
        overlay_img = Image.fromarray(overlay)
    else:
        overlay_img = image

    # Report
    pred_class = class_names[pred_idx].upper()
    conf_pct = probs[0, pred_idx].item() * 100
    report = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BONE FRACTURE ANALYSIS REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Classification:  {pred_class}
  Confidence:      {conf_pct:.1f}%

  Per-Class Probabilities:
"""
    for i, cn in enumerate(class_names):
        p = probs[0, i].item() * 100
        bar = "█" * int(p / 5) + "░" * (20 - int(p / 5))
        report += f"    {cn:>15s}:  {bar} {p:.1f}%\n"

    report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INTERPRETATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    if pred_idx == 0:  # fractured
        if conf_pct > 90:
            report += "  ⚠️  HIGH CONFIDENCE: Fracture indicators detected.\n  The Grad-CAM heatmap highlights the suspected fracture region."
        else:
            report += "  ⚠️  MODERATE CONFIDENCE: Possible fracture indicators.\n  Review the Grad-CAM heatmap for highlighted regions."
    else:
        if conf_pct > 90:
            report += "  ✅  HIGH CONFIDENCE: No fracture indicators detected.\n  The bone structure appears intact."
        else:
            report += "  ✅  MODERATE CONFIDENCE: Likely no fracture.\n  Consider clinical correlation if symptoms persist."

    report += "\n\n  ⚕️  This is a research tool. Always consult a qualified\n     radiologist for clinical decisions."

    return confidences, overlay_img, report


# ══════════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════════

custom_css = """
/* ── FORCE DARK THEME ─────────────────────────────────────────── */
body, .gradio-container, .main, .app, .contain,
.gradio-container > div, .gradio-container > div > div {
    background: #0d1117 !important;
    color: #e6edf3 !important;
}
.gradio-container *, .gradio-container p, .gradio-container span,
.gradio-container li, .gradio-container td, .gradio-container th,
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container h4, .gradio-container b, .gradio-container strong,
.gradio-container label, .gradio-container div, .gradio-container a,
.gradio-container ul, .gradio-container ol, .gradio-container pre,
.gradio-container input, .gradio-container textarea {
    color: #e6edf3 !important;
}
/* Blocks and panels */
.gradio-container .block,
.gradio-container .tabitem,
.gradio-container .panel,
.gradio-container .form,
.gradio-container .wrap {
    background: #161b22 !important;
    border-color: #30363d !important;
}
/* ── Header banner ──────────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #0f3460 50%, #16213e 100%) !important;
    padding: 30px 20px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 15px;
}
.main-header * { color: #ffffff !important; }
/* ── Section card ──────────────────────────────────────────────── */
.section-card {
    background: #161b22 !important;
    padding: 18px 20px;
    border-radius: 10px;
    border-left: 5px solid #58a6ff;
    margin: 12px 0;
}
.section-card * { color: #e6edf3 !important; }
/* ── Metric grid ───────────────────────────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
    margin: 15px 0;
}
.metric-card {
    background: linear-gradient(135deg, #161b22, #1c2333) !important;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    border: 1px solid #30363d;
}
.metric-card .value {
    font-size: 2em;
    font-weight: 800;
    color: #58a6ff !important;
    display: block;
    margin: 8px 0;
}
.metric-card .label {
    font-size: 0.9em;
    color: #8b949e !important;
    font-weight: 600;
}
/* ── Arch box (dark code block) ────────────────────────────────── */
.arch-box {
    background: #0d1117 !important;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #30363d;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 13px;
    line-height: 1.5;
    overflow-x: auto;
}
.arch-box, .arch-box *, .arch-box pre { color: #7ee787 !important; }
/* ── Info card ─────────────────────────────────────────────────── */
.info-card {
    background: #161b22 !important;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    border: 1px solid #30363d;
}
.info-card * { color: #e6edf3 !important; }
.info-card table { background: #161b22 !important; }
.info-card td, .info-card th { color: #e6edf3 !important; }
/* Dark-header rows in tables */
.dark-header th { color: #ffffff !important; background: #0f3460 !important; }
/* Light-row on dark theme = slightly lighter dark bg */
.light-row { background: #1c2333 !important; }
.light-row td, .light-row th { color: #e6edf3 !important; }
/* ── Warning box ───────────────────────────────────────────────── */
.warning-box {
    background: #2d1f00 !important;
    border-left: 5px solid #ff9800;
    padding: 18px 20px;
    border-radius: 8px;
    margin: 15px 0;
}
.warning-box * { color: #ffd580 !important; }
/* ── Footer ────────────────────────────────────────────────────── */
.footer-box {
    text-align: center;
    padding: 25px;
    background: #161b22 !important;
    border-radius: 10px;
    margin-top: 20px;
    border: 1px solid #30363d;
}
.footer-box * { color: #e6edf3 !important; }
/* ── Gradio component overrides ────────────────────────────────── */
.gradio-container .label-wrap { color: #e6edf3 !important; }
.gradio-container .output-class { color: #e6edf3 !important; }
.gradio-container input, .gradio-container textarea,
.gradio-container select {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    border-color: #30363d !important;
}
.gradio-container button.primary {
    background: #238636 !important;
    color: #ffffff !important;
    border: 1px solid #2ea043 !important;
}
.gradio-container button.primary:hover {
    background: #2ea043 !important;
}
hr { border-color: #30363d !important; }
"""


def create_interface():
    with gr.Blocks(title="Bone Fracture Classifier | KBG Hackathon 2025", css=custom_css) as demo:

        # Inject CSS as <style> tag for bulletproof application
        gr.HTML(f"<style>{custom_css}</style>")

        # ── HEADER ─────────────────────────────────────────────
        gr.HTML("""
        <div class="main-header">
            <h1 style="margin:0; font-size:2.5em;">🦴 Bone Fracture Classification System</h1>
            <p style="margin:8px 0 0; font-size:1.15em; opacity:0.9;">
                Hybrid CNN + Vision Transformer with Cross-Attention Fusion
            </p>
            <p style="margin:5px 0 0; font-size:0.95em; opacity:0.75;">
                KBG Hackathon 2025 &nbsp;•&nbsp; Kamand Bioengineering Group &nbsp;•&nbsp; IIT Mandi
            </p>
        </div>
        """)

        # ── KEY METRICS BAR ────────────────────────────────────
        gr.HTML("""
        <div class="metric-grid">
            <div class="metric-card">
                <span class="label">Test Accuracy</span>
                <span class="value">98.22%</span>
            </div>
            <div class="metric-card">
                <span class="label">Macro F1-Score</span>
                <span class="value">0.9822</span>
            </div>
            <div class="metric-card">
                <span class="label">AUC-ROC</span>
                <span class="value">0.9970</span>
            </div>
            <div class="metric-card">
                <span class="label">Inference Speed</span>
                <span class="value">5.67ms</span>
            </div>
            <div class="metric-card">
                <span class="label">Model Size</span>
                <span class="value">130 MB</span>
            </div>
            <div class="metric-card">
                <span class="label">Training Time</span>
                <span class="value">~45 min</span>
            </div>
        </div>
        """)

        # ── CLASSIFIER SECTION ─────────────────────────────────
        gr.HTML("""
        <div class="section-card">
            <h2 style="margin:0;">🔬 Live Classification with Grad-CAM Explainability</h2>
            <p style="margin:5px 0 0; color:#e6edf3;">Upload a bone X-ray image to classify it as <b>Fractured</b> or <b>Not Fractured</b>.
            The Grad-CAM heatmap highlights regions the model focuses on for its decision.</p>
        </div>
        """)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="📤 Upload Bone X-ray", height=380)
                predict_btn = gr.Button("🔍 Classify Fracture", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_label = gr.Label(label="📊 Prediction", num_top_classes=2)
                output_cam = gr.Image(label="🔥 Grad-CAM Heatmap Overlay", height=380)

        with gr.Row():
            with gr.Column():
                output_report = gr.Textbox(label="📋 Detailed Analysis Report", lines=18, max_lines=25)

        predict_btn.click(fn=predict, inputs=input_image, outputs=[output_label, output_cam, output_report])
        input_image.change(fn=predict, inputs=input_image, outputs=[output_label, output_cam, output_report])

        gr.HTML("<hr style='margin:30px 0; border:none; border-top:1px solid #30363d;'>")

        # ── MODEL ARCHITECTURE ─────────────────────────────────
        gr.HTML("""
        <div class="section-card">
            <h2 style="margin:0;">🧠 Model Architecture — HybridCNNViT</h2>
            <p style="margin:5px 0 0; color:#e6edf3;">Dual-backbone architecture combining local CNN features with global ViT attention</p>
        </div>
        """)

        gr.HTML("""
        <div class="arch-box">
<pre style="color:#e0e0e0 !important; margin:0;">
              ┌─────────────────────────────────────────────────────────┐
              │              Input Image (B, 3, 224, 224)               │
              └─────────────┬───────────────────────┬───────────────────┘
                            │                       │
              ┌─────────────▼──────────┐ ┌──────────▼───────────────┐
              │   EfficientNet-B3      │ │   ViT-Small-Patch16-224  │
              │   CNN Backbone         │ │   Vision Transformer     │
              │   (d=1536, ImageNet)   │ │   (d=384, ImageNet)      │
              │   Local Features:      │ │   Global Features:       │
              │   texture, edges,      │ │   spatial relationships, │
              │   bone cortex patterns │ │   fracture line context  │
              └─────────────┬──────────┘ └──────────┬───────────────┘
                            │                       │
              ┌─────────────▼───────────────────────▼───────────────┐
              │           Cross-Attention Fusion (d=512)            │
              │   Q = CNN features | K,V = ViT features            │
              │   Bidirectional feature interaction                 │
              └─────────────────────────┬───────────────────────────┘
                                        │
              ┌─────────────────────────▼───────────────────────────┐
              │   Classification Head                               │
              │   Linear(512→256) → ReLU → Dropout(0.3) → Linear(2)│
              └─────────────────────────┬───────────────────────────┘
                                        │
              ┌─────────────────────────▼───────────────────────────┐
              │          Output: Fractured / Not Fractured          │
              └─────────────────────────────────────────────────────┘
</pre>
        </div>
        """)

        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="info-card">
                    <h3 style="margin-top:0; color:#e6edf3;">⚙️ Training Strategy — 2-Phase Transfer Learning</h3>
                    <table style="width:100%; border-collapse:collapse; margin-top:10px; background:#161b22;">
                        <tr style="background:#1c2333;" class="light-row">
                            <th style="padding:10px; text-align:left; border-bottom:2px solid #30363d; color:#e6edf3;">Setting</th>
                            <th style="padding:10px; text-align:left; border-bottom:2px solid #30363d; color:#e6edf3;">Phase 1 (Frozen)</th>
                            <th style="padding:10px; text-align:left; border-bottom:2px solid #30363d; color:#e6edf3;">Phase 2 (Fine-tune)</th>
                        </tr>
                        <tr class="light-row"><td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;"><b>Epochs</b></td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;">1–10</td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;">11–15 (early stopped)</td></tr>
                        <tr class="light-row"><td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;"><b>Trainable Params</b></td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;">1.58M (fusion + head)</td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;">33.9M (all layers)</td></tr>
                        <tr class="light-row"><td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;"><b>Learning Rate</b></td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;">1e-4</td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;">1e-5</td></tr>
                        <tr class="light-row"><td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;"><b>Optimizer</b></td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;" colspan="2">AdamW (weight_decay=1e-2)</td></tr>
                        <tr class="light-row"><td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;"><b>Scheduler</b></td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;" colspan="2">CosineAnnealingLR</td></tr>
                        <tr class="light-row"><td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;"><b>Loss Function</b></td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;" colspan="2">CrossEntropyLoss</td></tr>
                        <tr class="light-row"><td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;"><b>Batch Size</b></td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;" colspan="2">16</td></tr>
                        <tr class="light-row"><td style="padding:8px; color:#e6edf3;"><b>Early Stopping</b></td>
                            <td style="padding:8px; color:#e6edf3;" colspan="2">Patience=10 on Val F1 (best: epoch 13)</td></tr>
                    </table>
                </div>
                """)
            with gr.Column():
                gr.HTML("""
                <div class="info-card">
                    <h3 style="margin-top:0; color:#e6edf3;">📊 Dataset & Preprocessing</h3>
                    <table style="width:100%; border-collapse:collapse; margin-top:10px; background:#161b22;">
                        <tr style="background:#1c2333;" class="light-row">
                            <th style="padding:10px; text-align:left; border-bottom:2px solid #30363d; color:#e6edf3;">Property</th>
                            <th style="padding:10px; text-align:left; border-bottom:2px solid #30363d; color:#e6edf3;">Value</th>
                        </tr>
                        <tr class="light-row"><td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;"><b>Source</b></td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;">Kaggle Bone Fracture Binary Classification</td></tr>
                        <tr class="light-row"><td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;"><b>Total Images</b></td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;">10,581</td></tr>
                        <tr class="light-row"><td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;"><b>Train / Val / Test</b></td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;">9,246 / 829 / 506</td></tr>
                        <tr class="light-row"><td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;"><b>Classes</b></td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;">Fractured, Not Fractured</td></tr>
                        <tr class="light-row"><td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;"><b>Image Size</b></td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;">224 × 224 (resized)</td></tr>
                        <tr class="light-row"><td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;"><b>Normalization</b></td>
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;">ImageNet μ, σ</td></tr>
                    </table>
                    <h4 style="margin-top:15px;">🔄 Augmentation (Training Only)</h4>
                    <ul style="line-height:2; margin:5px 0;">
                        <li><b>Horizontal Flip</b> (p=0.5)</li>
                        <li><b>Affine Rotation</b> ±15°</li>
                        <li><b>Brightness/Contrast</b> ±20%</li>
                        <li><b>CLAHE</b> (clip_limit=2.0)</li>
                    </ul>
                </div>
                """)

        gr.HTML("<hr style='margin:30px 0; border:none; border-top:1px solid #30363d;'>")

        # ── RESULTS ────────────────────────────────────────────
        gr.HTML("""
        <div class="section-card">
            <h2 style="margin:0;">📈 Performance Results</h2>
            <p style="margin:5px 0 0; color:#e6edf3;">Comprehensive evaluation on held-out test set (506 images)</p>
        </div>
        """)

        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="info-card">
                    <h3 style="margin-top:0; color:#e6edf3;">Per-Class Metrics</h3>
                    <table style="width:100%; border-collapse:collapse; margin-top:10px; background:#161b22;">
                        <tr style="background:#1c2333;" class="light-row">
                            <th style="padding:10px; text-align:left; border-bottom:2px solid #30363d; color:#e6edf3;">Class</th>
                            <th style="padding:10px; text-align:center; border-bottom:2px solid #30363d; color:#e6edf3;">Precision</th>
                            <th style="padding:10px; text-align:center; border-bottom:2px solid #30363d; color:#e6edf3;">Recall</th>
                            <th style="padding:10px; text-align:center; border-bottom:2px solid #30363d; color:#e6edf3;">F1-Score</th>
                            <th style="padding:10px; text-align:center; border-bottom:2px solid #30363d; color:#e6edf3;">Support</th>
                        </tr>
                        <tr class="light-row">
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;"><b>Fractured</b></td>
                            <td style="padding:8px; text-align:center; border-bottom:1px solid #30363d; color:#e6edf3;">0.9791</td>
                            <td style="padding:8px; text-align:center; border-bottom:1px solid #30363d; color:#e6edf3;">0.9832</td>
                            <td style="padding:8px; text-align:center; border-bottom:1px solid #30363d; color:#e6edf3;">0.9811</td>
                            <td style="padding:8px; text-align:center; border-bottom:1px solid #30363d; color:#e6edf3;">238</td>
                        </tr>
                        <tr class="light-row">
                            <td style="padding:8px; border-bottom:1px solid #30363d; color:#e6edf3;"><b>Not Fractured</b></td>
                            <td style="padding:8px; text-align:center; border-bottom:1px solid #30363d; color:#e6edf3;">0.9850</td>
                            <td style="padding:8px; text-align:center; border-bottom:1px solid #30363d; color:#e6edf3;">0.9813</td>
                            <td style="padding:8px; text-align:center; border-bottom:1px solid #30363d; color:#e6edf3;">0.9832</td>
                            <td style="padding:8px; text-align:center; border-bottom:1px solid #30363d; color:#e6edf3;">268</td>
                        </tr>
                        <tr style="background:#0d2818;" class="light-row">
                            <td style="padding:8px; color:#e6edf3;"><b>Macro Average</b></td>
                            <td style="padding:8px; text-align:center; color:#e6edf3;"><b>0.9820</b></td>
                            <td style="padding:8px; text-align:center; color:#e6edf3;"><b>0.9823</b></td>
                            <td style="padding:8px; text-align:center; color:#e6edf3;"><b>0.9822</b></td>
                            <td style="padding:8px; text-align:center; color:#e6edf3;"><b>506</b></td>
                        </tr>
                    </table>
                    <p style="margin-top:15px; color:#e6edf3;">
                        <b>Confusion Matrix:</b> 234 TP, 263 TN, 4 FP, 5 FN — only <b>9 errors</b> out of 506 images.
                    </p>
                </div>
                """)

            with gr.Column():
                if os.path.exists(CM_IMG):
                    gr.Image(value=CM_IMG, label="Confusion Matrix", height=350)
                else:
                    gr.HTML("""
                    <div class="info-card" style="text-align:center; padding:40px;">
                        <h3 style="color:#e6edf3;">Confusion Matrix</h3>
                        <pre style="font-size:16px; color:#e6edf3;">
              Predicted
            Frac   Not-Frac
Frac  [ 234      4   ]
Not   [   5    263   ]
                        </pre>
                    </div>
                    """)

        gr.HTML("<hr style='margin:30px 0; border:none; border-top:1px solid #30363d;'>")

        # ── EXPLAINABILITY ─────────────────────────────────────
        gr.HTML("""
        <div class="section-card">
            <h2 style="margin:0;">🔍 Explainability — Grad-CAM & Attention Maps</h2>
            <p style="margin:5px 0 0; color:#e6edf3;">
                <b>Grad-CAM</b> highlights CNN regions relevant to classification.
                <b>Attention Rollout</b> shows ViT global focus patterns.
                Both confirm the model focuses on clinically relevant bone structures.
            </p>
        </div>
        """)

        with gr.Row():
            if os.path.exists(EXPLAIN_FRAC):
                with gr.Column():
                    gr.Image(value=EXPLAIN_FRAC, label="Fractured — Grad-CAM + Attention Analysis", height=350)
            if os.path.exists(EXPLAIN_NOFRAC):
                with gr.Column():
                    gr.Image(value=EXPLAIN_NOFRAC, label="Not Fractured — Grad-CAM + Attention Analysis", height=350)

        gr.HTML("<hr style='margin:30px 0; border:none; border-top:1px solid #30363d;'>")

        # ── TRAINING CURVES ────────────────────────────────────
        gr.HTML("""
        <div class="section-card">
            <h2 style="margin:0;">📉 Training Analysis</h2>
            <p style="margin:5px 0 0; color:#e6edf3;">Epoch-by-epoch performance tracking across both training phases</p>
        </div>
        """)

        gr.HTML("""
        <div class="info-card">
            <table style="width:100%; border-collapse:collapse; font-size:14px; background:#161b22;">
                <tr style="background:#0f3460;" class="dark-header">
                    <th style="padding:8px; color:#ffffff;">Epoch</th>
                    <th style="padding:8px; color:#ffffff;">Train Loss</th>
                    <th style="padding:8px; color:#ffffff;">Val Loss</th>
                    <th style="padding:8px; color:#ffffff;">Train Acc</th>
                    <th style="padding:8px; color:#ffffff;">Val Acc</th>
                    <th style="padding:8px; color:#ffffff;">Overfit Gap</th>
                    <th style="padding:8px; color:#ffffff;">LR</th>
                    <th style="padding:8px; color:#ffffff;">Phase</th>
                </tr>
                <tr style="background:#1c2333;" class="light-row"><td style="padding:6px; text-align:center; color:#e6edf3;">1</td><td style="padding:6px; text-align:center; color:#e6edf3;">0.2965</td><td style="padding:6px; text-align:center; color:#e6edf3;">0.1967</td><td style="padding:6px; text-align:center; color:#e6edf3;">87.09%</td><td style="padding:6px; text-align:center; color:#e6edf3;">91.44%</td><td style="padding:6px; text-align:center; color:#3fb950;">-4.35</td><td style="padding:6px; text-align:center; color:#e6edf3;">1e-4</td><td style="padding:6px; text-align:center; color:#e6edf3;">❄️ Frozen</td></tr>
                <tr style="background:#161b22;" class="light-row"><td style="padding:6px; text-align:center; color:#e6edf3;">5</td><td style="padding:6px; text-align:center; color:#e6edf3;">0.0420</td><td style="padding:6px; text-align:center; color:#e6edf3;">0.0943</td><td style="padding:6px; text-align:center; color:#e6edf3;">98.59%</td><td style="padding:6px; text-align:center; color:#e6edf3;">96.26%</td><td style="padding:6px; text-align:center; color:#e6edf3;">2.33</td><td style="padding:6px; text-align:center; color:#e6edf3;">6.5e-5</td><td style="padding:6px; text-align:center; color:#e6edf3;">❄️ Frozen</td></tr>
                <tr style="background:#1c2333;" class="light-row"><td style="padding:6px; text-align:center; color:#e6edf3;">10</td><td style="padding:6px; text-align:center; color:#e6edf3;">0.0172</td><td style="padding:6px; text-align:center; color:#e6edf3;">0.0612</td><td style="padding:6px; text-align:center; color:#e6edf3;">99.53%</td><td style="padding:6px; text-align:center; color:#e6edf3;">97.95%</td><td style="padding:6px; text-align:center; color:#e6edf3;">1.58</td><td style="padding:6px; text-align:center; color:#e6edf3;">2e-6</td><td style="padding:6px; text-align:center; color:#e6edf3;">❄️ Frozen</td></tr>
                <tr style="background:#0d2818;" class="light-row"><td style="padding:6px; text-align:center; color:#e6edf3;">11</td><td style="padding:6px; text-align:center; color:#e6edf3;">0.0312</td><td style="padding:6px; text-align:center; color:#e6edf3;">0.0423</td><td style="padding:6px; text-align:center; color:#e6edf3;">98.93%</td><td style="padding:6px; text-align:center; color:#e6edf3;">98.31%</td><td style="padding:6px; text-align:center; color:#e6edf3;">0.62</td><td style="padding:6px; text-align:center; color:#e6edf3;">1e-5</td><td style="padding:6px; text-align:center; color:#e6edf3;">🔥 Fine-tune</td></tr>
                <tr style="background:#0d3320; font-weight:bold;" class="light-row"><td style="padding:6px; text-align:center; color:#e6edf3;">13 ⭐</td><td style="padding:6px; text-align:center; color:#e6edf3;">0.0082</td><td style="padding:6px; text-align:center; color:#e6edf3;">0.0203</td><td style="padding:6px; text-align:center; color:#e6edf3;">99.74%</td><td style="padding:6px; text-align:center; color:#e6edf3;">99.16%</td><td style="padding:6px; text-align:center; color:#3fb950;">0.58</td><td style="padding:6px; text-align:center; color:#e6edf3;">1e-5</td><td style="padding:6px; text-align:center; color:#e6edf3;">🔥 Best</td></tr>
                <tr style="background:#0d2818;" class="light-row"><td style="padding:6px; text-align:center; color:#e6edf3;">15</td><td style="padding:6px; text-align:center; color:#e6edf3;">0.0083</td><td style="padding:6px; text-align:center; color:#e6edf3;">0.0268</td><td style="padding:6px; text-align:center; color:#e6edf3;">99.82%</td><td style="padding:6px; text-align:center; color:#e6edf3;">98.91%</td><td style="padding:6px; text-align:center; color:#e6edf3;">0.91</td><td style="padding:6px; text-align:center; color:#e6edf3;">9e-6</td><td style="padding:6px; text-align:center; color:#e6edf3;">🔥 Fine-tune</td></tr>
            </table>
            <div style="margin-top:15px; display:flex; gap:20px; flex-wrap:wrap;">
                <div style="background:#1c2333; padding:12px 18px; border-radius:8px; flex:1; min-width:200px;">
                    <b style="color:#e6edf3;">Max Overfit Gap:</b> <span style="color:#e6edf3;">2.73% (epoch 3) — </span><span style="color:#3fb950;">✓ minimal</span>
                </div>
                <div style="background:#1c2333; padding:12px 18px; border-radius:8px; flex:1; min-width:200px;">
                    <b style="color:#e6edf3;">Best Val Accuracy:</b> <span style="color:#e6edf3;">99.16% (epoch 13)</span>
                </div>
                <div style="background:#1c2333; padding:12px 18px; border-radius:8px; flex:1; min-width:200px;">
                    <b style="color:#e6edf3;">Train/Test Delta:</b> <span style="color:#e6edf3;">0.94% — </span><span style="color:#3fb950;">✓ excellent generalization</span>
                </div>
            </div>
        </div>
        """)

        gr.HTML("<hr style='margin:30px 0; border:none; border-top:1px solid #30363d;'>")

        # ── APPROACH JUSTIFICATION ─────────────────────────────
        gr.HTML("""
        <div class="section-card">
            <h2 style="margin:0;">📚 Approach Justification</h2>
        </div>
        """)

        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="info-card">
                    <h3 style="margin-top:0; color:#e6edf3;">Why Hybrid CNN + ViT?</h3>
                    <ul style="line-height:2; color:#e6edf3;">
                        <li><b>EfficientNet-B3 (CNN):</b> Captures local texture, edge, and bone cortex patterns — critical for detecting fine fracture lines</li>
                        <li><b>ViT-Small (Transformer):</b> Models long-range spatial dependencies — understands whole-bone geometry and fracture continuity</li>
                        <li><b>Cross-Attention Fusion:</b> Lets CNN and ViT features attend to each other — richer than simple concatenation</li>
                        <li><b>2-Phase Training:</b> First learns good fusion weights (frozen backbones), then fine-tunes end-to-end with low LR</li>
                    </ul>
                </div>
                """)
            with gr.Column():
                gr.HTML("""
                <div class="info-card">
                    <h3 style="margin-top:0; color:#e6edf3;">Key Design Decisions</h3>
                    <ul style="line-height:2; color:#e6edf3;">
                        <li><b>Only ImageNet pretrained:</b> No fracture/medical pretraining used — compliant with hackathon rules</li>
                        <li><b>Cross-Attention > Concat:</b> Q=CNN, K=V=ViT allows bidirectional feature interaction (d=512)</li>
                        <li><b>EfficientNet-B3:</b> Best FLOPs/accuracy trade-off among EfficientNet variants</li>
                        <li><b>ViT-Small Patch16:</b> 22M params with 16×16 patches — good balance of detail and efficiency</li>
                        <li><b>CLAHE Augmentation:</b> Enhances local contrast in bone structures for better feature extraction</li>
                    </ul>
                </div>
                """)

        gr.HTML("<hr style='margin:30px 0; border:none; border-top:1px solid #30363d;'>")

        # ── SYSTEM INFO ────────────────────────────────────────
        gr.HTML("""
        <div class="section-card">
            <h2 style="margin:0;">⚙️ System & Reproducibility</h2>
        </div>
        """)

        gr.HTML("""
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:15px; margin:15px 0;">
            <div class="info-card">
                <h3 style="margin-top:0; color:#e6edf3;">🔧 Technical Stack</h3>
                <ul style="line-height:2; color:#e6edf3;">
                    <li><b>Framework:</b> PyTorch 2.10.0</li>
                    <li><b>Model Library:</b> timm 1.0.25</li>
                    <li><b>Augmentation:</b> Albumentations</li>
                    <li><b>Visualization:</b> Matplotlib + OpenCV</li>
                    <li><b>Web UI:</b> Gradio</li>
                    <li><b>Device:</b> Apple M2 (MPS)</li>
                </ul>
            </div>
            <div class="info-card">
                <h3 style="margin-top:0; color:#e6edf3;">🔁 Reproducibility</h3>
                <ul style="line-height:2; color:#e6edf3;">
                    <li><b>Random Seed:</b> 42 (all libraries)</li>
                    <li><b>Config:</b> config.yaml (all hyperparams)</li>
                    <li><b>Requirements:</b> requirements.txt (pinned versions)</li>
                    <li><b>Checkpoint:</b> model.pth (epoch 13)</li>
                    <li><b>Codebase:</b> Modular (data_loader, model, train, evaluate, explainability)</li>
                </ul>
            </div>
            <div class="info-card">
                <h3 style="margin-top:0; color:#e6edf3;">📂 Project Structure</h3>
                <pre style="font-size:13px; line-height:1.6; margin:0; color:#e6edf3;">
├── config.yaml
├── data_loader.py
├── model.py
├── train.py
├── evaluate.py
├── explainability.py
├── app.py
├── requirements.txt
├── checkpoints/model.pth
└── results/
    ├── confusion_matrix.png
    └── explainability/</pre>
            </div>
        </div>
        """)

        # ── DISCLAIMER ─────────────────────────────────────────
        gr.HTML("""
        <div class="warning-box">
            <h3 style="margin-top:0; color:#e6edf3;">⚠️ Medical Disclaimer</h3>
            <p style="margin:0; line-height:1.6; color:#e6edf3;">
                This system is provided for <b>educational and research purposes only</b> as part of the
                KBG Hackathon 2025 at IIT Mandi. All results must be verified by qualified healthcare professionals.
                This is <b>NOT a medical device</b> and should not be used as the sole basis for clinical decisions.
            </p>
        </div>
        """)

        # ── FOOTER ─────────────────────────────────────────────
        gr.HTML("""
        <div class="footer-box">
            <p style="font-size:18px; margin:0; font-weight:600;">
                Developed by <span style="color:#58a6ff;"><b>Aditya Rai</b></span>
            </p>
            <p style="font-size:14px; margin:8px 0 0; color:#e6edf3;">
                KBG Hackathon 2025 &nbsp;•&nbsp; Kamand Bioengineering Group &nbsp;•&nbsp; IIT Mandi
            </p>
            <p style="font-size:13px; margin:5px 0 0; color:#8b949e;">
                Powered by PyTorch, timm, Gradio & Hugging Face
            </p>
        </div>
        """)

    return demo


# ── Launch ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Bone Fracture Classification — KBG Hackathon 2025".center(60))
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    dev = "CUDA" if torch.cuda.is_available() else ("MPS" if torch.backends.mps.is_available() else "CPU")
    print(f"Device:  {dev}")
    print(f"Model:   {CHECKPOINT}")
    print("=" * 60 + "\n")

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
