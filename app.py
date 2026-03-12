#!/usr/bin/env python3
"""
Unified Medical X-ray Classifier
KBG Hackathon 2025 — IIT Mandi
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

# ── Paths ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
UNIFIED_CHECKPOINT = os.path.join(ROOT, "checkpoints", "unified_model.pth")

# Class names
CLASS_NAMES = ["fractured", "not fractured", "NORMAL", "PNEUMONIA"]
DISPLAY_NAMES = {
    "fractured": "Fractured",
    "not fractured": "Not Fractured",
    "NORMAL": "Normal Lungs",
    "PNEUMONIA": "Pneumonia",
}
BONE_CLASSES = {0, 1}
CHEST_CLASSES = {2, 3}


# ── Device ─────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model loader ───────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def load_model():
    device = get_device()
    if not os.path.exists(UNIFIED_CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found at {UNIFIED_CHECKPOINT}. "
            "Run train_unified.py first."
        )

    ckpt = torch.load(UNIFIED_CHECKPOINT, map_location=device, weights_only=False)
    class_names = ckpt.get("class_names", CLASS_NAMES)
    config = ckpt.get("config", {})
    config["num_classes"] = len(class_names)
    config.setdefault("model_backbone", "efficientnet_b3")
    config.setdefault("vit_model", "vit_small_patch16_224")
    config.setdefault("fusion_method", "cross_attention")
    config.setdefault("dropout", 0.3)
    config.setdefault("image_size", 224)

    mdl = build_model(config)
    mdl.load_state_dict(ckpt["model_state_dict"])
    mdl.to(device).eval()

    val_acc = ckpt.get("val_accuracy", "N/A")
    epoch = ckpt.get("epoch", "N/A")
    print(f"[app] Model loaded | {device} | Val Acc: {val_acc}% | Epoch: {epoch}")
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

    confidences = {}
    for i, cn in enumerate(class_names):
        confidences[DISPLAY_NAMES.get(cn, cn)] = float(probs[0, i].item())

    colormap = cv2.COLORMAP_MAGMA if pred_idx in CHEST_CLASSES else cv2.COLORMAP_JET
    if cam is not None:
        cam_r = cv2.resize(cam, (orig_w, orig_h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_r), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.clip(
            np.float32(heatmap) * 0.4 + np.float32(img_np) * 0.6, 0, 255
        ).astype(np.uint8)
        overlay_img = Image.fromarray(overlay)
    else:
        overlay_img = image

    pred_class = class_names[pred_idx]
    pred_display = DISPLAY_NAMES.get(pred_class, pred_class)
    conf_pct = probs[0, pred_idx].item() * 100
    xray_type = "Bone X-ray" if pred_idx in BONE_CLASSES else "Chest X-ray"

    bone_frac = probs[0, 0].item() * 100
    bone_nofrac = probs[0, 1].item() * 100
    chest_normal = probs[0, 2].item() * 100
    chest_pneum = probs[0, 3].item() * 100

    def bar(val):
        filled = int(val / 5)
        return "=" * filled + "-" * (20 - filled)

    report = f"""
  MEDICAL X-RAY ANALYSIS REPORT
  -----------------------------------------------

  Detected Type  :  {xray_type}
  Classification :  {pred_display}
  Confidence     :  {conf_pct:.1f}%

  -----------------------------------------------
  Bone Analysis
  -----------------------------------------------
      Fractured     :  [{bar(bone_frac)}] {bone_frac:.1f}%
      Not Fractured :  [{bar(bone_nofrac)}] {bone_nofrac:.1f}%

  -----------------------------------------------
  Chest Analysis
  -----------------------------------------------
      Normal        :  [{bar(chest_normal)}] {chest_normal:.1f}%
      Pneumonia     :  [{bar(chest_pneum)}] {chest_pneum:.1f}%

  -----------------------------------------------
  Interpretation
  -----------------------------------------------
"""
    if pred_idx == 0:
        report += "  Fracture indicators detected.\n"
        report += "  The heatmap highlights the suspected fracture region.\n"
    elif pred_idx == 1:
        report += "  No fracture indicators detected.\n"
        report += "  The bone structure appears intact.\n"
    elif pred_idx == 2:
        report += "  No pneumonia indicators detected.\n"
        report += "  The lung fields appear clear.\n"
    elif pred_idx == 3:
        report += "  Pneumonia indicators detected.\n"
        report += "  The heatmap highlights the affected lung regions.\n"

    if conf_pct < 80:
        report += "\n  Note: Moderate confidence — consider clinical correlation.\n"

    report += "\n  This is a research tool. Always consult a qualified\n  radiologist for clinical decisions."

    return confidences, overlay_img, report


# ══════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════

custom_css = """
body, .gradio-container, .main, .app, .contain,
.gradio-container > div, .gradio-container > div > div {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
}
.gradio-container *, .gradio-container p, .gradio-container span,
.gradio-container li, .gradio-container td, .gradio-container th,
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container h4, .gradio-container b, .gradio-container strong,
.gradio-container label, .gradio-container div, .gradio-container a,
.gradio-container pre, .gradio-container input, .gradio-container textarea {
    color: #c9d1d9 !important;
}
.gradio-container .block, .gradio-container .tabitem,
.gradio-container .panel, .gradio-container .form,
.gradio-container .wrap {
    background: #161b22 !important;
    border-color: #21262d !important;
}
.header-bar {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%) !important;
    padding: 32px 24px; border-radius: 8px; text-align: center;
    margin-bottom: 16px; border: 1px solid #21262d;
}
.header-bar h1 { color: #f0f6fc !important; font-size: 1.8em; margin: 0; font-weight: 600; letter-spacing: -0.02em; }
.header-bar p { color: #8b949e !important; margin: 6px 0 0; font-size: 0.95em; }
.stat-row {
    display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; justify-content: center;
}
.stat-pill {
    background: #161b22 !important; border: 1px solid #21262d;
    padding: 10px 20px; border-radius: 6px; text-align: center; flex: 1; min-width: 140px;
}
.stat-pill .num { font-size: 1.4em; font-weight: 700; color: #58a6ff !important; display: block; }
.stat-pill .lbl { font-size: 0.78em; color: #8b949e !important; text-transform: uppercase; letter-spacing: 0.05em; }
.subtle-card {
    background: #161b22 !important; padding: 16px 20px; border-radius: 8px;
    border: 1px solid #21262d; margin: 8px 0;
}
.subtle-card h3 { margin: 0 0 8px; font-size: 0.95em; color: #c9d1d9 !important; font-weight: 600; }
.subtle-card p, .subtle-card li { color: #8b949e !important; font-size: 0.88em; line-height: 1.7; }
.disclaimer {
    background: #161b22 !important; border: 1px solid #30363d;
    padding: 14px 20px; border-radius: 6px; margin: 20px 0 8px;
}
.disclaimer p { color: #8b949e !important; font-size: 0.82em; margin: 0; line-height: 1.5; }
.disclaimer b { color: #c9d1d9 !important; }
.footer-line {
    text-align: center; padding: 16px; margin-top: 8px;
}
.footer-line p { color: #484f58 !important; font-size: 0.8em; margin: 2px 0; }
.footer-line span { color: #58a6ff !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select {
    background: #0d1117 !important; color: #c9d1d9 !important; border-color: #21262d !important;
}
.gradio-container button.primary {
    background: #238636 !important; color: #fff !important; border: 1px solid #2ea043 !important;
    font-weight: 600;
}
.gradio-container button.primary:hover { background: #2ea043 !important; }
hr { border-color: #21262d !important; }
"""


def create_interface():
    with gr.Blocks(title="Medical X-ray Analyzer") as demo:

        gr.HTML(f"<style>{custom_css}</style>")

        # Header
        gr.HTML("""
        <div class="header-bar">
            <h1>Medical X-ray Analysis System</h1>
            <p>Bone Fracture Detection &middot; Lung Pneumonia Detection &middot; Unified Model</p>
        </div>
        """)

        # Stats
        gr.HTML("""
        <div class="stat-row">
            <div class="stat-pill">
                <span class="num">HybridCNNViT</span>
                <span class="lbl">Model</span>
            </div>
            <div class="stat-pill">
                <span class="num">95.86%</span>
                <span class="lbl">Val Accuracy</span>
            </div>
            <div class="stat-pill">
                <span class="num">87.5%</span>
                <span class="lbl">Test Accuracy</span>
            </div>
            <div class="stat-pill">
                <span class="num">4</span>
                <span class="lbl">Classes</span>
            </div>
            <div class="stat-pill">
                <span class="num">14,462</span>
                <span class="lbl">Training Images</span>
            </div>
        </div>
        """)

        # Classifier
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload X-ray", height=400)
                predict_btn = gr.Button("Analyze", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_label = gr.Label(label="Classification", num_top_classes=4)
                output_cam = gr.Image(label="Grad-CAM Heatmap", height=400)

        with gr.Row():
            with gr.Column():
                output_report = gr.Textbox(label="Report", lines=22, max_lines=30)

        predict_btn.click(
            fn=predict,
            inputs=input_image,
            outputs=[output_label, output_cam, output_report],
        )
        input_image.change(
            fn=predict,
            inputs=input_image,
            outputs=[output_label, output_cam, output_report],
        )

        gr.HTML("<hr style='margin:24px 0; border:none; border-top:1px solid #21262d;'>")

        # Features
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="subtle-card">
                    <h3>How it works</h3>
                    <p>Upload any bone or chest X-ray. The model automatically
                    determines the type and classifies it into one of four
                    categories: Fractured, Not Fractured, Normal Lungs,
                    or Pneumonia.</p>
                </div>
                """)
            with gr.Column():
                gr.HTML("""
                <div class="subtle-card">
                    <h3>Grad-CAM Explainability</h3>
                    <p>Each prediction includes a heatmap overlay showing which
                    regions of the image the model focused on. Bone X-rays use
                    a warm colormap, chest X-rays use a purple-bright scale.</p>
                </div>
                """)
            with gr.Column():
                gr.HTML("""
                <div class="subtle-card">
                    <h3>Model Details</h3>
                    <ul style="margin:0; padding-left:18px;">
                        <li>EfficientNet-B3 + ViT-Small backbone</li>
                        <li>Cross-attention fusion</li>
                        <li>Trained on 14,462 images (8 epochs)</li>
                        <li>Weighted loss for class imbalance</li>
                    </ul>
                </div>
                """)

        # Disclaimer
        gr.HTML("""
        <div class="disclaimer">
            <p><b>Disclaimer</b> — This system is for educational and research
            purposes only (KBG Hackathon 2025, IIT Mandi). It is not a certified
            medical device. All results must be verified by qualified healthcare
            professionals before any clinical decision is made.</p>
        </div>
        """)

        # Footer
        gr.HTML("""
        <div class="footer-line">
            <p>Built by <span>Aditya Rai</span></p>
            <p>KBG Hackathon 2025 &middot; IIT Mandi &middot; PyTorch &middot; Gradio</p>
        </div>
        """)

    return demo


# ── Launch ────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
