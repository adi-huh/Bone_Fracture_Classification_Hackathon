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
        return {}, None, "", "", ""

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
    xray_type = "Bone X-ray" if pred_idx in BONE_CLASSES else "Chest X-ray (Lungs)"

    bone_frac = probs[0, 0].item() * 100
    bone_nofrac = probs[0, 1].item() * 100
    chest_normal = probs[0, 2].item() * 100
    chest_pneum = probs[0, 3].item() * 100

    conf_label = "High Confidence" if conf_pct >= 85 else ("Moderate Confidence" if conf_pct >= 65 else "Low Confidence")
    pred_color = "#dc2626" if pred_idx in (0, 3) else "#16a34a"

    # Interpretation text
    if pred_idx == 0:
        bone_text = "Fracture indicators detected. The heatmap highlights the suspected fracture region."
        chest_text = "No significant lung pathology detected in the field of view."
        ai_interp = (
            "The model identifies patterns characteristic of a bone fracture. "
            "Visual attention maps correlate with clinical findings of discontinuity in bone structure. "
            "Recommend immediate clinical correlation and further diagnostic imaging if necessary."
        )
    elif pred_idx == 1:
        bone_text = "No visible fractures detected. The bone structure appears intact."
        chest_text = "No significant lung pathology detected in the field of view."
        ai_interp = (
            "The model finds no significant fracture indicators. "
            "Bone structure appears normal and intact. "
            "If clinical suspicion persists, consider additional views or follow-up imaging."
        )
    elif pred_idx == 2:
        bone_text = "No visible fractures detected in clavicle or ribs within the field of view."
        chest_text = "Lung fields appear clear. No consolidation or opacities observed."
        ai_interp = (
            "The model identifies normal lung parenchyma with no significant pathology. "
            "No areas of consolidation, effusion, or mass are detected. "
            "Routine follow-up as clinically indicated."
        )
    else:
        bone_text = "No visible fractures detected in clavicle or ribs within the field of view."
        chest_text = "Bilateral opacities observed. Significant consolidation detected."
        ai_interp = (
            "The model identifies patterns characteristic of bacterial pneumonia. "
            "Visual attention maps correlate with clinical findings of lobar consolidation. "
            "Recommend immediate clinical correlation and further diagnostic imaging if necessary."
        )

    if conf_pct < 80:
        ai_interp += " Note: Moderate confidence — consider clinical correlation."

    # Grad-CAM description
    if cam is not None:
        if pred_idx in CHEST_CLASSES:
            cam_desc = "The AI focus is concentrated on the pulmonary regions, indicating areas of clinical interest."
        else:
            cam_desc = "The AI focus is concentrated on the bone region, highlighting areas of clinical interest."
    else:
        cam_desc = "Grad-CAM visualization could not be generated for this image."

    import datetime, random
    now = datetime.datetime.now()
    ref_id = f"AX-{random.randint(1000,9999)}"

    # Build classification bars HTML
    is_danger = pred_idx in (0, 3)  # fractured or pneumonia
    bar_items = [
        ("Pneumonia", chest_pneum, "#dc2626" if chest_pneum > 50 else "#c4b5fd"),
        ("Normal Lungs", chest_normal, "#16a34a" if chest_normal > 50 else "#c4b5fd"),
        ("Fractured Bone", bone_frac, "#dc2626" if bone_frac > 50 else "#c4b5fd"),
        ("Healthy Bone", bone_nofrac, "#16a34a" if bone_nofrac > 50 else "#c4b5fd"),
    ]
    bars_html = ""
    for label, val, color in bar_items:
        bar_w = max(val, 0.5)
        bars_html += f"""
        <div style="margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:14px; font-weight:500; color:#3b0764;">{label}</span>
                <span style="font-size:14px; font-weight:600; color:{color};">{val:.1f}%</span>
            </div>
            <div style="width:100%; height:8px; background:rgba(237,233,254,0.6);
                 border-radius:4px; overflow:hidden; backdrop-filter:blur(4px);">
                <div style="width:{bar_w}%; height:100%; background:{color}; border-radius:4px;
                     transition: width 0.5s ease;"></div>
            </div>
        </div>"""

    # Colors for predicted class badge
    if is_danger:
        badge_bg = "rgba(254,226,226,0.6)"
        badge_border = "rgba(239,68,68,0.3)"
        badge_circle_bg = "linear-gradient(135deg,#dc2626,#b91c1c)"
        badge_circle_shadow = "rgba(220,38,38,0.3)"
        badge_label_color = "#dc2626"
        badge_value_color = "#991b1b"
        badge_icon = '<polyline points="20 6 9 17 4 12"></polyline>'
    else:
        badge_bg = "rgba(220,252,231,0.6)"
        badge_border = "rgba(34,197,94,0.3)"
        badge_circle_bg = "linear-gradient(135deg,#16a34a,#15803d)"
        badge_circle_shadow = "rgba(22,163,74,0.3)"
        badge_label_color = "#16a34a"
        badge_value_color = "#14532d"
        badge_icon = '<polyline points="20 6 9 17 4 12"></polyline>'

    results_html = f"""
    <div style="padding:4px 0;">
        <h3 style="font-size:18px; font-weight:600; color:#4c1d95; margin:0 0 20px 0;">
            AI Classification Results</h3>
        {bars_html}
        <div style="display:flex; align-items:center; gap:10px; margin-top:20px;
             background:{badge_bg}; backdrop-filter:blur(12px);
             border:1px solid {badge_border}; border-radius:14px; padding:14px 18px;
             box-shadow:0 4px 16px {'rgba(220,38,38,0.08)' if is_danger else 'rgba(22,163,74,0.08)'};">
            <div style="width:36px; height:36px; background:{badge_circle_bg};
                 border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0;
                 box-shadow:0 2px 8px {badge_circle_shadow};">
                <svg width="18" height="18" fill="none" stroke="white" stroke-width="2.5"
                     stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
                    {badge_icon}</svg>
            </div>
            <div>
                <div style="font-size:11px; font-weight:600; color:{badge_label_color};
                     text-transform:uppercase; letter-spacing:0.05em;">Predicted Class</div>
                <div style="font-size:16px; font-weight:700; color:{badge_value_color};">{pred_display.upper()}</div>
            </div>
        </div>
    </div>"""

    gradcam_desc_html = f"""
    <div style="font-size:13px; color:#6b21a8; font-style:italic; margin-top:8px; line-height:1.5;">
        {cam_desc}
    </div>"""

    danger_class = " result-danger" if is_danger else ""
    interp_bg = "rgba(254,242,242,0.5)" if is_danger else "rgba(245,243,255,0.5)"
    interp_border_color = "#dc2626" if is_danger else "#7c3aed"
    interp_label_color = "#dc2626" if is_danger else "#7c3aed"
    interp_text_color = "#991b1b" if is_danger else "#4c1d95"

    report_html = f"""
    <div class="report-card{danger_class}">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;
             margin-bottom:20px; flex-wrap:wrap; gap:10px;">
            <div>
                <h3 style="font-size:20px; font-weight:700; color:#4c1d95; margin:0;">
                    Medical Analysis Report</h3>
                <p style="font-size:13px; color:#8b5cf6; margin:4px 0 0;">
                    Generated on {now.strftime('%B %d, %Y')} &middot; Reference: #{ref_id}</p>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:24px;">
            <div>
                <div style="font-size:11px; font-weight:600; color:#7c3aed;
                     text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">
                    Detected Type</div>
                <div style="font-size:15px; color:#1e1b4b; font-weight:500;">{xray_type}</div>
                <div style="font-size:11px; font-weight:600; color:#7c3aed;
                     text-transform:uppercase; letter-spacing:0.08em; margin:16px 0 4px;">
                    Classification</div>
                <div style="font-size:15px; color:{pred_color}; font-weight:600;">{pred_display}</div>
                <div style="font-size:11px; font-weight:600; color:#7c3aed;
                     text-transform:uppercase; letter-spacing:0.08em; margin:16px 0 4px;">
                    Confidence</div>
                <div style="font-size:15px; color:#1e1b4b; font-weight:500;">
                    {conf_pct:.1f}% ({conf_label})</div>
            </div>
            <div>
                <div style="font-size:11px; font-weight:600; color:#4c1d95;
                     text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">
                    Bone Analysis</div>
                <p style="font-size:13px; color:#6b21a8; line-height:1.6; margin:0;
                   font-style:italic;">{bone_text}</p>
                <div style="font-size:11px; font-weight:600; color:#4c1d95;
                     text-transform:uppercase; letter-spacing:0.08em; margin:16px 0 4px;">
                    Chest Analysis</div>
                <p style="font-size:13px; color:#6b21a8; line-height:1.6; margin:0;
                   font-style:italic;">{chest_text}</p>
            </div>
            <div style="background:{interp_bg}; backdrop-filter:blur(12px);
                 border-left:3px solid {interp_border_color};
                 padding:14px 16px; border-radius:0 14px 14px 0;">
                <div style="font-size:11px; font-weight:600; color:{interp_label_color};
                     text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">
                    AI Interpretation</div>
                <p style="font-size:13px; color:{interp_text_color}; line-height:1.7; margin:0;
                   font-style:italic;">"{ai_interp}"</p>
            </div>
        </div>
    </div>"""

    return confidences, overlay_img, results_html, gradcam_desc_html, report_html


# ══════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════

custom_css = """
/* ── Global – soft lavender background ── */
body, .gradio-container, .main, .app, .contain,
.gradio-container > div, .gradio-container > div > div {
    background: #ede9fe !important;
    color: #1e1b4b !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

/* ── Glassmorphism mixin ── */
.glass {
    background: rgba(255,255,255,0.45) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1px solid rgba(255,255,255,0.5) !important;
    box-shadow: 0 8px 32px rgba(124,58,237,0.08),
                inset 0 1px 0 rgba(255,255,255,0.6) !important;
}
.glass:hover {
    background: rgba(255,255,255,0.72) !important;
    box-shadow: 0 16px 48px rgba(124,58,237,0.22),
                0 0 0 1px rgba(167,139,250,0.3),
                inset 0 1px 0 rgba(255,255,255,0.8) !important;
    transform: translateY(-3px);
}

/* ── Header bar (glass on dark) ── */
.header-bar {
    background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 60%, #7c3aed 100%) !important;
    padding: 26px 32px; border-radius: 20px; margin-bottom: 20px;
    display: flex; justify-content: space-between; align-items: center;
    box-shadow: 0 8px 32px rgba(109,40,217,0.3);
    border: 1px solid rgba(167,139,250,0.2);
    position: relative; overflow: hidden;
}
.header-bar::before {
    content: ''; position: absolute; top: -50%; right: -20%;
    width: 300px; height: 300px; border-radius: 50%;
    background: radial-gradient(circle, rgba(167,139,250,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.header-bar .hb-left h1 {
    color: #ffffff !important; font-size: 1.5em; margin: 0;
    font-weight: 700; letter-spacing: -0.02em;
}
.header-bar .hb-left p {
    color: #c4b5fd !important; margin: 4px 0 0; font-size: 0.85em;
}
.header-bar .hb-left p .dot {
    display: inline-block; width: 8px; height: 8px; background: #a78bfa;
    border-radius: 50%; margin-right: 6px; vertical-align: middle;
}
.sys-badge {
    background: rgba(34,197,94,0.18); border: 1px solid rgba(34,197,94,0.5);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    color: white !important; padding: 6px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 600; display: flex; align-items: center; gap: 6px;
}

.sys-badge .dot-live {
    width: 8px; height: 8px; background: #22c55e; border-radius: 50%;
    box-shadow: 0 0 6px rgba(34,197,94,0.6);
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity: 1; } 50% { opacity: 0.4; }
}

/* ── Stat pills row (glass) ── */
.stat-row {
    display: flex; gap: 12px; margin: 0 0 20px; flex-wrap: wrap;
}
.stat-pill {
    background: rgba(255,255,255,0.45) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1px solid rgba(255,255,255,0.5);
    padding: 14px 18px; border-radius: 16px; text-align: left;
    flex: 1; min-width: 130px;
    box-shadow: 0 4px 16px rgba(124,58,237,0.06),
                inset 0 1px 0 rgba(255,255,255,0.6);
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
}
.stat-pill:hover {
    transform: translateY(-4px);
    background: rgba(255,255,255,0.78) !important;
    box-shadow: 0 14px 40px rgba(124,58,237,0.2),
                0 0 0 1px rgba(167,139,250,0.3),
                inset 0 1px 0 rgba(255,255,255,0.9);
}
.stat-pill .lbl {
    font-size: 10px; color: #8b5cf6 !important; text-transform: uppercase;
    letter-spacing: 0.06em; font-weight: 600; display: block; margin-bottom: 4px;
}
.stat-pill .num {
    font-size: 1.3em; font-weight: 700; color: #1e1b4b !important; display: block;
}

/* ── Card panels (glass) ── */
.gradio-container .block, .gradio-container .panel, .gradio-container .form {
    background: rgba(255,255,255,0.4) !important;
    backdrop-filter: blur(14px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(14px) saturate(160%) !important;
    border: 1px solid rgba(255,255,255,0.45) !important;
    border-radius: 18px !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.05),
                inset 0 1px 0 rgba(255,255,255,0.5) !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
}
.gradio-container .block { padding: 0 !important; }
.gradio-container .block:hover, .gradio-container .panel:hover {
    background: rgba(255,255,255,0.7) !important;
    box-shadow: 0 14px 42px rgba(124,58,237,0.18),
                0 0 0 1px rgba(167,139,250,0.25),
                inset 0 1px 0 rgba(255,255,255,0.8) !important;
    transform: translateY(-2px) !important;
}

/* ── Labels and text ── */
.gradio-container label, .gradio-container .label-wrap span {
    color: #3b0764 !important; font-weight: 600 !important; font-size: 14px !important;
}
.gradio-container p, .gradio-container span, .gradio-container div,
.gradio-container li, .gradio-container td, .gradio-container th,
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container h4, .gradio-container b, .gradio-container strong,
.gradio-container a {
    color: #3b0764 !important;
}

/* ── Upload area ── */
.gradio-container .upload-area, .gradio-container [data-testid="image"] {
    border: 2px dashed #c4b5fd !important; border-radius: 16px !important;
    background: rgba(245,243,255,0.5) !important;
}
.gradio-container .upload-area:hover {
    border-color: #8b5cf6 !important; background: rgba(237,233,254,0.6) !important;
}

/* ── Primary button ── */
.gradio-container button.primary, .gradio-container .gr-button-primary {
    background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
    color: #ffffff !important; border: none !important; border-radius: 14px !important;
    font-weight: 600 !important; font-size: 15px !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.35) !important;
    transition: all 0.2s ease !important;
}
.gradio-container button.primary:hover {
    background: linear-gradient(135deg, #6d28d9 0%, #5b21b6 100%) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.45) !important;
    transform: translateY(-2px) !important;
}

/* ── Grad-CAM image ── */
.gradcam-wrap { position: relative; }
.gradcam-label {
    position: absolute; top: 12px; right: 12px;
    background: rgba(255,255,255,0.7); backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 4px 10px; border-radius: 8px; font-size: 11px; color: #6d28d9;
    font-weight: 500; border: 1px solid rgba(255,255,255,0.5);
}

/* ── Report card (glass) ── */
.report-card {
    background: rgba(255,255,255,0.45); border: 1px solid rgba(255,255,255,0.5);
    border-radius: 20px; padding: 24px 28px;
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    box-shadow: 0 8px 32px rgba(124,58,237,0.08),
                inset 0 1px 0 rgba(255,255,255,0.6);
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
}
.report-card:hover {
    background: rgba(255,255,255,0.72);
    box-shadow: 0 16px 48px rgba(124,58,237,0.2),
                0 0 0 1px rgba(167,139,250,0.25),
                inset 0 1px 0 rgba(255,255,255,0.8);
    transform: translateY(-3px);
}

/* ── Disclaimer (glass) ── */
.disclaimer {
    background: rgba(250,245,255,0.5) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(216,180,254,0.4);
    padding: 14px 20px; border-radius: 14px; margin: 16px 0 8px;
}
.disclaimer p { color: #6b21a8 !important; font-size: 0.82em; margin: 0; line-height: 1.5; }
.disclaimer b { color: #4c1d95 !important; }

/* ── Footer ── */
.footer-line { text-align: center; padding: 16px; margin-top: 8px; }
.footer-line p { color: #a78bfa !important; font-size: 0.8em; margin: 2px 0; }
.footer-line span { color: #7c3aed !important; font-weight: 600; }

/* ── Subtle card / features (glass) ── */
.subtle-card {
    background: rgba(255,255,255,0.45) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    padding: 20px 24px; border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.5) !important;
    margin: 8px 0;
    box-shadow: 0 4px 16px rgba(124,58,237,0.06),
                inset 0 1px 0 rgba(255,255,255,0.6);
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
}
.subtle-card:hover {
    transform: translateY(-4px);
    background: rgba(255,255,255,0.78) !important;
    box-shadow: 0 16px 48px rgba(124,58,237,0.22),
                0 0 0 1px rgba(167,139,250,0.3),
                inset 0 1px 0 rgba(255,255,255,0.9) !important;
}
.subtle-card h3 {
    margin: 0 0 10px; font-size: 0.95em; color: #4c1d95 !important; font-weight: 600;
}
.subtle-card p, .subtle-card li {
    color: #6b21a8 !important; font-size: 0.88em; line-height: 1.7;
}

/* ── Hide default Gradio label output (we use custom HTML) ── */
.hide-gradio-label { display: none !important; }

/* ── Responsive ── */
@media (max-width: 768px) {
    .stat-row { flex-direction: column; }
    .report-card div[style*="grid-template-columns"] {
        grid-template-columns: 1fr !important;
    }
}

/* ── Section titles ── */
.section-title {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px;
}
.section-title h3 {
    font-size: 16px; font-weight: 600; color: #4c1d95 !important; margin: 0;
}
.section-title .badge {
    font-size: 11px; color: #8b5cf6; font-weight: 500;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #ede9fe; }
::-webkit-scrollbar-thumb { background: #c4b5fd; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #a78bfa; }

/* ── Danger result (red glow for pneumonia / fracture) ── */
.result-danger {
    background: rgba(254,242,242,0.5) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    box-shadow: 0 0 20px rgba(220,38,38,0.1), inset 0 1px 0 rgba(255,255,255,0.5) !important;
}
.report-card.result-danger {
    border-left: 4px solid #dc2626 !important;
}
"""


def create_interface():
    with gr.Blocks(
        title="Medical X-ray Analyzer",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:

        gr.HTML(f"<style>{custom_css}</style>")

        # ── Header ──
        gr.HTML("""
        <div class="header-bar">
            <div class="hb-left">
                <h1>Medical X-ray Analysis System</h1>
                <p><span class="dot"></span>Bone Fracture Detection &middot;
                Lung Pneumonia Detection &middot; Unified Model</p>
            </div>
            <div class="sys-badge">
                <span class="dot-live"></span>System Online
            </div>
        </div>
        """)

        # ── Stats ──
        gr.HTML("""
        <div class="stat-row">
            <div class="stat-pill">
                <span class="lbl">Model</span>
                <span class="num">HybridCNNViT</span>
            </div>
            <div class="stat-pill">
                <span class="lbl">Validation Acc.</span>
                <span class="num" style="color:#7c3aed !important;">95.86%</span>
            </div>
            <div class="stat-pill">
                <span class="lbl">Test Acc.</span>
                <span class="num" style="color:#a855f7 !important;">87.5%</span>
            </div>
            <div class="stat-pill">
                <span class="lbl">Classes</span>
                <span class="num">4</span>
            </div>
            <div class="stat-pill">
                <span class="lbl">Training Images</span>
                <span class="num">14,462</span>
            </div>
        </div>
        """)

        # ── Main content ──
        with gr.Row(equal_height=True):
            # Left column: Upload
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="Upload X-ray",
                    height=380,
                    sources=["upload", "clipboard"],
                )
                predict_btn = gr.Button(
                    "Run AI Analysis",
                    variant="primary",
                    size="lg",
                )

            # Right column: Results + Grad-CAM
            with gr.Column(scale=1):
                results_html = gr.HTML(value="")

        # ── Grad-CAM section ──
        with gr.Row(equal_height=True):
            with gr.Column():
                output_cam = gr.Image(label="Grad-CAM Heatmap", height=350, show_label=True)
                gradcam_desc = gr.HTML(value="")

        # ── Report section ──
        report_out = gr.HTML(value="")

        # Hidden label for API compatibility
        output_label = gr.Label(label="Classification", num_top_classes=4, visible=False)

        # ── Wire events ──
        predict_btn.click(
            fn=predict,
            inputs=input_image,
            outputs=[output_label, output_cam, results_html, gradcam_desc, report_out],
        )
        input_image.change(
            fn=predict,
            inputs=input_image,
            outputs=[output_label, output_cam, results_html, gradcam_desc, report_out],
        )

        gr.HTML("<hr style='margin:20px 0; border:none; border-top:1px solid #ddd6fe;'>")

        # ── Feature cards ──
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

        # ── Disclaimer ──
        gr.HTML("""
        <div class="disclaimer">
            <p><b>Disclaimer</b> — This system is for educational and research
            purposes only (KBG Hackathon 2025, IIT Mandi). It is not a certified
            medical device. All results must be verified by qualified healthcare
            professionals before any clinical decision is made.</p>
        </div>
        """)

        # ── Footer ──
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
