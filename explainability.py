"""
explainability.py — Grad-CAM & Attention Map Visualisation
===========================================================

Provides:
  1. **Grad-CAM** heatmaps for the CNN backbone (EfficientNet-B3 / ResNet50).
  2. **Attention rollout** maps for the ViT head.
  3. A helper that generates sample overlays for every fracture class and
     saves them to ``results/explainability/``.

Usage
-----
    python explainability.py --config config.yaml --checkpoint checkpoints/model.pth

The script picks one correctly-classified test image per class and saves:
    results/explainability/{class_name}_gradcam.png
    results/explainability/{class_name}_attention.png
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from data_loader import load_config, set_seed, get_dataloaders
from model import build_model


# ------------------------------------------------------------------ #
#  Device helper                                                      #
# ------------------------------------------------------------------ #

def get_device(config):
    requested = config.get("device", "auto")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


# ------------------------------------------------------------------ #
#  Grad-CAM for the CNN backbone                                      #
# ------------------------------------------------------------------ #

class GradCAM:
    """
    Grad-CAM for the CNN branch of HybridCNNViT.

    We hook into the *last convolutional feature map* of the CNN backbone.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._gradients = None
        self._activations = None
        self._hook_handles = []

        # Register hooks on the CNN backbone's last feature layer
        # timm models expose .forward_features(); we hook the backbone itself.
        target = model.cnn  # timm model
        # The last block that produces feature maps differs by arch.
        # We use register_forward_hook on the whole backbone to grab output.
        h1 = target.register_forward_hook(self._save_activation)
        # For gradients we need a *tensor* hook — we'll attach after forward.
        self._hook_handles.append(h1)

    def _save_activation(self, module, inp, out):
        # out may be (B, d) after global pool; we need *before* pool.
        # So instead we use model.cnn_features directly.
        pass

    def generate(self, image: torch.Tensor, class_idx: int | None = None):
        """
        Parameters
        ----------
        image : (1, 3, H, W) tensor
        class_idx : target class (default = predicted class)

        Returns
        -------
        cam : (H, W) numpy array in [0, 1]
        pred_class : int
        """
        self.model.eval()
        image = image.to(self.device).requires_grad_(True)

        # Get CNN feature maps (before global pool)
        feat_map = self.model.cnn_features(image)  # (1, C, h, w)

        # If ViT returns (B, N, D), we still need a forward pass for logits
        # So we do a full forward and use the feat_map for cam.
        logits = self.model(image)  # (1, num_classes)
        pred_class = logits.argmax(dim=1).item()
        if class_idx is None:
            class_idx = pred_class

        # Backprop w.r.t. target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        # Gradients w.r.t. feature map
        grads = image.grad  # won't work — need feat_map grad
        # Re-do with feat_map needing grad
        return self._generate_internal(image, class_idx)

    def _generate_internal(self, image, class_idx):
        self.model.eval()
        self.model.zero_grad()

        x = image.detach().to(self.device).requires_grad_(False)
        # Forward CNN features
        feat_map = self.model.cnn.forward_features(x)  # (1, C, h, w) or (1, N, C) for ViT-like
        if feat_map.dim() == 3:
            # Some timm CNN models output (B, N, C) — unlikely for EfficientNet/ResNet
            pass

        feat_map.retain_grad()

        # Global pool + rest of pipeline
        cnn_feat = self.model.cnn.forward_head(feat_map, pre_logits=True)  # (1, d_cnn)
        if cnn_feat.dim() > 2:
            cnn_feat = F.adaptive_avg_pool2d(cnn_feat, 1).flatten(1)

        vit_feat = self.model.vit(x)  # (1, d_vit)
        fused = self.model.fusion(cnn_feat, vit_feat)
        logits = self.model.classifier(fused)

        pred_class = logits.argmax(dim=1).item()
        target = class_idx if class_idx is not None else pred_class

        score = logits[0, target]
        score.backward()

        grads = feat_map.grad  # (1, C, h, w)
        acts = feat_map.detach()

        if grads is None:
            # Fallback: return uniform map
            return np.ones((224, 224), dtype=np.float32) * 0.5, pred_class

        weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP over spatial
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, pred_class

    def __del__(self):
        for h in self._hook_handles:
            h.remove()


# ------------------------------------------------------------------ #
#  Attention rollout for ViT                                          #
# ------------------------------------------------------------------ #

def get_vit_attention_map(model, image: torch.Tensor, device):
    """
    Simple attention rollout for the ViT branch.

    Works with timm ViT models that have ``.blocks`` containing
    ``.attn.attn_drop`` (standard timm structure).

    Returns
    -------
    attn_map : (H, W) numpy array in [0, 1]
    """
    model.eval()
    x = image.to(device)

    attentions = []

    def _hook(module, inp, out):
        # out is the attention weight tensor (B, heads, N, N) in timm
        attentions.append(out.detach().cpu())

    hooks = []
    vit = model.vit
    # timm ViTs: vit.blocks[i].attn  — the softmax output is inside attn
    # We hook attn.attn_drop which receives the attention weights
    if hasattr(vit, "blocks"):
        for blk in vit.blocks:
            if hasattr(blk, "attn"):
                # Hook the attention's softmax output
                h = blk.attn.attn_drop.register_forward_hook(_hook)
                hooks.append(h)

    with torch.no_grad():
        _ = model.vit(x)

    for h in hooks:
        h.remove()

    if not attentions:
        return np.ones((224, 224), dtype=np.float32) * 0.5

    # Attention rollout
    # Each attn: (B, heads, N, N) — average over heads
    result = None
    for attn in attentions:
        attn = attn.squeeze(0).mean(dim=0)  # (N, N)
        attn = attn + torch.eye(attn.size(0))  # residual
        attn = attn / attn.sum(dim=-1, keepdim=True)
        if result is None:
            result = attn
        else:
            result = attn @ result

    # Take CLS token row (index 0), discard CLS token itself
    mask = result[0, 1:]  # (num_patches,)
    num_patches = mask.shape[0]
    side = int(num_patches ** 0.5)
    if side * side != num_patches:
        side = int(np.ceil(num_patches ** 0.5))
        # Pad
        pad = side * side - num_patches
        mask = torch.cat([mask, torch.zeros(pad)])
    mask = mask.reshape(side, side).numpy()
    mask = cv2.resize(mask, (224, 224))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


# ------------------------------------------------------------------ #
#  Overlay helper                                                     #
# ------------------------------------------------------------------ #

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def _denormalize(tensor_img):
    """Convert (3, H, W) normalised tensor back to (H, W, 3) uint8."""
    img = tensor_img.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def overlay_heatmap(image_np, heatmap, alpha=0.4):
    """Overlay a [0,1] heatmap on an RGB image."""
    cmap = plt.cm.jet(heatmap)[..., :3]
    cmap = (cmap * 255).astype(np.uint8)
    blended = cv2.addWeighted(image_np, 1 - alpha, cmap, alpha, 0)
    return blended


# ------------------------------------------------------------------ #
#  Generate sample explanations                                       #
# ------------------------------------------------------------------ #

def generate_explanations(config_path="config.yaml", checkpoint_path=None,
                          out_dir="results/explainability", max_per_class=1):
    """
    For each fracture class, find a correctly-classified test image and
    save Grad-CAM + attention overlays.
    """
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    device = get_device(config)

    _, _, test_loader, class_names = get_dataloaders(config)
    num_classes = len(class_names)
    config["num_classes"] = num_classes

    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            config.get("checkpoint_dir", "checkpoints"), "model.pth"
        )

    model = build_model(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    os.makedirs(out_dir, exist_ok=True)

    gradcam = GradCAM(model, device)
    found = {i: 0 for i in range(num_classes)}

    for images, labels in test_loader:
        for i in range(images.size(0)):
            label = labels[i].item()
            if found[label] >= max_per_class:
                continue

            img_tensor = images[i].unsqueeze(0).to(device)

            # Quick check: is it correctly classified?
            with torch.no_grad():
                pred = model(img_tensor).argmax(dim=1).item()
            if pred != label:
                continue

            cn = class_names[label]
            img_np = _denormalize(images[i])

            # Grad-CAM
            cam, _ = gradcam._generate_internal(img_tensor, label)
            overlay_gc = overlay_heatmap(img_np, cam)

            # Attention map
            attn = get_vit_attention_map(model, img_tensor, device)
            overlay_at = overlay_heatmap(img_np, attn)

            # Save
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np)
            axes[0].set_title(f"Original — {cn}")
            axes[0].axis("off")
            axes[1].imshow(overlay_gc)
            axes[1].set_title("Grad-CAM (CNN)")
            axes[1].axis("off")
            axes[2].imshow(overlay_at)
            axes[2].set_title("Attention (ViT)")
            axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{cn}_explainability.png"), dpi=150)
            plt.close()

            # Also save individual maps
            cv2.imwrite(
                os.path.join(out_dir, f"{cn}_gradcam.png"),
                cv2.cvtColor(overlay_gc, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(out_dir, f"{cn}_attention.png"),
                cv2.cvtColor(overlay_at, cv2.COLOR_RGB2BGR),
            )

            found[label] += 1
            print(f"[explainability] Saved {cn} (label={label})")

        if all(v >= max_per_class for v in found.values()):
            break

    print(f"[explainability] All outputs → {out_dir}/")


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out-dir", default="results/explainability")
    args = parser.parse_args()

    generate_explanations(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
    )
