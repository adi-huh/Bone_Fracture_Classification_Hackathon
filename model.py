"""
model.py — Hybrid CNN + Vision Transformer Architecture
========================================================

Architecture Overview
---------------------
This model fuses **local** features from a CNN backbone with **global**
attention-based features from a Vision Transformer head.

    ┌────────────────────────────────────────────────────────────────┐
    │                        Input (B, 3, 224, 224)                  │
    │                                                                │
    │  ┌──────────────────┐              ┌────────────────────────┐  │
    │  │  CNN Backbone     │              │  ViT / Swin Head       │  │
    │  │  (EfficientNet-B3 │              │  (vit_small_patch16    │  │
    │  │   or ResNet50,    │              │   _224, ImageNet       │  │
    │  │   ImageNet wts)   │              │   pretrained)          │  │
    │  └────────┬─────────┘              └──────────┬─────────────┘  │
    │           │ cnn_feat (d_cnn)                   │ vit_feat (d_vit)│
    │           └──────────┬─────────────────────────┘               │
    │                      │                                         │
    │            ┌─────────▼──────────┐                              │
    │            │   Fusion Layer     │                              │
    │            │  (Cross-Attention  │                              │
    │            │   or Concat + FC)  │                              │
    │            └─────────┬──────────┘                              │
    │                      │ fused (d_fused)                         │
    │            ┌─────────▼──────────┐                              │
    │            │  Classification    │                              │
    │            │  Linear→ReLU→      │                              │
    │            │  Dropout(0.3)→     │                              │
    │            │  Linear(num_cls)   │                              │
    │            └────────────────────┘                              │
    └────────────────────────────────────────────────────────────────┘

Allowed pre-training:
  • ImageNet weights ONLY (no fracture / bone X-ray weights).

Usage
-----
    from model import HybridCNNViT
    model = HybridCNNViT(num_classes=7, backbone="efficientnet_b3")
"""

import torch
import torch.nn as nn
import timm


# ------------------------------------------------------------------ #
#  Cross-Attention Fusion                                             #
# ------------------------------------------------------------------ #

class CrossAttentionFusion(nn.Module):
    """
    Light-weight cross-attention between CNN and ViT features.

    Q = CNN features,  K = V = ViT features  (single head for speed).
    """

    def __init__(self, d_cnn: int, d_vit: int, d_out: int = 512):
        super().__init__()
        self.proj_q = nn.Linear(d_cnn, d_out)
        self.proj_k = nn.Linear(d_vit, d_out)
        self.proj_v = nn.Linear(d_vit, d_out)
        self.scale  = d_out ** -0.5
        self.out_proj = nn.Linear(d_out, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor):
        # cnn_feat: (B, d_cnn)  vit_feat: (B, d_vit)
        # unsqueeze to (B, 1, d) so we can do matmul
        Q = self.proj_q(cnn_feat).unsqueeze(1)   # (B, 1, d_out)
        K = self.proj_k(vit_feat).unsqueeze(1)    # (B, 1, d_out)
        V = self.proj_v(vit_feat).unsqueeze(1)    # (B, 1, d_out)

        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, 1, 1)
        attn = attn.softmax(dim=-1)
        out = (attn @ V).squeeze(1)                     # (B, d_out)
        out = self.out_proj(out)
        out = self.norm(out + cnn_feat[:, :out.size(-1)] if cnn_feat.size(-1) == out.size(-1) else out)
        return out                                       # (B, d_out)


# ------------------------------------------------------------------ #
#  Concatenation Fusion (simpler alternative)                         #
# ------------------------------------------------------------------ #

class ConcatFusion(nn.Module):
    def __init__(self, d_cnn: int, d_vit: int, d_out: int = 512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_cnn + d_vit, d_out),
            nn.LayerNorm(d_out),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor):
        return self.fc(torch.cat([cnn_feat, vit_feat], dim=-1))


# ------------------------------------------------------------------ #
#  Hybrid Model                                                       #
# ------------------------------------------------------------------ #

class HybridCNNViT(nn.Module):
    """
    Hybrid CNN + ViT classifier for multi-class bone fracture detection.

    Parameters
    ----------
    num_classes : int
        Number of fracture categories.
    backbone : str
        CNN backbone name understood by ``timm`` (e.g. "efficientnet_b3",
        "resnet50").
    vit_name : str
        ViT model name understood by ``timm`` (e.g.
        "vit_small_patch16_224", "swin_tiny_patch4_window7_224").
    fusion : str
        "cross_attention" or "concatenation".
    dropout : float
        Dropout probability in classification head.
    pretrained : bool
        Whether to load ImageNet-pretrained weights (required True).
    """

    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = "efficientnet_b3",
        vit_name: str = "vit_small_patch16_224",
        fusion: str = "cross_attention",
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.vit_name = vit_name

        # ---------- CNN backbone ----------
        self.cnn = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d_cnn = self.cnn.num_features  # feature dim after global pool

        # ---------- ViT / Swin head ----------
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        d_vit = self.vit.num_features

        # ---------- Fusion ----------
        d_fused = 512
        if fusion == "cross_attention":
            self.fusion = CrossAttentionFusion(d_cnn, d_vit, d_fused)
        else:
            self.fusion = ConcatFusion(d_cnn, d_vit, d_fused)

        # ---------- Classification head ----------
        self.classifier = nn.Sequential(
            nn.Linear(d_fused, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        print(
            f"[model] HybridCNNViT  |  CNN={backbone}(d={d_cnn})  "
            f"ViT={vit_name}(d={d_vit})  fusion={fusion}  "
            f"classes={num_classes}"
        )

    # ---- feature accessors (used by explainability) ---- #

    def cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return CNN backbone feature map *before* global pool."""
        return self.cnn.forward_features(x)

    def vit_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return ViT feature tokens (B, N, d_vit)."""
        return self.vit.forward_features(x)

    # ---- freeze / unfreeze for transfer learning ---- #

    def freeze_backbones(self):
        """Freeze CNN and ViT backbones — only fusion + classifier train."""
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[model] Backbones frozen. Trainable params: {trainable:,}")

    def unfreeze_backbones(self):
        """Unfreeze everything for end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[model] All layers unfrozen. Trainable params: {trainable:,}")

    # ---- forward ---- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn(x)          # (B, d_cnn)
        vit_feat = self.vit(x)          # (B, d_vit)
        fused    = self.fusion(cnn_feat, vit_feat)   # (B, d_fused)
        logits   = self.classifier(fused)            # (B, num_classes)
        return logits


# ------------------------------------------------------------------ #
#  Factory helper                                                     #
# ------------------------------------------------------------------ #

def build_model(config: dict) -> HybridCNNViT:
    """Instantiate model from a config dict (loaded from config.yaml)."""
    return HybridCNNViT(
        num_classes=config.get("num_classes", 7),
        backbone=config.get("model_backbone", "efficientnet_b3"),
        vit_name=config.get("vit_model", "vit_small_patch16_224"),
        fusion=config.get("fusion_method", "cross_attention"),
        dropout=config.get("dropout", 0.3),
        pretrained=True,
    )
