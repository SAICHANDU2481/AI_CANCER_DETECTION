"""
================================================================================
V3-A Multi-Scale Shared-Weight Vision Transformer (ViT) Architecture
================================================================================
Integrates:
- Shared-weight pretrained ViT backbone (vit_base_patch16_224)
- 3 contextual scale branches (Local 1.0x+10%, Medium 1.5x, Large 2.0x)
- Multi-scale feature concatenation (768 * 3 = 2304 dimensions)
- Bottleneck Fusion & Classification Head (2304 -> 512 -> 128 -> 2)
- Built-in Staged Fine-Tuning controllers
"""

import logging
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import timm

logger = logging.getLogger("ModelV3")


class MultiScaleMachineRoiViT(nn.Module):
    """
    Multi-Scale Machine-ROI Vision Transformer with Shared-Weight Feature Extraction
    and Contextual Feature Fusion for Mammographic Malignancy Classification.
    """
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout_rate_1: float = 0.20,
        dropout_rate_2: float = 0.10,
        num_scales: int = 3
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_scales = num_scales
        self.embed_dim = 768
        self.fused_dim = self.embed_dim * self.num_scales  # 2304

        # 1. Pretrained Shared-Weight ViT Backbone (num_classes=0 returns pool/CLS embedding)
        logger.info(f"Loading pretrained shared ViT backbone: '{backbone_name}'...")
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0
        )

        # 2. Contextual Fusion & Classification Head
        self.fusion_head = nn.Sequential(
            nn.Linear(self.fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=dropout_rate_1),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_rate_2),
            nn.Linear(128, num_classes)
        )

        self._init_head_weights()

    def _init_head_weights(self):
        """Kaiming normal initialization for classification head layers."""
        for m in self.fusion_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def freeze_backbone(self):
        """Stage 1: Freeze entire ViT backbone; train only fusion head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.fusion_head.parameters():
            param.requires_grad = True
        logger.info("ViT backbone completely FROZEN (Stage 1 Head Warmup).")

    def unfreeze_top_blocks(self, num_blocks: int = 2):
        """
        Stage 2: Unfreeze the top `num_blocks` transformer blocks and final norm layer.
        """
        self.freeze_backbone()

        # Unfreeze final LayerNorm if present
        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        # Unfreeze top N transformer blocks
        if hasattr(self.backbone, "blocks"):
            total_blocks = len(self.backbone.blocks)
            unfreeze_start = max(0, total_blocks - num_blocks)
            for i in range(unfreeze_start, total_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True
            logger.info(f"Unfroze top {num_blocks} transformer blocks (Blocks {unfreeze_start} to {total_blocks - 1}).")

    def unfreeze_all_blocks(self):
        """Stage 3 (Optional): Unfreeze entire backbone for full end-to-end tuning."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Unfroze ALL ViT backbone parameters.")

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 0.01
    ) -> List[Dict[str, Any]]:
        """
        Returns separate parameter groups with differential learning rates for staged tuning.
        """
        backbone_decay, backbone_no_decay = [], []
        head_decay, head_no_decay = [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            is_bias_or_norm = "bias" in name or "norm" in name or "LayerNorm" in name

            if "backbone" in name:
                if is_bias_or_norm:
                    backbone_no_decay.append(param)
                else:
                    backbone_decay.append(param)
            else:
                if is_bias_or_norm:
                    head_no_decay.append(param)
                else:
                    head_decay.append(param)

        param_groups = [
            {"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
            {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay},
            {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0}
        ]
        # Filter out empty parameter groups
        return [g for g in param_groups if len(g["params"]) > 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        Input: [B, 3, 3, 224, 224] (B samples, 3 scale branches, 3 channels, 224, 224)
        Output: [B, 2] logits
        """
        B, num_scales, C, H, W = x.shape
        assert num_scales == self.num_scales, f"Expected {self.num_scales} scales, got {num_scales}"

        # Reshape to [B*3, 3, 224, 224]
        x_flat = x.reshape(B * num_scales, C, H, W)

        # Pass through shared-weight ViT backbone: [B*3, 768]
        features_flat = self.backbone(x_flat)

        # Reshape to [B, 3*768] = [B, 2304]
        features_fused = features_flat.reshape(B, num_scales * self.embed_dim)

        # Pass through fusion & classification head: [B, 2]
        logits = self.fusion_head(features_fused)
        return logits
