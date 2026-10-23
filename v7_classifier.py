"""
================================================================================
V7 Architecture: Multi-Scale Top-K + Full-Mammogram Vision Transformer
================================================================================
Fuses full-mammogram global context (1 x 768) with Top-K multi-scale ROI lesion
representations (4 x 768 = 3072) + calcification patch (1 x 768)
for a comprehensive 4608-dim latent representation.

Top-K Candidate Aggregation per scale:
- Candidate 1..K Local (B, K, 768)   -> Aggregated Local (B, 768)
- Candidate 1..K Medium (B, K, 768)  -> Aggregated Medium (B, 768)
- Candidate 1..K Large (B, K, 768)   -> Aggregated Large (B, 768)
- Candidate 1..K Context (B, K, 768) -> Aggregated Context (B, 768)

Total Fused Vector: 4608 -> 1024 -> 512 -> 128 -> 2
Sequential/chunked shared-weight ViT encoding to guarantee safe 4GB GPU operation.
================================================================================
"""

import math
import logging
from typing import Optional, Tuple, Dict, Any, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ViT_V7")


class CandidateSoftmaxAggregator(nn.Module):
    """
    Continuous Softmax attention weighting across K candidate proposals.
    Input: (B, K, D) -> Output: (B, D)
    """
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, D)
        scores = self.score_net(x)  # (B, K, 1)
        weights = torch.softmax(scores, dim=1)  # (B, K, 1)
        out = torch.sum(x * weights, dim=1)  # (B, D)
        return out


class MultiScaleViTV7(nn.Module):
    """
    V7 Multi-Scale Top-K ViT with Full-Mammogram Contextual Fusion.
    """
    def __init__(
        self,
        backbone_name: str = 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
        num_classes: int = 2,
        use_full_image: bool = True,
        use_calc_patch: bool = True,
        scales_used: Tuple[str, ...] = ('local', 'medium', 'large', 'context'),
        dropout_rate: float = 0.25,
        pretrained: bool = True,
        k: int = 3,
        chunk_size: int = 32
    ):
        super().__init__()
        self.use_full_image = use_full_image
        self.use_calc_patch = use_calc_patch
        self.scales_used = tuple(scales_used)
        self.k = k
        self.chunk_size = chunk_size

        # Shared ViT backbone
        logger.info(f"Instantiating shared ViT backbone: {backbone_name} (pretrained={pretrained})")
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        self.embed_dim = self.backbone.num_features  # 768 for vit_base

        # Top-K Aggregation Modules per scale
        self.aggregators = nn.ModuleDict()
        for s in self.scales_used:
            self.aggregators[s] = CandidateSoftmaxAggregator(self.embed_dim)

        # Feature dimension calculations
        num_scales = len(self.scales_used)
        total_roi_dim = num_scales * self.embed_dim  # 4 * 768 = 3072
        total_dim = total_roi_dim + (self.embed_dim if use_full_image else 0) + (self.embed_dim if use_calc_patch else 0)
        self.total_dim = total_dim

        # Deep MLP Classification Head with LayerNorm and GELU
        self.fusion_norm = nn.LayerNorm(total_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )

    def freeze_backbone(self):
        """Freezes entire ViT backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("ViT Backbone completely FROZEN.")

    def unfreeze_last_n_blocks(self, n: int = 2):
        """
        Unfreezes only the last n transformer blocks of the ViT backbone.
        """
        self.freeze_backbone()
        if n <= 0:
            return

        total_blocks = len(self.backbone.blocks)
        unfreeze_start = max(0, total_blocks - n)
        for i in range(unfreeze_start, total_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

        logger.info(f"Unfroze last {n} of {total_blocks} ViT blocks (blocks {unfreeze_start} to {total_blocks-1}).")

    def encode_tokens_chunked(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Passes a tensor of shape (N, 3, 224, 224) through shared ViT in chunks to prevent VRAM spikes.
        """
        N = img_tensor.shape[0]
        if N <= self.chunk_size:
            return self.backbone(img_tensor)
            
        chunks = []
        for i in range(0, N, self.chunk_size):
            chunk = img_tensor[i:i + self.chunk_size]
            chunks.append(self.backbone(chunk))
        return torch.cat(chunks, dim=0)

    def aggregate_candidates(self, cand_tensor: torch.Tensor, scale_name: str) -> torch.Tensor:
        """
        cand_tensor: (B, K, 3, 224, 224)
        Returns: (B, 768)
        """
        B, K, C, H, W = cand_tensor.shape
        stacked = cand_tensor.view(B * K, C, H, W)
        tokens = self.encode_tokens_chunked(stacked)
        tokens_bk = tokens.view(B, K, self.embed_dim)
        return self.aggregators[scale_name](tokens_bk)

    def forward(
        self,
        candidate_crops_dict: Dict[str, torch.Tensor],
        full_image_tensor: Optional[torch.Tensor] = None,
        calc_patch_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward classification pass through MultiScaleViTV7.
        """
        scale_feats = []
        for s in self.scales_used:
            cand_input = candidate_crops_dict[s]
            if cand_input.dim() == 4:
                # Shape (B, 3, 224, 224) single candidate
                f = self.encode_tokens_chunked(cand_input)
            elif cand_input.dim() == 5 and cand_input.shape[1] == 1:
                f = self.encode_tokens_chunked(cand_input.squeeze(1))
            else:
                f = self.aggregate_candidates(cand_input, scale_name=s)
            scale_feats.append(f)

        parts = []
        if self.use_full_image:
            if full_image_tensor is None:
                raise ValueError("full_image_tensor required when use_full_image=True")
            full_feat = self.encode_tokens_chunked(full_image_tensor)
            parts.append(full_feat)

        if scale_feats:
            roi_fused = torch.cat(scale_feats, dim=-1)
            parts.append(roi_fused)

        if self.use_calc_patch:
            if calc_patch_tensor is None:
                raise ValueError("calc_patch_tensor required when use_calc_patch=True")
            calc_feat = self.encode_tokens_chunked(calc_patch_tensor)
            parts.append(calc_feat)

        joint_feat = torch.cat(parts, dim=-1)
        normed = self.fusion_norm(joint_feat)
        logits = self.fusion_mlp(normed)
        return logits
