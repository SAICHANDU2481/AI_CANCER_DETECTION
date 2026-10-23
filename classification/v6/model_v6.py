"""
================================================================================
V6 Architecture: Multi-Scale Top-K + Full-Mammogram Vision Transformer
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
logger = logging.getLogger("ViT_V6")


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


class MultiScaleViTV6(nn.Module):
    """
    V6 Multi-Scale Top-K ViT with Full-Mammogram Contextual Fusion.
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
        k: int = 3
    ):
        super().__init__()
        self.use_full_image = use_full_image
        self.use_calc_patch = use_calc_patch
        self.scales_used = tuple(scales_used)
        self.k = k

        # Shared ViT backbone
        logger.info(f"Loading shared ViT backbone: {backbone_name} (pretrained={pretrained})")
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

    def unfreeze_last_n_blocks(self, n: int = 2):
        """
        Unfreezes only the last n transformer blocks of the ViT backbone.
        """
        self.freeze_backbone()
        if n <= 0:
            logger.info("ViT Backbone completely FROZEN (0 blocks fine-tuned).")
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

    def encode_tokens(self, img_tensor: torch.Tensor, chunk_size: int = 8) -> torch.Tensor:
        """
        Passes a (N, 3, 224, 224) tensor through the ViT backbone to obtain 768-dim embeddings.
        Applies batch chunking (chunk_size) to prevent activation memory spikes.
        """
        if img_tensor is None:
            return None
        N = img_tensor.shape[0]
        if N <= chunk_size:
            return self.backbone(img_tensor)
        feats = []
        for i in range(0, N, chunk_size):
            chunk = img_tensor[i:i + chunk_size]
            feats.append(self.backbone(chunk))
        return torch.cat(feats, dim=0)

    def aggregate_candidates(self, cand_input: Union[List[torch.Tensor], torch.Tensor], scale_name: str, chunk_size: int = 8) -> torch.Tensor:
        """
        cand_input: List of K tensors [ (B, 3, 224, 224), ... ] OR single tensor (B, K, 3, 224, 224).
        Returns aggregated tensor of shape (B, 768).
        """
        if isinstance(cand_input, torch.Tensor) and cand_input.dim() == 5:
            B, K, C, H, W = cand_input.shape
            stacked_inputs = cand_input.view(B * K, C, H, W)
            tokens = self.encode_tokens(stacked_inputs, chunk_size=chunk_size)
            tokens_bk = tokens.view(B, K, self.embed_dim)
        else:
            candidate_tensors = list(cand_input)
            B = candidate_tensors[0].shape[0]
            K = len(candidate_tensors)
            stacked_inputs = torch.cat(candidate_tensors, dim=0)  # (B*K, 3, 224, 224)
            tokens = self.encode_tokens(stacked_inputs, chunk_size=chunk_size)            # (B*K, 768)
            tokens_bk = tokens.view(K, B, self.embed_dim).permute(1, 0, 2)  # (B, K, 768)

        return self.aggregators[scale_name](tokens_bk)

    def forward(
        self,
        candidate_crops_dict: Dict[str, Union[List[torch.Tensor], torch.Tensor]],
        full_image_tensor: Optional[torch.Tensor] = None,
        calc_patch_tensor: Optional[torch.Tensor] = None,
        chunk_size: int = 8
    ) -> torch.Tensor:
        """
        Forward classification pass through MultiScaleViTV6 with chunked execution.
        """
        scale_feats = []
        for s in self.scales_used:
            cand_input = candidate_crops_dict[s]
            if isinstance(cand_input, list) and len(cand_input) == 1:
                f = self.encode_tokens(cand_input[0], chunk_size=chunk_size)
            elif isinstance(cand_input, torch.Tensor) and cand_input.dim() == 4:
                f = self.encode_tokens(cand_input, chunk_size=chunk_size)
            elif isinstance(cand_input, torch.Tensor) and cand_input.dim() == 5 and cand_input.shape[1] == 1:
                f = self.encode_tokens(cand_input.squeeze(1), chunk_size=chunk_size)
            else:
                f = self.aggregate_candidates(cand_input, scale_name=s, chunk_size=chunk_size)
            scale_feats.append(f)

        parts = []
        if self.use_full_image:
            if full_image_tensor is None:
                raise ValueError("full_image_tensor must be provided when use_full_image=True")
            full_feat = self.encode_tokens(full_image_tensor, chunk_size=chunk_size)  # (B, 768)
            parts.append(full_feat)

        if scale_feats:
            roi_fused = torch.cat(scale_feats, dim=-1)  # (B, total_roi_dim)
            parts.append(roi_fused)

        if self.use_calc_patch:
            if calc_patch_tensor is None:
                raise ValueError("calc_patch_tensor must be provided when use_calc_patch=True")
            calc_feat = self.encode_tokens(calc_patch_tensor, chunk_size=chunk_size)  # (B, 768)
            parts.append(calc_feat)

        joint_feat = torch.cat(parts, dim=-1)  # (B, total_dim)
        normed = self.fusion_norm(joint_feat)
        logits = self.fusion_mlp(normed)
        return logits
