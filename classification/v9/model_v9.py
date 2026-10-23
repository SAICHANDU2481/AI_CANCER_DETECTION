"""
================================================================================
V9 Architecture: Multi-Candidate Vision Transformer with Attention Pooling & Multi-Source Fusion
================================================================================
Implements modular Vision Transformer architectures for Arms A through G:
- Arm A: Top-1 Single Candidate Local ROI (768 -> 256 -> 2)
- Arm B: K=3 Candidates + Mean Pooling (768 -> 256 -> 2)
- Arm C: K=5 Candidates + Mean Pooling (768 -> 256 -> 2)
- Arm D: K=5 Candidates + Quality-Gated Candidate Attention (768 -> 256 -> 2)
- Arm E: K=5 Candidates + Attention + Full Mammogram Context (1536 -> 768 -> 256 -> 2)
- Arm F: K=5 Candidates + Full Mammo + High-Res Calcification (2304 -> 768 -> 256 -> 2)
- Arm G: K=5 Candidates + Full Mammo + Calc + Context ROI (3072 -> 768 -> 256 -> 2)

Shared ViT Encoder: vit_base_patch16_224 (768-dim) across all visual pathways.
================================================================================
"""

import math
import logging
from typing import Optional, Tuple, Dict, Any, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

logger = logging.getLogger("ViT_V9")


class CandidateQualityAttention(nn.Module):
    """
    Quality-Gated Candidate Attention Module.
    Combines 768-dim visual embedding with 7-dim normalized proposal quality vector:
    [mean_prob, max_prob, agree, calc_resp, contrast, detail, breast_cons]
    Computes normalized softmax attention weights across K proposals.
    """
    def __init__(self, embed_dim: int = 768, quality_dim: int = 7, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.quality_dim = quality_dim
        
        self.visual_proj = nn.Linear(embed_dim, hidden_dim)
        self.quality_proj = nn.Linear(quality_dim, hidden_dim)
        self.score_net = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z: torch.Tensor, q: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Visual candidate embeddings (B, K, embed_dim)
            q: Proposal quality features (B, K, quality_dim)
        Returns:
            z_pooled: Attention-aggregated visual embedding (B, embed_dim)
            weights: Normalized attention weights (B, K)
        """
        B, K, D = z.shape
        h_vis = self.visual_proj(z)  # (B, K, hidden_dim)
        
        if q is not None and q.shape[-1] == self.quality_dim:
            h_qual = self.quality_proj(q)  # (B, K, hidden_dim)
            h = h_vis + h_qual
        else:
            h = h_vis
            
        scores = self.score_net(h).squeeze(-1)  # (B, K)
        weights = torch.softmax(scores, dim=-1)  # (B, K)
        
        # Weighted sum of candidate features
        z_pooled = torch.bmm(weights.unsqueeze(1), z).squeeze(1)  # (B, D)
        return z_pooled, weights


class V9MultiCandidateClassifier(nn.Module):
    """
    Comprehensive V9 Multi-Candidate Vision Transformer Classifier.
    Supports Arms A through G with shared ViT backbone, candidate attention,
    and multi-pathway fusion heads.
    """
    def __init__(
        self,
        arm_type: str = 'Arm_D',
        k: int = 5,
        backbone_name: str = 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
        num_classes: int = 2,
        dropout_rate: float = 0.25,
        pretrained: bool = True
    ):
        super().__init__()
        self.arm_type = arm_type
        self.k = k
        self.num_classes = num_classes
        
        # Shared ViT backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        self.embed_dim = self.backbone.num_features  # 768
        
        # Configure pathways based on Arm
        self.use_attention = arm_type in ['Arm_D', 'Arm_E', 'Arm_F', 'Arm_G']
        self.use_full_mammo = arm_type in ['Arm_E', 'Arm_F', 'Arm_G']
        self.use_calc_patch = arm_type in ['Arm_F', 'Arm_G']
        self.use_context_roi = arm_type == 'Arm_G'
        
        # Attention Module
        if self.use_attention:
            self.attention_module = CandidateQualityAttention(embed_dim=self.embed_dim, quality_dim=7, hidden_dim=256)
        else:
            self.attention_module = None
            
        # Calculate Total Feature Dimension
        total_dim = self.embed_dim  # Candidate ROI pooled (768)
        if self.use_full_mammo:
            total_dim += self.embed_dim  # Full mammo (768) -> 1536
        if self.use_calc_patch:
            total_dim += self.embed_dim  # Calcification patch (768) -> 2304
        if self.use_context_roi:
            total_dim += self.embed_dim  # Context ROI (768) -> 3072
            
        self.total_dim = total_dim
        
        # Classification Fusion Head
        self.fusion_norm = nn.LayerNorm(total_dim)
        if total_dim > 768:
            self.fusion_head = nn.Sequential(
                nn.Linear(total_dim, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(768, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.7),
                nn.Linear(256, num_classes)
            )
        else:
            self.fusion_head = nn.Sequential(
                nn.Linear(total_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )

    def freeze_backbone(self):
        """Freezes entire ViT backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 2):
        """Unfreezes only the last n transformer blocks of the ViT backbone."""
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

    def encode_tokens(self, img_tensor: torch.Tensor, chunk_size: int = 8) -> torch.Tensor:
        """
        Passes a (N, 3, 224, 224) tensor through the ViT backbone to obtain 768-dim embeddings.
        Applies batch chunking to prevent GPU activation memory spikes.
        """
        if img_tensor is None:
            return None
        N = img_tensor.shape[0]
        if N <= chunk_size:
            return self.backbone(img_tensor)
            
        chunks = []
        for i in range(0, N, chunk_size):
            chunk_input = img_tensor[i:i + chunk_size]
            chunk_out = self.backbone(chunk_input)
            chunks.append(chunk_out)
        return torch.cat(chunks, dim=0)

    def forward(
        self,
        candidate_crops: torch.Tensor,
        candidate_qualities: Optional[torch.Tensor] = None,
        full_mammo: Optional[torch.Tensor] = None,
        calc_patch: Optional[torch.Tensor] = None,
        context_roi: Optional[torch.Tensor] = None,
        chunk_size: int = 8
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            candidate_crops: (B, K, 3, 224, 224) candidate tensors
            candidate_qualities: (B, K, 7) proposal quality features
            full_mammo: (B, 3, 224, 224) full mammogram image
            calc_patch: (B, 3, 224, 224) calcification patch
            context_roi: (B, 3, 224, 224) context ROI
        Returns:
            logits: (B, 2) classification logits
            attention_weights: (B, K) attention distribution or None
        """
        B, K, C, H, W = candidate_crops.shape
        
        # Flatten candidates for batch encoding: (B * K, 3, 224, 224)
        flat_cands = candidate_crops.contiguous().view(B * K, C, H, W)
        flat_embeds = self.encode_tokens(flat_cands, chunk_size=chunk_size)  # (B * K, 768)
        cand_embeds = flat_embeds.view(B, K, self.embed_dim)  # (B, K, 768)
        
        # Aggregate candidate representations
        attention_weights = None
        if self.arm_type == 'Arm_A':
            # Single top-1 candidate
            z_roi = cand_embeds[:, 0, :]  # (B, 768)
        elif self.use_attention and self.attention_module is not None:
            z_roi, attention_weights = self.attention_module(cand_embeds, candidate_qualities)
        else:
            # Simple mean pooling (Arms B, C)
            z_roi = torch.mean(cand_embeds, dim=1)  # (B, 768)
            
        # Collect representations for fusion
        fusion_components = [z_roi]
        
        if self.use_full_mammo and full_mammo is not None:
            full_embed = self.encode_tokens(full_mammo, chunk_size=chunk_size)  # (B, 768)
            fusion_components.append(full_embed)
            
        if self.use_calc_patch and calc_patch is not None:
            calc_embed = self.encode_tokens(calc_patch, chunk_size=chunk_size)  # (B, 768)
            fusion_components.append(calc_embed)
            
        if self.use_context_roi and context_roi is not None:
            context_embed = self.encode_tokens(context_roi, chunk_size=chunk_size)  # (B, 768)
            fusion_components.append(context_embed)
            
        # Multi-source concatenation
        if len(fusion_components) > 1:
            fused_feat = torch.cat(fusion_components, dim=-1)  # (B, total_dim)
        else:
            fused_feat = fusion_components[0]
            
        fused_norm = self.fusion_norm(fused_feat)
        logits = self.fusion_head(fused_norm)
        
        return logits, attention_weights
