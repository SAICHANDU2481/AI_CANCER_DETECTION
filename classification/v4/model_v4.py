import torch
import torch.nn as nn
import timm
from typing import Dict, Any, List, Optional, Tuple

class MultiScaleViTV4(nn.Module):
    """
    V4 Multi-Scale Vision Transformer with Quality-Aware Fusion and Optional Quality-Gated Routing.
    
    Scales used:
      - Local ROI (1.0x / expanded)
      - Medium Context (1.5x)
      - Large Context (2.0x)
      - Surrounding Context (2.5x - 3.0x)
      
    Quality vector:
      - 16-dimensional continuous descriptor
      
    Fusion dimension:
      - 768 * num_scales + (16 if use_quality_vector else 0)
    """
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.20,
        scales_used: Tuple[str, ...] = ('local', 'medium', 'large', 'context'),
        use_quality_vector: bool = True,
        enable_gating: bool = False,
        gating_threshold: float = 0.40
    ):
        super().__init__()
        self.scales_used = tuple(scales_used)
        self.num_scales = len(self.scales_used)
        self.use_quality_vector = use_quality_vector
        self.enable_gating = enable_gating
        self.gating_threshold = gating_threshold
        self.num_classes = num_classes
        
        # Shared-weight ViT backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True
        )
        self.embed_dim = self.backbone.num_features  # 768
        
        # Visual feature concatenation dimension
        self.visual_dim = self.embed_dim * self.num_scales  # 768 * 4 = 3072
        self.fusion_in_dim = self.visual_dim + (16 if self.use_quality_vector else 0)  # 3088
        
        # Quality-aware fusion head
        self.head = nn.Sequential(
            nn.Linear(self.fusion_in_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Optional gated Path B head (for low-quality / fallback routing)
        if self.enable_gating:
            # Path B uses expanded + context (2 * 768 = 1536 + 16 = 1552)
            self.path_b_in_dim = (self.embed_dim * 2) + (16 if self.use_quality_vector else 0)
            self.head_path_b = nn.Sequential(
                nn.Linear(self.path_b_in_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes a batch of 3-channel crop tensors [B, 3, H, W] through the shared ViT backbone.
        Returns: feature embedding [B, 768]
        """
        return self.backbone(x)

    def forward(
        self,
        local_crop: Optional[torch.Tensor] = None,
        medium_crop: Optional[torch.Tensor] = None,
        large_crop: Optional[torch.Tensor] = None,
        context_crop: Optional[torch.Tensor] = None,
        quality_vector: Optional[torch.Tensor] = None,
        roi_quality_score: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through MultiScaleViTV4.
        """
        feature_list = []
        scale_tensors = {
            'local': local_crop,
            'medium': medium_crop,
            'large': large_crop,
            'context': context_crop
        }
        
        for scale_name in self.scales_used:
            t = scale_tensors.get(scale_name)
            if t is None:
                raise ValueError(f"Required scale '{scale_name}' was not provided to forward().")
            feat = self.extract_features(t)
            feature_list.append(feat)
            
        visual_feats = torch.cat(feature_list, dim=-1)  # [B, num_scales * 768]
        
        # Append 16-dimensional quality vector if configured
        if self.use_quality_vector:
            if quality_vector is None:
                raise ValueError("use_quality_vector is True but quality_vector was not provided.")
            if quality_vector.shape[-1] != 16:
                raise ValueError(f"Expected quality_vector with 16 features, got {quality_vector.shape}")
            fused_input = torch.cat([visual_feats, quality_vector], dim=-1)
        else:
            fused_input = visual_feats
            
        if self.enable_gating and roi_quality_score is not None:
            # Gated routing based on quality score
            logits_a = self.head(fused_input)
            
            # Path B features: local (or expanded) + context
            feat_loc = feature_list[0]
            feat_ctx = feature_list[-1]
            path_b_feats = torch.cat([feat_loc, feat_ctx], dim=-1)
            if self.use_quality_vector:
                path_b_feats = torch.cat([path_b_feats, quality_vector], dim=-1)
            logits_b = self.head_path_b(path_b_feats)
            
            # Routing condition: score >= threshold -> Path A, else Path B
            mask_a = (roi_quality_score >= self.gating_threshold).float().unsqueeze(-1)
            logits = mask_a * logits_a + (1.0 - mask_a) * logits_b
        else:
            logits = self.head(fused_input)
            
        return logits
