import os, sys, logging, torch, torch.nn as nn, timm
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger('ModelV3B')

class CrossAttentionFusionBlock(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.10):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query_token: torch.Tensor, context_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if query_token.dim() == 2:
            query = query_token.unsqueeze(1)
        else:
            query = query_token
        attn_out, attn_weights = self.mha(query=query, key=context_tokens, value=context_tokens)
        x = self.norm1(query + attn_out)
        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out)
        fused_vec = out.squeeze(1)
        return fused_vec, attn_weights

class MultiScaleViTV3B(nn.Module):
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.20,
        fusion_mode: str = 'cross_attention',
        scales_used: Tuple[str, ...] = ('local', 'medium', 'large'),
        num_heads: int = 8
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.scales_used = tuple(scales_used)
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True
        )
        self.embed_dim = self.backbone.num_features

        if self.fusion_mode == 'cross_attention':
            self.cross_attn_block = CrossAttentionFusionBlock(embed_dim=self.embed_dim, num_heads=num_heads, dropout=dropout / 2.0)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim, num_classes)
            )
        elif self.fusion_mode == 'mlp':
            num_scales = len(self.scales_used)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim * num_scales, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes)
            )
        elif self.fusion_mode == 'single_roi':
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim, num_classes)
            )
        else:
            raise ValueError(f'Unsupported fusion_mode: {fusion_mode}')

        logger.info(f'Initialized MultiScaleViTV3B (fusion={fusion_mode}, scales={self.scales_used}, embed_dim={self.embed_dim})')

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone completely frozen.')

    def unfreeze_top_blocks(self, num_blocks: int = 2):
        for param in self.backbone.parameters():
            param.requires_grad = False
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, 'blocks'):
            total_blocks = len(self.backbone.blocks)
            for i in range(total_blocks - num_blocks, total_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True
            logger.info(f'Unfrozen top {num_blocks} blocks (blocks {total_blocks - num_blocks} to {total_blocks - 1}) and norm.')
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info('Unfrozen full backbone (blocks attribute not found).')

    def get_parameter_groups(self, backbone_lr: float, head_lr: float) -> List[Dict[str, Any]]:
        backbone_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr}
        ]

    def forward(
        self,
        local_tensor: torch.Tensor,
        medium_tensor: Optional[torch.Tensor] = None,
        large_tensor: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ):
        local_feat = self.backbone(local_tensor)
        attn_weights = None

        if self.fusion_mode == 'single_roi':
            fused_feat = local_feat
        elif self.fusion_mode == 'mlp':
            feats = [local_feat]
            if 'medium' in self.scales_used and medium_tensor is not None:
                feats.append(self.backbone(medium_tensor))
            if 'large' in self.scales_used and large_tensor is not None:
                feats.append(self.backbone(large_tensor))
            fused_feat = torch.cat(feats, dim=-1)
        elif self.fusion_mode == 'cross_attention':
            context_list = []
            if 'medium' in self.scales_used and medium_tensor is not None:
                context_list.append(self.backbone(medium_tensor))
            if 'large' in self.scales_used and large_tensor is not None:
                context_list.append(self.backbone(large_tensor))
            if len(context_list) > 0:
                context_tensor = torch.stack(context_list, dim=1)
                fused_feat, attn_weights = self.cross_attn_block(local_feat, context_tensor)
            else:
                fused_feat = local_feat

        logits = self.classifier(fused_feat)
        if return_attn:
            return logits, attn_weights
        return logits