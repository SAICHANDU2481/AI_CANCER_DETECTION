"""
================================================================================
V6 Architecture: Multi-Scale Kolmogorov-Arnold Network with Deep Supervision
                 and Dedicated High-Resolution Detail Pathway (U-KAN V6)
================================================================================
Primary Goal:
- Substantially enhance microcalcification localization, small-lesion recall,
  and lesion margin boundary fidelity without excessive parameter bloat.

Key Innovations in V6:
1. KANLinear B-spline Basis Functions (Cox-de Boor recursion + learnable grid)
2. Multi-Scale Dilated Residual Blocks (parallel r=1, 2, 4 receptive fields)
3. Dedicated High-Resolution Micro-Detail Branch (preserving high-frequency calcifications)
4. Cross-Scale Gated Feature Fusion (CSFF)
5. Multi-Stage Deep Supervision (auxiliary heads at 1/4 and 1/8 decoder stages)
6. Memory-efficient channel progression: 1 -> 32 -> 64 -> 128 -> 256 -> KAN Bottleneck
================================================================================
"""

import math
import logging
from typing import Optional, Tuple, Dict, Any, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("UKAN_V6")


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Linear Layer.
    phi(x) = w_b * b(x) + w_s * sum_k (c_k * B_k(x))
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.out_features, self.in_features, self.grid_size + self.spline_order) - 0.5) * self.scale_noise
            self.spline_weight.copy_(noise * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.grid.to(x.device)
        x_expanded = x.unsqueeze(-1)

        bases = ((x_expanded >= grid[:-1]) & (x_expanded < grid[1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            denom1 = grid[k:-1] - grid[:-(k + 1)]
            denom2 = grid[k + 1:] - grid[1:-k]

            denom1 = torch.where(denom1 == 0, torch.ones_like(denom1), denom1)
            denom2 = torch.where(denom2 == 0, torch.ones_like(denom2), denom2)

            term1 = (x_expanded - grid[:-(k + 1)]) / denom1 * bases[:, :, :-1]
            term2 = (grid[k + 1:] - x_expanded) / denom2 * bases[:, :, 1:]
            bases = term1 + term2

        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() > 2:
            x_flat = x.reshape(-1, self.in_features)
        else:
            x_flat = x

        base_out = F.linear(F.silu(x_flat), self.base_weight)
        spline_basis = self.b_splines(x_flat)
        spline_out = F.linear(
            spline_basis.reshape(x_flat.shape[0], -1),
            self.spline_weight.reshape(self.out_features, -1)
        )
        out = base_out + spline_out

        if len(orig_shape) > 2:
            out = out.reshape(*orig_shape[:-1], self.out_features)
        return out


class DoubleConv(nn.Module):
    """Standard Convolution Block: [Conv -> BatchNorm -> GELU] x 2 with residual connection."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class MultiScaleDilatedBlockV6(nn.Module):
    """
    Multi-Scale Dilated Residual Block:
    Extracts features simultaneously at 3 dilation rates:
    - Branch 1: r=1 (dense local morphology)
    - Branch 2: r=2 (peritumoral boundary context)
    - Branch 3: r=4 (macro parenchymal architecture)
    Fused with Gated Channel Attention.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = max(16, out_ch // 3)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU()
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU()
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU()
        )
        cat_ch = mid_ch * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(cat_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        # Gated Channel Squeeze-and-Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, max(8, out_ch // 8), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(8, out_ch // 8), out_ch, kernel_size=1),
            nn.Sigmoid()
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.b1(x)
        out2 = self.b2(x)
        out3 = self.b3(x)
        cat = torch.cat([out1, out2, out3], dim=1)
        fused = self.fuse(cat)
        gated = fused * self.se(fused)
        return self.act(gated + self.shortcut(x))


class DedicatedDetailBranch(nn.Module):
    """
    High-Resolution Micro-Detail Pathway.
    Extracts high-frequency spatial gradients, microcalcification edges,
    and subtle textural transitions without aggressive spatial downsampling.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.detail_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False), # Depthwise
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),                          # Pointwise
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.detail_conv(x)


class CrossScaleFeatureFusion(nn.Module):
    """Cross-Scale Feature Fusion with Gated Channel Attention."""
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels * 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, g: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([g, d], dim=1)
        gated = cat * self.gate(cat)
        return self.proj(gated)


class KANBottleneck(nn.Module):
    """KAN bottleneck operating on flattened spatial tokens."""
    def __init__(self, channels: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.kan1 = KANLinear(channels, channels, grid_size=grid_size, spline_order=spline_order)
        self.norm = nn.LayerNorm(channels)
        self.kan2 = KANLinear(channels, channels, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        residual = tokens
        h = self.kan1(tokens)
        h = self.norm(h)
        h = self.kan2(h)
        out_tokens = h + residual

        out = out_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


class UpBlockV6(nn.Module):
    """Bilinear upsampling followed by multi-scale skip connection concatenation and DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MultiScaleUKANV6(nn.Module):
    """
    V6 Multi-Scale High-Resolution U-KAN with Deep Supervision and Detail Branch.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 32,
        grid_size: int = 5,
        spline_order: int = 3,
        use_detail_branch: bool = True,
        use_multi_scale_blocks: bool = True,
        deep_supervision: bool = True
    ):
        super().__init__()
        self.use_detail_branch = use_detail_branch
        self.use_multi_scale_blocks = use_multi_scale_blocks
        self.deep_supervision = deep_supervision
        f = init_features

        # -------------------------------------------------------------
        # 1. ENCODER STAGES
        # -------------------------------------------------------------
        # Stage 1: Full Input Resolution (e.g., 512x512 or 768x768)
        self.enc1 = DoubleConv(in_channels, f)
        self.msdb1 = MultiScaleDilatedBlockV6(f, f) if use_multi_scale_blocks else DoubleConv(f, f)
        self.csff1 = CrossScaleFeatureFusion(f)
        
        # Dedicated High-Resolution Detail Branch
        self.detail_branch = DedicatedDetailBranch(in_channels, f) if use_detail_branch else None

        # Stage 2: 1/2 Resolution
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(f, f * 2)
        self.msdb2 = MultiScaleDilatedBlockV6(f * 2, f * 2) if use_multi_scale_blocks else DoubleConv(f * 2, f * 2)
        self.csff2 = CrossScaleFeatureFusion(f * 2)

        # Stage 3: 1/4 Resolution
        self.down2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.msdb3 = MultiScaleDilatedBlockV6(f * 4, f * 4) if use_multi_scale_blocks else DoubleConv(f * 4, f * 4)
        self.csff3 = CrossScaleFeatureFusion(f * 4)

        # Stage 4: 1/8 Resolution
        self.down3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(f * 4, f * 8)
        self.msdb4 = MultiScaleDilatedBlockV6(f * 8, f * 8) if use_multi_scale_blocks else DoubleConv(f * 8, f * 8)
        self.csff4 = CrossScaleFeatureFusion(f * 8)

        # -------------------------------------------------------------
        # 2. BOTTLENECK STAGE (1/16 Resolution)
        # -------------------------------------------------------------
        self.down4 = nn.MaxPool2d(2)
        self.bottleneck_conv = DoubleConv(f * 8, f * 8)
        self.kan_bottleneck = KANBottleneck(
            channels=f * 8,
            grid_size=grid_size,
            spline_order=spline_order
        )

        # -------------------------------------------------------------
        # 3. DECODER STAGES
        # -------------------------------------------------------------
        self.up1 = UpBlockV6(f * 8 + f * 8, f * 8)
        self.up2 = UpBlockV6(f * 8 + f * 4, f * 4)
        self.up3 = UpBlockV6(f * 4 + f * 2, f * 2)
        
        # Stage 4 Decoder concatenation (with optional dedicated detail branch skip)
        dec4_in_ch = (f * 2 + f + f) if use_detail_branch else (f * 2 + f)
        self.up4 = UpBlockV6(dec4_in_ch, f)

        # -------------------------------------------------------------
        # 4. HEADS & DEEP SUPERVISION
        # -------------------------------------------------------------
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

        if deep_supervision:
            # Auxiliary segmentation head 1 (from 1/4 resolution u2)
            self.aux_head1 = nn.Conv2d(f * 4, out_channels, kernel_size=1)
            # Auxiliary segmentation head 2 (from 1/2 resolution u3)
            self.aux_head2 = nn.Conv2d(f * 2, out_channels, kernel_size=1)
        else:
            self.aux_head1 = None
            self.aux_head2 = None

    def forward(
        self,
        x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Forward pass.
        During training with deep supervision enabled: returns (main_logits, aux1_logits, aux2_logits).
        During eval/inference: returns main_logits.
        """
        # Dedicated Detail Pathway
        detail_feat = self.detail_branch(x) if self.use_detail_branch else None

        # Stage 1
        e1 = self.enc1(x)
        m1 = self.msdb1(e1)
        f1 = self.csff1(e1, m1)

        # Stage 2
        p1 = self.down1(f1)
        e2 = self.enc2(p1)
        m2 = self.msdb2(e2)
        f2 = self.csff2(e2, m2)

        # Stage 3
        p2 = self.down2(f2)
        e3 = self.enc3(p2)
        m3 = self.msdb3(e3)
        f3 = self.csff3(e3, m3)

        # Stage 4
        p3 = self.down3(f3)
        e4 = self.enc4(p3)
        m4 = self.msdb4(e4)
        f4 = self.csff4(e4, m4)

        # Bottleneck
        p4 = self.down4(f4)
        b = self.bottleneck_conv(p4)
        b_kan = self.kan_bottleneck(b)

        # Decoder
        u1 = self.up1(b_kan, f4)
        u2 = self.up2(u1, f3)
        u3 = self.up3(u2, f2)

        # Final decoder stage
        if self.use_detail_branch and detail_feat is not None:
            f1_combined = torch.cat([f1, detail_feat], dim=1)
            u4 = self.up4(u3, f1_combined)
        else:
            u4 = self.up4(u3, f1)

        logits_main = self.outc(u4)

        if self.training and self.deep_supervision and self.aux_head1 is not None and self.aux_head2 is not None:
            aux1 = self.aux_head1(u2)
            aux2 = self.aux_head2(u3)
            return logits_main, aux1, aux2

        return logits_main


def build_ukan_v6(
    in_channels: int = 1,
    out_channels: int = 1,
    init_features: int = 32,
    grid_size: int = 5,
    spline_order: int = 3,
    use_detail_branch: bool = True,
    use_multi_scale_blocks: bool = True,
    deep_supervision: bool = True
) -> MultiScaleUKANV6:
    return MultiScaleUKANV6(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
        grid_size=grid_size,
        spline_order=spline_order,
        use_detail_branch=use_detail_branch,
        use_multi_scale_blocks=use_multi_scale_blocks,
        deep_supervision=deep_supervision
    )
