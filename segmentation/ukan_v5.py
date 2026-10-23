"""
================================================================================
V5 Architecture: High-Resolution Multi-Scale Kolmogorov-Arnold U-Net (U-KAN V5)
================================================================================
Investigating high-resolution multi-scale feature representation and detailed
lesion localization in mammography, specifically for small lesions and calcifications.

Key Elements:
- KANLinear with learnable B-spline basis functions (Cox-de Boor recursion)
- Shallow downsampling & High-Resolution detail extraction branch
- Multi-Scale Dilated Residual Detail Blocks (r=1, 2, 4)
- Cross-Scale Feature Fusion (CSFF) with Gated Channel Attention
- High-Resolution decoder with multi-scale skip connections
- Deep supervision / multi-scale output head support
"""

import math
import logging
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("UKAN_V5")


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
    """Two consecutive conv-bn-silu blocks with residual connection."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class MultiScaleDetailBlock(nn.Module):
    """Multi-scale dilated convolutional block (r=1, 2, 4) for microcalcifications and fine boundaries."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = max(out_ch // 3, 8)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True)
        )
        total_mid = mid_ch * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(total_mid, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        fused = self.fuse(torch.cat([b1, b2, b3], dim=1))
        return fused + self.shortcut(x)


class CrossScaleFeatureFusion(nn.Module):
    """Gated Cross-Scale Feature Fusion with channel attention."""
    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, max(channels // 4, 8), kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(channels // 4, 8), channels * 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, f_global: torch.Tensor, f_detail: torch.Tensor) -> torch.Tensor:
        cat_feat = torch.cat([f_global, f_detail], dim=1)
        gates = self.attn(cat_feat)
        gated_feat = cat_feat * gates
        fused = self.proj(gated_feat)
        return fused + f_global


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


class UpBlockV5(nn.Module):
    """Upsampling followed by concatenation with multi-scale skip connection and DoubleConv."""
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


class MultiScaleUKANV5(nn.Module):
    """
    V5 High-Resolution Multi-Scale U-KAN.
    Dual-stream high-resolution detail and global context pathways with KAN spline bottleneck.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 32,
        grid_size: int = 5,
        spline_order: int = 3
    ):
        super().__init__()
        f = init_features

        # Stage 1: Full Input Resolution (e.g., 512x512 or 768x768)
        self.global_inc = DoubleConv(in_channels, f)
        self.detail_inc = MultiScaleDetailBlock(in_channels, f)
        self.csff1 = CrossScaleFeatureFusion(f)

        # Stage 2: 1/2 Resolution
        self.down1 = nn.MaxPool2d(2)
        self.global_enc2 = DoubleConv(f, f * 2)
        self.detail_enc2 = MultiScaleDetailBlock(f, f * 2)
        self.csff2 = CrossScaleFeatureFusion(f * 2)

        # Stage 3: 1/4 Resolution
        self.down2 = nn.MaxPool2d(2)
        self.global_enc3 = DoubleConv(f * 2, f * 4)
        self.detail_enc3 = MultiScaleDetailBlock(f * 2, f * 4)
        self.csff3 = CrossScaleFeatureFusion(f * 4)

        # Stage 4: 1/8 Resolution
        self.down3 = nn.MaxPool2d(2)
        self.global_enc4 = DoubleConv(f * 4, f * 8)
        self.detail_enc4 = MultiScaleDetailBlock(f * 4, f * 8)
        self.csff4 = CrossScaleFeatureFusion(f * 8)

        # Bottleneck Stage: 1/16 Resolution
        self.down4 = nn.MaxPool2d(2)
        self.bottleneck_conv = DoubleConv(f * 8, f * 16)
        self.kan_bottleneck = KANBottleneck(
            channels=f * 16,
            grid_size=grid_size,
            spline_order=spline_order
        )

        # Decoder Stages with Multi-Scale Skip Connections
        self.up1 = UpBlockV5(f * 16 + f * 8, f * 8)
        self.up2 = UpBlockV5(f * 8 + f * 4, f * 4)
        self.up3 = UpBlockV5(f * 4 + f * 2, f * 2)
        self.up4 = UpBlockV5(f * 2 + f, f)

        # High-resolution output projection
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        g1 = self.global_inc(x)
        d1 = self.detail_inc(x)
        f1 = self.csff1(g1, d1)

        # Stage 2
        p1 = self.down1(f1)
        g2 = self.global_enc2(p1)
        d2 = self.detail_enc2(p1)
        f2 = self.csff2(g2, d2)

        # Stage 3
        p2 = self.down2(f2)
        g3 = self.global_enc3(p2)
        d3 = self.detail_enc3(p2)
        f3 = self.csff3(g3, d3)

        # Stage 4
        p3 = self.down3(f3)
        g4 = self.global_enc4(p3)
        d4 = self.detail_enc4(p3)
        f4 = self.csff4(g4, d4)

        # Bottleneck
        p4 = self.down4(f4)
        b = self.bottleneck_conv(p4)
        b_kan = self.kan_bottleneck(b)

        # Decoder
        u1 = self.up1(b_kan, f4)
        u2 = self.up2(u1, f3)
        u3 = self.up3(u2, f2)
        u4 = self.up4(u3, f1)

        logits = self.outc(u4)
        return logits


def build_ukan_v5(
    in_channels: int = 1,
    out_channels: int = 1,
    init_features: int = 32,
    grid_size: int = 5,
    spline_order: int = 3
) -> MultiScaleUKANV5:
    return MultiScaleUKANV5(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
        grid_size=grid_size,
        spline_order=spline_order
    )
