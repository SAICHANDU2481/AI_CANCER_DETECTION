import math
import argparse
import logging
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("UKAN")


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Linear Layer.
    Replaces fixed activation functions with learnable univariate B-spline basis functions:
        phi(x) = w_b * b(x) + w_s * sum_k (c_k * B_k(x))
    where:
        - b(x) = SiLU(x) (base non-linear function)
        - B_k(x) are B-spline basis functions evaluated on a uniform grid
        - c_k are learnable spline coefficients
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

        # Generate uniform spline knot grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        )
        self.register_buffer("grid", grid)

        # Base linear weight: b(x) = silu(x)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        # Spline linear weight
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
        """
        Evaluates Cox-de Boor recursion for B-spline basis functions.
        x: (batch_size, in_features)
        Returns: (batch_size, in_features, grid_size + spline_order)
        """
        grid = self.grid.to(x.device)
        x_expanded = x.unsqueeze(-1) # (B, in_feat, 1)

        # Order 0 B-splines
        bases = ((x_expanded >= grid[:-1]) & (x_expanded < grid[1:])).to(x.dtype)

        # Recursive Cox-de Boor for higher order B-splines
        for k in range(1, self.spline_order + 1):
            denom1 = grid[k:-1] - grid[:-(k + 1)]
            denom2 = grid[k + 1:] - grid[1:-k]
            
            # Prevent division by zero
            denom1 = torch.where(denom1 == 0, torch.ones_like(denom1), denom1)
            denom2 = torch.where(denom2 == 0, torch.ones_like(denom2), denom2)

            term1 = (x_expanded - grid[:-(k + 1)]) / denom1 * bases[:, :, :-1]
            term2 = (grid[k + 1:] - x_expanded) / denom2 * bases[:, :, 1:]
            bases = term1 + term2

        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation of KAN Linear transformation.
        Supports both 2D (B, in_features) and 3D tokenized (B, N, in_features) tensors.
        """
        orig_shape = x.shape
        if x.dim() > 2:
            x_flat = x.reshape(-1, self.in_features)
        else:
            x_flat = x

        # 1. Base activation
        base_out = F.linear(F.silu(x_flat), self.base_weight)

        # 2. Spline basis activation
        spline_basis = self.b_splines(x_flat)  # (B_total, in_features, num_bases)
        # Vectorized GEMM contraction: (B_total, in_features * num_bases) x (out_features, in_features * num_bases)^T
        spline_out = F.linear(
            spline_basis.reshape(x_flat.shape[0], -1),
            self.spline_weight.reshape(self.out_features, -1)
        )

        out = base_out + spline_out

        if len(orig_shape) > 2:
            out = out.reshape(*orig_shape[:-1], self.out_features)
        return out


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> SiLU) * 2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class KANBottleneck(nn.Module):
    """
    Bottleneck layer leveraging Kolmogorov-Arnold representations on flattened spatial tokens.
    """
    def __init__(self, channels: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.kan1 = KANLinear(channels, channels, grid_size=grid_size, spline_order=spline_order)
        self.norm = nn.LayerNorm(channels)
        self.kan2 = KANLinear(channels, channels, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C) # (B, HW, C)
        
        residual = tokens
        h = self.kan1(tokens)
        h = self.norm(h)
        h = self.kan2(h)
        out_tokens = h + residual
        
        out = out_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


class UpBlock(nn.Module):
    """Upsampling followed by concatenation with skip connection and DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 if odd dimensions differ
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UKAN(nn.Module):
    """
    U-KAN: Kolmogorov-Arnold Network Integrated U-Net Architecture
    for Medical Mammogram Semantic Lesion Segmentation.
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

        # Encoder stages
        self.inc = DoubleConv(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 2, f * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 4, f * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f * 8, f * 16))

        # KAN Bottleneck stage
        self.kan_bottleneck = KANBottleneck(
            channels=f * 16,
            grid_size=grid_size,
            spline_order=spline_order
        )

        # Decoder stages with skip connections
        self.up1 = UpBlock(f * 16 + f * 8, f * 8)
        self.up2 = UpBlock(f * 8 + f * 4, f * 4)
        self.up3 = UpBlock(f * 4 + f * 2, f * 2)
        self.up4 = UpBlock(f * 2 + f, f)

        # Final 1x1 convolution projection head to logit mask
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # KAN Bottleneck
        bottleneck = self.kan_bottleneck(x5)

        # Decoder with skip connections
        x = self.up1(bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


def build_ukan_from_config(config_dict: Optional[dict] = None) -> UKAN:
    """Instantiates a UKAN model according to configuration parameters."""
    if config_dict is None:
        return UKAN()
    
    seg_cfg = config_dict.get("segmentation", {})
    kan_cfg = seg_cfg.get("kan", {})
    
    return UKAN(
        in_channels=seg_cfg.get("in_channels", 1),
        out_channels=seg_cfg.get("out_channels", 1),
        init_features=seg_cfg.get("init_features", 32),
        grid_size=kan_cfg.get("grid_size", 5),
        spline_order=kan_cfg.get("spline_order", 3)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test UKAN model architecture and parameter count.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    model = UKAN(in_channels=1, out_channels=1, init_features=16)
    dummy_input = torch.randn(args.batch_size, 1, args.size, args.size)
    
    with torch.no_grad():
        out = model(dummy_input)
        
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print("U-KAN ARCHITECTURE VERIFICATION")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Trainable parameters: {num_params:,}")
    print("=" * 60)
