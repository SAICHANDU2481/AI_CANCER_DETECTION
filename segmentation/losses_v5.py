"""
================================================================================
V5 Segmentation Losses: Small-Object & Calcification Aware Formulations
================================================================================
Loss functions tailored for mammographic lesion segmentation with extreme foreground/
background class imbalance (tiny calcifications occupying <0.1% of mammogram pixels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Smooth Dice loss for binary segmentation."""
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal loss focusing on hard, ambiguous lesion boundary pixels."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * ((1 - p_t) ** self.gamma) * bce

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


class TverskyLoss(nn.Module):
    """
    Tversky Loss for asymmetric penalty on False Negatives (higher beta = higher recall).
    alpha: FP weight, beta: FN weight. (beta=0.7, alpha=0.3 prioritizes lesion detection).
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs_flat) * targets_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class BCEDiceLoss(nn.Module):
    """Standard weighted BCE + Dice loss."""
    def __init__(self, pos_weight: float = 3.0, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.bce.pos_weight.device != logits.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


class FocalDiceLoss(nn.Module):
    """Focal Loss + Dice Loss for microcalcifications and hard examples."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, focal_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_weight * self.focal(logits, targets) + self.dice_weight * self.dice(logits, targets)


class TverskyDiceLoss(nn.Module):
    """Tversky Loss (FN-focused) + Dice Loss."""
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, tversky_weight: float = 0.6, dice_weight: float = 0.4):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.dice = DiceLoss()
        self.tversky_weight = tversky_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.tversky_weight * self.tversky(logits, targets) + self.dice_weight * self.dice(logits, targets)


def get_segmentation_loss_v5(loss_type: str = 'bce_dice', **kwargs) -> nn.Module:
    """Factory function for V5 segmentation losses."""
    loss_type = loss_type.lower()
    if loss_type in ['bce_dice', 'bce+dice']:
        return BCEDiceLoss(**kwargs)
    elif loss_type in ['focal_dice', 'focal+dice']:
        return FocalDiceLoss(**kwargs)
    elif loss_type in ['tversky_dice', 'tversky+dice']:
        return TverskyDiceLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
