"""
================================================================================
V6 Segmentation Losses: Boundary-Aware, Deep Supervision & Small-Lesion Adaptive
================================================================================
Strict Governance:
- Boundary Loss and Size Weights are computed from Ground-Truth ONLY during TRAINING.
- Zero Ground-Truth information is used at inference time.
- Fully configurable loss combinations for rigorous ablation testing.
================================================================================
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union


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
    """Focal loss focusing on hard, ambiguous lesion boundary and microcalcification pixels."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = alpha_t * ((1.0 - p_t) ** self.gamma) * bce

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


class TverskyLoss(nn.Module):
    """
    Tversky Loss for asymmetric penalty on False Negatives (higher beta = higher recall for small lesions).
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


class BoundaryLoss(nn.Module):
    """
    Differentiable surface/boundary loss approximation using morphological gradient on target masks.
    Penalizes contour boundary misalignments for spiculated masses and fine calcification clusters.
    """
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        # MaxPool2d - MinPool approximation for morphological dilation - erosion
        dilated = F.max_pool2d(mask, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        eroded = -F.max_pool2d(-mask, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        boundary = (dilated - eroded).clamp(0.0, 1.0)
        return boundary

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target_boundary = self._get_boundary(targets)
        pred_boundary = self._get_boundary(probs)

        # Smooth L1 / L2 loss on predicted boundary vs target boundary
        boundary_loss = F.smooth_l1_loss(pred_boundary, target_boundary)
        return boundary_loss


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
    """Focal Loss + Dice Loss for microcalcifications and hard boundary examples."""
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


class FocalTverskyBoundaryLoss(nn.Module):
    """
    Composite V6 Loss: Focal Loss (hard pixels) + Tversky Loss (small lesion recall) + Boundary Loss (contour crispness).
    """
    def __init__(
        self,
        focal_weight: float = 0.35,
        tversky_weight: float = 0.45,
        boundary_weight: float = 0.20,
        alpha_tversky: float = 0.3,
        beta_tversky: float = 0.7,
        alpha_focal: float = 0.75,
        gamma_focal: float = 2.0
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha_focal, gamma=gamma_focal)
        self.tversky = TverskyLoss(alpha=alpha_tversky, beta=beta_tversky)
        self.boundary = BoundaryLoss()
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_focal = self.focal(logits, targets)
        l_tversky = self.tversky(logits, targets)
        l_boundary = self.boundary(logits, targets)
        return (
            self.focal_weight * l_focal +
            self.tversky_weight * l_tversky +
            self.boundary_weight * l_boundary
        )


class DeepSupervisedCompositeLoss(nn.Module):
    """
    Wraps base segmentation loss across main and auxiliary multi-resolution decoder outputs.
    L_total = w_main * L(main, y) + w_aux1 * L(aux1, y) + w_aux2 * L(aux2, y)
    """
    def __init__(
        self,
        base_loss: nn.Module,
        w_main: float = 0.60,
        w_aux1: float = 0.25,
        w_aux2: float = 0.15
    ):
        super().__init__()
        self.base_loss = base_loss
        self.w_main = w_main
        self.w_aux1 = w_aux1
        self.w_aux2 = w_aux2

    def forward(
        self,
        preds: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor], Dict[str, torch.Tensor]],
        targets: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(preds, torch.Tensor):
            return self.base_loss(preds, targets)
        elif isinstance(preds, (tuple, list)):
            main_logits = preds[0]
            loss = self.w_main * self.base_loss(main_logits, targets)
            if len(preds) > 1 and preds[1] is not None:
                # Upsample aux1 if resolution is lower
                aux1 = preds[1]
                if aux1.shape[-2:] != targets.shape[-2:]:
                    aux1 = F.interpolate(aux1, size=targets.shape[-2:], mode='bilinear', align_corners=True)
                loss = loss + self.w_aux1 * self.base_loss(aux1, targets)
            if len(preds) > 2 and preds[2] is not None:
                # Upsample aux2 if resolution is lower
                aux2 = preds[2]
                if aux2.shape[-2:] != targets.shape[-2:]:
                    aux2 = F.interpolate(aux2, size=targets.shape[-2:], mode='bilinear', align_corners=True)
                loss = loss + self.w_aux2 * self.base_loss(aux2, targets)
            return loss
        elif isinstance(preds, dict):
            main_logits = preds['main']
            loss = self.w_main * self.base_loss(main_logits, targets)
            if 'aux1' in preds and preds['aux1'] is not None:
                aux1 = preds['aux1']
                if aux1.shape[-2:] != targets.shape[-2:]:
                    aux1 = F.interpolate(aux1, size=targets.shape[-2:], mode='bilinear', align_corners=True)
                loss = loss + self.w_aux1 * self.base_loss(aux1, targets)
            if 'aux2' in preds and preds['aux2'] is not None:
                aux2 = preds['aux2']
                if aux2.shape[-2:] != targets.shape[-2:]:
                    aux2 = F.interpolate(aux2, size=targets.shape[-2:], mode='bilinear', align_corners=True)
                loss = loss + self.w_aux2 * self.base_loss(aux2, targets)
            return loss
        return self.base_loss(preds, targets)


class SmallLesionWeightedLoss(nn.Module):
    """
    Optional Training-Only lesion size adaptive loss weighting.
    Small (<500 px): weight 1.35
    Medium (500-2000 px): weight 1.00
    Large (>2000 px): weight 0.85
    """
    def __init__(
        self,
        base_loss: nn.Module,
        small_weight: float = 1.35,
        medium_weight: float = 1.00,
        large_weight: float = 0.85,
        small_threshold: int = 500,
        large_threshold: int = 2000
    ):
        super().__init__()
        self.base_loss = base_loss
        self.small_weight = small_weight
        self.medium_weight = medium_weight
        self.large_weight = large_weight
        self.small_thresh = small_threshold
        self.large_thresh = large_threshold

    def forward(
        self,
        preds: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
        targets: torch.Tensor
    ) -> torch.Tensor:
        # Compute per-sample size category from ground-truth mask
        B = targets.shape[0]
        sample_losses = []

        for b in range(B):
            tgt_b = targets[b:b+1]
            if isinstance(preds, torch.Tensor):
                pred_b = preds[b:b+1]
            elif isinstance(preds, (tuple, list)):
                pred_b = tuple(p[b:b+1] if p is not None else None for p in preds)
            else:
                pred_b = preds

            # Target area in pixels
            area = float(tgt_b.sum().item())
            if area < self.small_thresh:
                w = self.small_weight
            elif area > self.large_thresh:
                w = self.large_weight
            else:
                w = self.medium_weight

            l_b = self.base_loss(pred_b, tgt_b)
            sample_losses.append(w * l_b)

        return torch.stack(sample_losses).mean()
