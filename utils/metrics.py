import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

class DiceLoss(nn.Module):
    """
    Computes soft Dice loss for binary segmentation.
    Loss = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1).float()
        
        intersection = (probs_flat * targets_flat).sum()
        total = probs_flat.sum() + targets_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1.0 - dice


class CombinedSegmentationLoss(nn.Module):
    """
    Weighted combination of Binary Cross Entropy (BCE) and Dice Loss.
    """
    def __init__(self, dice_weight: float = 0.6, bce_weight: float = 0.4, pos_weight: torch.Tensor = None):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dice_l = self.dice_loss(logits, targets)
        bce_l = self.bce_loss(logits, targets.float())
        total_l = self.dice_weight * dice_l + self.bce_weight * bce_l
        return total_l, dice_l, bce_l


def calculate_segmentation_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Computes medical image segmentation metrics on binarized masks.

    Args:
        y_pred: Predicted probability map or binary mask (values in [0, 1]).
        y_true: Ground truth binary mask (values in {0, 1}).
        threshold: Binarization cutoff threshold.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Dictionary containing Dice, IoU, Precision, Recall, Sensitivity, Specificity.
    """
    pred_bin = (y_pred >= threshold).astype(np.uint8).flatten()
    true_bin = (y_true >= 0.5).astype(np.uint8).flatten()

    tp = np.sum((pred_bin == 1) & (true_bin == 1))
    fp = np.sum((pred_bin == 1) & (true_bin == 0))
    fn = np.sum((pred_bin == 0) & (true_bin == 1))
    tn = np.sum((pred_bin == 0) & (true_bin == 0))

    # Dice coefficient (F1 score for segmentation)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    
    # Intersection over Union (Jaccard Index)
    iou = (tp + eps) / (tp + fp + fn + eps)
    
    # Precision / Positive Predictive Value
    precision = (tp + eps) / (tp + fp + eps)
    
    # Recall / Sensitivity / True Positive Rate
    recall = (tp + eps) / (tp + fn + eps)
    sensitivity = recall
    
    # Specificity / True Negative Rate
    specificity = (tn + eps) / (tn + fp + eps)

    # Pixel Accuracy
    pixel_accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "pixel_accuracy": float(pixel_accuracy)
    }


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, Any]:
    """
    Calculates comprehensive classification metrics for Benign (0) vs Malignant (1).

    Args:
        y_true: Ground truth integer class labels (0 or 1).
        y_pred: Predicted integer class labels (0 or 1).
        y_prob: Predicted probability for the positive class (Malignant).

    Returns:
        Dictionary of real calculated classification metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp + 1e-7))
    else:
        specificity = 0.0

    roc_auc = None
    if y_prob is not None:
        try:
            # Requires at least one positive and one negative sample
            if len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, y_prob))
            else:
                roc_auc = None
        except Exception:
            roc_auc = None

    report = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"], output_dict=True, zero_division=0)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "sensitivity": float(rec), # Sensitivity is identical to Recall
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
