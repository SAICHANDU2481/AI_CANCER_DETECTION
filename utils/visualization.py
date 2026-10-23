import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve

# Set style for publication quality figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


def plot_preprocessing_comparison(
    original: np.ndarray,
    wiener: np.ndarray,
    enhanced: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots a 3-panel side-by-side comparison: Original -> Wiener Filtered -> Contrast Enhanced.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("1. Original Mammogram", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(wiener, cmap="gray")
    axes[1].set_title("2. Wiener Filtered (Denoised)", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(enhanced, cmap="gray")
    axes[2].set_title("3. CLAHE Enhanced", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_segmentation_prediction(
    image: np.ndarray,
    pred_mask: np.ndarray,
    true_mask: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots segmentation results: Image, Predicted Mask, and Overlay with Bounding Box.
    """
    num_panels = 4 if true_mask is not None else 3
    fig, axes = plt.subplots(1, num_panels, figsize=(5 * num_panels, 5))
    
    # 1. Preprocessed Input
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Preprocessed Input", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    idx = 1
    # 2. Ground truth if available
    if true_mask is not None:
        axes[idx].imshow(true_mask, cmap="Blues")
        axes[idx].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")
        axes[idx].axis("off")
        idx += 1
        
    # 3. Predicted Mask
    axes[idx].imshow(pred_mask, cmap="Reds")
    axes[idx].set_title("U-KAN Predicted Mask", fontsize=11, fontweight="bold")
    axes[idx].axis("off")
    idx += 1
    
    # 4. Overlay with Bounding Box
    axes[idx].imshow(image, cmap="gray")
    axes[idx].imshow(pred_mask, cmap="Reds", alpha=0.45)
    if bbox is not None:
        ymin, xmin, ymax, xmax = bbox
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor="cyan", facecolor="none", linestyle="--"
        )
        axes[idx].add_patch(rect)
    axes[idx].set_title("Overlay + Extracted ROI BBox", fontsize=11, fontweight="bold")
    axes[idx].axis("off")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ["Benign", "Malignant"],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots publication-ready Confusion Matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    ax.set_title("Test Set Confusion Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: Optional[float] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots Receiver Operating Characteristic (ROC) curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    auc_label = f"ViT (AUC = {roc_auc:.3f})" if roc_auc is not None else "ViT ROC"
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=auc_label)
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title("ROC Curve - Breast Cancer Classification", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title_prefix: str = "Model",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots training and validation loss and performance metric curves across epochs.
    """
    keys = [k for k in history.keys() if not k.startswith("val_")]
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4.5))
    if len(keys) == 1:
        axes = [axes]
        
    epochs = range(1, len(history[keys[0]]) + 1)
    
    for ax, metric in zip(axes, keys):
        train_metric = history[metric]
        val_metric = history.get(f"val_{metric}", None)
        
        ax.plot(epochs, train_metric, label=f"Train {metric.capitalize()}", color="#1f77b4", lw=2)
        if val_metric is not None:
            ax.plot(epochs, val_metric, label=f"Val {metric.capitalize()}", color="#ff7f0e", lw=2, linestyle="--")
            
        ax.set_xlabel("Epoch", fontsize=10, fontweight="bold")
        ax.set_ylabel(metric.capitalize(), fontsize=10, fontweight="bold")
        ax.set_title(f"{title_prefix} {metric.capitalize()} Progression", fontsize=11, fontweight="bold")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_attention_overlay(
    roi: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.55,
    colormap: str = "jet",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots the ROI, Attention Heatmap, and Attention Blended Overlay.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # 1. Cropped ROI
    axes[0].imshow(roi, cmap="gray" if len(roi.shape) == 2 or roi.shape[-1] == 1 else None)
    axes[0].set_title("Extracted Lesion ROI", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    # 2. Raw Attention Map
    im1 = axes[1].imshow(attention_map, cmap=colormap)
    axes[1].set_title("ViT Self-Attention Map", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Blended Overlay
    axes[2].imshow(roi, cmap="gray" if len(roi.shape) == 2 or roi.shape[-1] == 1 else None)
    axes[2].imshow(attention_map, cmap=colormap, alpha=alpha)
    axes[2].set_title("Explainability Attention Overlay", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
