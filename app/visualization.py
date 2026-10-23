"""
================================================================================
Breast Cancer AI Analysis - Visualization & Rendering Module
================================================================================
Specialized visualization utilities for medical mammography AI analysis:
1. Probability field heatmaps & Segmentation overlays (Alpha blending)
2. Top-K Candidate proposal bounding box annotations & Tag labels
3. Concentric Multi-Scale RoI inspection panels
4. Vision Transformer Self-Attention Rollout overlays
5. Confusion Matrix & ROC/PR visual representations
================================================================================
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_segmentation_overlay(
    img_512: np.ndarray,
    prob_512: np.ndarray,
    proposals: List[Dict[str, Any]],
    threshold: float = 0.15
) -> np.ndarray:
    """
    Creates a medical visualization overlay:
    - Base mammogram in grayscale RGB
    - Jet colormap for predicted lesion probability field
    - Color-coded bounding boxes for candidate proposals with tags
    """
    base_u8 = (np.clip(img_512, 0.0, 1.0) * 255.0).astype(np.uint8)
    base_rgb = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2RGB)

    # Probability Heatmap
    prob_u8 = (np.clip(prob_512, 0.0, 1.0) * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)

    # Mask regions with probability above threshold for alpha blending
    mask = (prob_512 >= threshold).astype(np.float32)[:, :, np.newaxis]
    alpha = 0.45 * mask
    blended = (base_rgb * (1.0 - alpha) + heatmap * alpha).astype(np.uint8)

    # Bounding Box Colors: Cand 1 = Bright Coral/Red, Cand 2 = Amber/Gold, Cand 3 = Cyan
    colors = [
        (239, 68, 68),    # Coral Red (#EF4444)
        (245, 158, 11),   # Amber Gold (#F59E0B)
        (6, 182, 212),    # Cyan Teal (#06B6D4)
        (168, 85, 247),   # Purple (#A855F7)
        (34, 197, 94)     # Green (#22C55E)
    ]

    for idx, prop in enumerate(proposals[:len(colors)]):
        if 'bbox' not in prop:
            continue
        ymin, xmin, ymax, xmax = prop['bbox']
        color = colors[idx]
        thickness = 2 if idx > 0 else 3
        cv2.rectangle(blended, (xmin, ymin), (xmax, ymax), color, thickness)

        # Proposal Tag Label
        score_val = prop.get('score', 0.0)
        tag = f"ROI #{idx+1} ({score_val:.2f})"
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        tag_ymin = max(0, ymin - th - 6)
        cv2.rectangle(blended, (xmin, tag_ymin), (xmin + tw + 6, tag_ymin + th + 6), (15, 23, 42), -1)
        cv2.putText(blended, tag, (xmin + 3, tag_ymin + th + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

        # Center marker
        if 'center' in prop:
            cy, cx = int(round(prop['center'][0])), int(round(prop['center'][1]))
            cv2.circle(blended, (cx, cy), 3, color, -1)

    return blended


def create_attention_overlay(
    image_2d: np.ndarray,
    attn_map_2d: np.ndarray,
    colormap: int = cv2.COLORMAP_VIRIDIS,
    alpha: float = 0.55
) -> np.ndarray:
    """
    Blends an authentic ViT self-attention rollout heatmap onto a target mammogram / crop.
    image_2d: 2D float32 array in [0.0, 1.0]
    attn_map_2d: 2D float32 array in [0.0, 1.0]
    Returns: RGB uint8 image of the attention overlay.
    """
    # Resize attention map to image shape if necessary
    if attn_map_2d.shape != image_2d.shape:
        attn_map_2d = cv2.resize(attn_map_2d, (image_2d.shape[1], image_2d.shape[0]), interpolation=cv2.INTER_CUBIC)

    base_u8 = (np.clip(image_2d, 0.0, 1.0) * 255.0).astype(np.uint8)
    base_rgb = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2RGB)

    attn_u8 = (np.clip(attn_map_2d, 0.0, 1.0) * 255.0).astype(np.uint8)
    attn_colored = cv2.applyColorMap(attn_u8, colormap)

    # Blending
    blended = cv2.addWeighted(base_rgb, 1.0 - alpha, attn_colored, alpha, 0)
    return blended


def render_confusion_matrix_plot(cm_dict: Dict[str, int]) -> plt.Figure:
    """
    Renders a styled confusion matrix heatmap for the locked V6 test benchmark:
    TN, FP, FN, TP
    """
    tn = cm_dict.get('tn', 195)
    fp = cm_dict.get('fp', 116)
    fn = cm_dict.get('fn', 71)
    tp = cm_dict.get('tp', 144)

    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    # Custom colormap
    cax = ax.matshow(matrix, cmap='Blues', alpha=0.85)

    # Labels and styling
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign (Pred)', 'Malignant (Pred)'], color='#e2e8f0', fontsize=10, fontweight='bold')
    ax.set_yticklabels(['Benign (True)', 'Malignant (True)'], color='#e2e8f0', fontsize=10, fontweight='bold')
    ax.tick_params(colors='#e2e8f0', bottom=False, left=False)

    for (i, j), z in np.ndenumerate(matrix):
        pct = z / np.sum(matrix) * 100.0
        label_text = f"{z}\n({pct:.1f}%)"
        text_color = '#ffffff' if z > (matrix.max() / 2) else '#0f172a'
        ax.text(j, i, label_text, ha='center', va='center', color=text_color, fontsize=12, fontweight='bold')

    ax.set_title("V6 Locked Test Confusion Matrix (N=526)", color='#f8fafc', fontsize=12, pad=16, fontweight='bold')
    plt.tight_layout()
    return fig
