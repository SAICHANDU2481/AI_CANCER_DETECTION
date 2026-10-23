import logging
from typing import Tuple, Optional
import numpy as np
import cv2
import matplotlib.cm as cm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ExplainabilityViz")

EXPLAINABILITY_DISCLAIMER = (
    "NOTICE: The attention visualization represents transformer self-attention token flow "
    "and feature saliency. It is generated for algorithmic interpretability and research "
    "purposes only, and does not constitute histological, pathological, or diagnostic evidence."
)


def create_attention_heatmap(
    attention_map: np.ndarray,
    colormap: str = "jet"
) -> np.ndarray:
    """
    Transforms a 2D float attention map [0, 1] into an RGB heatmap.

    Args:
        attention_map: 2D numpy array with values in [0, 1].
        colormap: Matplotlib or OpenCV colormap name ('jet', 'inferno', 'magma', 'viridis').

    Returns:
        RGB numpy array with values in range [0, 255] uint8.
    """
    norm_map = np.clip(attention_map, 0.0, 1.0)
    
    # Get colormap from matplotlib
    try:
        import matplotlib
        if hasattr(matplotlib, "colormaps"):
            cmap = matplotlib.colormaps[colormap]
        else:
            cmap = cm.get_cmap(colormap)
        rgba = cmap(norm_map) # (H, W, 4) in [0, 1]
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    except Exception:
        # Fallback to OpenCV JET
        u8_map = (norm_map * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(u8_map, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb


def overlay_attention_map(
    base_image: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.55,
    colormap: str = "jet"
) -> np.ndarray:
    """
    Overlays colored attention heatmap onto grayscale or RGB base image.

    Args:
        base_image: 2D grayscale [0, 1] or 3D RGB [0, 255] / [0, 1].
        attention_map: 2D float array [0, 1] matching base image spatial dimensions.
        alpha: Blending weight for attention heatmap (0.0 = only image, 1.0 = only heatmap).
        colormap: Heatmap colormap.

    Returns:
        RGB uint8 numpy array [0, 255] representing the blended overlay.
    """
    h, w = base_image.shape[:2]
    
    # Resize attention map if dimensions differ
    if attention_map.shape[:2] != (h, w):
        attention_map = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)

    # 1. Prepare base RGB uint8
    if base_image.ndim == 2:
        norm_img = np.clip(base_image, 0.0, 1.0)
        base_u8 = (norm_img * 255).astype(np.uint8)
        base_rgb = np.stack([base_u8, base_u8, base_u8], axis=-1)
    elif base_image.ndim == 3:
        if base_image.dtype == np.float32 or base_image.dtype == np.float64:
            base_rgb = (np.clip(base_image, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            base_rgb = base_image.copy()

    # 2. Generate heatmap RGB
    heatmap_rgb = create_attention_heatmap(attention_map, colormap=colormap)

    # 3. Alpha blend
    blended = cv2.addWeighted(heatmap_rgb, float(alpha), base_rgb, 1.0 - float(alpha), 0)

    return blended
