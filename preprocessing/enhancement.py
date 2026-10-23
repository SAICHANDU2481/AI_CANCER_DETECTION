import os
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
import yaml
from PIL import Image

from .normalization import normalize_intensity, percentile_clip, to_uint8
from .wiener_filter import apply_wiener_filter

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("Enhancement")


def apply_clahe_enhancement(
    image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance mammographic contrast.

    Args:
        image: Grayscale numpy array in range [0, 1] or [0, 255].
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for local histogram equalization.

    Returns:
        Enhanced grayscale float32 array in range [0.0, 1.0].
    """
    # Convert to uint8 for OpenCV CLAHE
    img_u8 = to_uint8(image)
    
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(tile_grid_size)
    )
    enhanced_u8 = clahe.apply(img_u8)
    
    return normalize_intensity(enhanced_u8, 0.0, 1.0)


def enhance_mammogram(
    image: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = "config.yaml"
) -> Dict[str, np.ndarray]:
    """
    Executes the full preprocessing pipeline:
    1. Input validation and Grayscale conversion.
    2. Outlier percentile clipping.
    3. Wiener filtering for speckle/noise reduction.
    4. CLAHE contrast enhancement.
    5. Final range normalization.

    Args:
        image: Input 2D or 3D numpy array.
        config: Optional dictionary of preprocessing parameters.
        config_path: Optional path to config.yaml if config is None.

    Returns:
        Dictionary containing:
        - "original": Normalized original image [0, 1]
        - "wiener": Wiener filtered image [0, 1]
        - "enhanced": Final preprocessed and CLAHE enhanced image [0, 1]
    """
    # Load default settings from config if not passed directly
    if config is None and config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            full_cfg = yaml.safe_load(f)
            config = full_cfg.get("preprocessing", {})
            
    if config is None:
        config = {}

    wiener_cfg = config.get("wiener", {})
    clahe_cfg = config.get("clahe", {})
    norm_cfg = config.get("normalization", {})

    window_size = wiener_cfg.get("window_size", [5, 5])
    wiener_mode = wiener_cfg.get("mode", "scipy")
    noise_var = wiener_cfg.get("noise_variance", None)
    
    clip_limit = clahe_cfg.get("clip_limit", 2.5)
    tile_grid = clahe_cfg.get("tile_grid_size", [8, 8])
    
    p_clip = norm_cfg.get("percentile_clip", [0.5, 99.5])

    # Convert to single channel if RGB
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image.squeeze(2)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 1. Base normalization
    orig_norm = normalize_intensity(image, 0.0, 1.0)
    
    # 2. Outlier clipping
    clipped = percentile_clip(orig_norm, lower_percentile=p_clip[0], upper_percentile=p_clip[1])
    clipped = normalize_intensity(clipped, 0.0, 1.0)
    
    # 3. Wiener filter
    wiener_filtered = apply_wiener_filter(
        clipped,
        window_size=window_size,
        noise_variance=noise_var,
        mode=wiener_mode
    )
    
    # 4. CLAHE contrast enhancement
    clahe_enhanced = apply_clahe_enhancement(
        wiener_filtered,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid
    )
    
    return {
        "original": orig_norm,
        "wiener": wiener_filtered,
        "enhanced": clahe_enhanced
    }
