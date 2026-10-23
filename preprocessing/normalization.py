import numpy as np
from typing import Tuple, Union


def percentile_clip(
    image: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5
) -> np.ndarray:
    """
    Clips extreme intensity outliers (dead pixels, radiopaque markers, hot pixels) in mammograms.

    Args:
        image: 2D or 3D numpy array.
        lower_percentile: Lower percentile threshold.
        upper_percentile: Upper percentile threshold.

    Returns:
        Clipped image array.
    """
    p_low, p_high = np.percentile(image, (lower_percentile, upper_percentile))
    clipped = np.clip(image, p_low, p_high)
    return clipped


def normalize_intensity(
    image: np.ndarray,
    target_min: float = 0.0,
    target_max: float = 1.0,
    dtype: type = np.float32
) -> np.ndarray:
    """
    Linearly normalizes image intensity into [target_min, target_max] safely.

    Args:
        image: Numpy array of any numeric type.
        target_min: Desired minimum output value.
        target_max: Desired maximum output value.
        dtype: Output numpy datatype.

    Returns:
        Normalized numpy array.
    """
    img_float = image.astype(np.float64)
    img_min = float(np.min(img_float))
    img_max = float(np.max(img_float))
    
    if abs(img_max - img_min) < 1e-7:
        # Constant image fallback
        return np.zeros_like(img_float, dtype=dtype)
        
    norm = (img_float - img_min) / (img_max - img_min)
    rescaled = norm * (target_max - target_min) + target_min
    return rescaled.astype(dtype)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Converts any floating point [0, 1] or raw mammogram array to uint8 [0, 255] for OpenCV operations.
    """
    norm = normalize_intensity(image, 0.0, 255.0)
    return norm.astype(np.uint8)
