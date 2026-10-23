"""
Segmentation module for breast lesion ROI localization using U-KAN.
"""

from .ukan import UKAN, KANLinear, KANBottleneck, build_ukan_from_config
from .ukan_v5 import MultiScaleUKANV5

try:
    from .dataset import MammographySegmentationDataset, get_segmentation_loaders
except ImportError:
    pass

try:
    from .inference import extract_lesion_roi, segment_and_crop
except ImportError:
    pass

KANConvBlock = KANBottleneck

__all__ = [
    "UKAN",
    "KANLinear",
    "KANBottleneck",
    "KANConvBlock",
    "build_ukan_from_config",
    "MultiScaleUKANV5"
]

