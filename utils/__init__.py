"""
Utility functions for breast cancer detection pipeline.
Provides deterministic seeding, metric calculations, dataset exploration, and plotting utilities.
"""

from .seed import set_seed, get_device
from .metrics import (
    calculate_segmentation_metrics,
    calculate_classification_metrics,
    DiceLoss,
    CombinedSegmentationLoss
)
from .visualization import (
    plot_preprocessing_comparison,
    plot_segmentation_prediction,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
    plot_attention_overlay
)
from .dataset_analysis import analyze_dataset
from .dataset_builder import CBISDDSMAdapter

__all__ = [
    "set_seed",
    "get_device",
    "calculate_segmentation_metrics",
    "calculate_classification_metrics",
    "DiceLoss",
    "CombinedSegmentationLoss",
    "plot_preprocessing_comparison",
    "plot_segmentation_prediction",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_training_history",
    "plot_attention_overlay",
    "analyze_dataset",
    "CBISDDSMAdapter",
]
