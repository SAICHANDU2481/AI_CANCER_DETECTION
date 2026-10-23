"""
================================================================================
V3-A Multi-Scale Machine-ROI Generator & Extractor
================================================================================
Extracts synchronized 3-scale contextual crop triplets using frozen V2-B Multi-Scale U-KAN:
- Local ROI (1.0x with 10% safety margin: 1.1x)
- Medium ROI (1.5x context)
- Large ROI (2.0x global parenchymal context)
- Fallback concentric center crops if no component >= 50px detected.
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from segmentation.ukan_v2b import MultiScaleUKAN


def extract_primary_component_bbox(
    prob_mask: np.ndarray,
    threshold: float = 0.5,
    min_area_pixels: int = 50
) -> Tuple[bool, Tuple[int, int, int, int], int]:
    """
    Identifies the primary lesion connected component >= min_area_pixels.
    Returns: (roi_detected, (xmin, ymin, xmax, ymax), area)
    """
    binary_mask = (prob_mask >= threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    best_bbox = (0, 0, 0, 0)
    max_area = 0
    roi_detected = False

    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area_pixels and area > max_area:
            max_area = int(area)
            xmin = int(stats[label_idx, cv2.CC_STAT_LEFT])
            ymin = int(stats[label_idx, cv2.CC_STAT_TOP])
            bw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            best_bbox = (xmin, ymin, xmin + bw, ymin + bh)
            roi_detected = True

    return roi_detected, best_bbox, max_area


def compute_multiscale_boxes(
    bbox: Tuple[int, int, int, int],
    roi_detected: bool,
    img_shape: Tuple[int, int] = (512, 512)
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Computes local, medium, and large bounding boxes clamped to img_shape.
    """
    H, W = img_shape

    if roi_detected:
        xmin, ymin, xmax, ymax = bbox
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        bw = max(xmax - xmin, 8.0)
        bh = max(ymax - ymin, 8.0)

        # 1. Local ROI: 1.0x with 10% safety margin (1.1x)
        lx1 = int(round(cx - 0.55 * bw))
        lx2 = int(round(cx + 0.55 * bw))
        ly1 = int(round(cy - 0.55 * bh))
        ly2 = int(round(cy + 0.55 * bh))

        # 2. Medium ROI: 1.5x
        mx1 = int(round(cx - 0.75 * bw))
        mx2 = int(round(cx + 0.75 * bw))
        my1 = int(round(cy - 0.75 * bh))
        my2 = int(round(cy + 0.75 * bh))

        # 3. Large ROI: 2.0x
        gx1 = int(round(cx - 1.00 * bw))
        gx2 = int(round(cx + 1.00 * bw))
        gy1 = int(round(cy - 1.00 * bh))
        gy2 = int(round(cy + 1.00 * bh))
    else:
        # Fallback concentric center crops
        cx, cy = W / 2.0, H / 2.0
        # Local: 128x128
        lx1, lx2 = int(cx - 64), int(cx + 64)
        ly1, ly2 = int(cy - 64), int(cy + 64)
        # Medium: 192x192
        mx1, mx2 = int(cx - 96), int(cx + 96)
        my1, my2 = int(cy - 96), int(cy + 96)
        # Large: 256x256
        gx1, gx2 = int(cx - 128), int(cx + 128)
        gy1, gy2 = int(cy - 128), int(cy + 128)

    # Helper to clamp and ensure minimum 8px valid box
    def _clamp(x1, y1, x2, y2):
        x1 = max(0, min(W - 8, x1))
        x2 = max(x1 + 8, min(W, x2))
        y1 = max(0, min(H - 8, y1))
        y2 = max(y1 + 8, min(H, y2))
        return (x1, y1, x2, y2)

    return {
        "local": _clamp(lx1, ly1, lx2, ly2),
        "medium": _clamp(mx1, my1, mx2, my2),
        "large": _clamp(gx1, gy1, gx2, gy2)
    }


def crop_and_resize(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Crops a 2D bounding box from image and resizes to target_size.
    Returns float32 array in [0, 1] of shape (224, 224).
    """
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
    
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


def generate_v3_triplet_crops(
    image_512: np.ndarray,
    prob_mask: np.ndarray,
    threshold: float = 0.5,
    min_area_pixels: int = 50,
    target_size: Tuple[int, int] = (224, 224)
) -> Dict[str, Any]:
    """
    Processes a single 512x512 preprocessed image and its U-KAN probability mask.
    Returns a dict containing:
    - local_crop: np.ndarray (224, 224)
    - medium_crop: np.ndarray (224, 224)
    - large_crop: np.ndarray (224, 224)
    - roi_source: 'machine' | 'fallback'
    - mask_area_px: int
    - boxes: dict of (x1, y1, x2, y2)
    """
    roi_detected, raw_bbox, mask_area = extract_primary_component_bbox(
        prob_mask, threshold=threshold, min_area_pixels=min_area_pixels
    )
    roi_source = "machine" if roi_detected else "fallback"
    boxes = compute_multiscale_boxes(raw_bbox, roi_detected, img_shape=image_512.shape[:2])

    local_crop = crop_and_resize(image_512, boxes["local"], target_size=target_size)
    medium_crop = crop_and_resize(image_512, boxes["medium"], target_size=target_size)
    large_crop = crop_and_resize(image_512, boxes["large"], target_size=target_size)

    return {
        "local_crop": local_crop,
        "medium_crop": medium_crop,
        "large_crop": large_crop,
        "roi_source": roi_source,
        "mask_area_px": mask_area,
        "raw_bbox": raw_bbox,
        "boxes": boxes
    }
