"""
================================================================================
V6 Multi-Scale & Top-K Proposal ROI Generator
================================================================================
Extracts deterministic multi-scale ROI crops for Top-K proposals alongside the
full-mammogram contextual canvas and calcification-enhanced patches.

Multi-Scale Receptive Fields:
1. Local Crop (s=1.2x)
2. Medium Crop (s=1.8x)
3. Large Crop (s=2.5x)
4. Context Crop (s=4.0x)

Plus:
5. Full-Mammogram Canvas (224x224)
6. Dedicated High-Resolution Calcification Enhanced Crop
================================================================================
"""

import math
import numpy as np
import cv2
import torch
from typing import Dict, Any, List, Tuple, Optional
from classification.v6.topk_proposals_v6 import TopKProposalGeneratorV6


class MultiScaleROIGeneratorV6:
    """
    V6 Multi-Scale & Top-K ROI Generator with full-mammogram context extraction.
    """
    def __init__(
        self,
        local_scale: float = 1.2,
        medium_scale: float = 1.8,
        large_scale: float = 2.5,
        context_scale: float = 4.0,
        target_crop_size: int = 224,
        full_image_size: int = 224,
        default_k: int = 3
    ):
        self.local_scale = local_scale
        self.medium_scale = medium_scale
        self.large_scale = large_scale
        self.context_scale = context_scale
        self.target_crop_size = target_crop_size
        self.full_image_size = full_image_size
        self.default_k = default_k
        self.proposal_generator = TopKProposalGeneratorV6(default_k=default_k)

    def extract_scaled_crop(
        self,
        image: np.ndarray,
        center: Tuple[float, float],
        base_size: float,
        scale: float,
        target_size: int = 224
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extracts a square crop centered at (cy, cx) with size = base_size * scale.
        Preserves aspect ratio and zero-pads at borders if necessary.
        """
        H, W = image.shape
        cy, cx = center
        crop_s = max(24.0, base_size * scale)
        half_s = crop_s / 2.0

        ymin = int(round(cy - half_s))
        ymax = int(round(cy + half_s))
        xmin = int(round(cx - half_s))
        xmax = int(round(cx + half_s))

        # Pad if bounds exceed image boundaries
        pad_top = max(0, -ymin)
        pad_bottom = max(0, ymax - H)
        pad_left = max(0, -xmin)
        pad_right = max(0, xmax - W)

        src_ymin = max(0, ymin)
        src_ymax = min(H, ymax)
        src_xmin = max(0, xmin)
        src_xmax = min(W, xmax)

        patch = image[src_ymin:src_ymax, src_xmin:src_xmax]

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

        if patch.shape[0] != target_size or patch.shape[1] != target_size:
            patch = cv2.resize(patch, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        crop_info = {
            "scale": scale,
            "crop_size": crop_s,
            "bbox_raw": (ymin, xmin, ymax, xmax),
            "padding": (pad_top, pad_bottom, pad_left, pad_right)
        }
        return patch.astype(np.float32), crop_info

    def enhance_calcification_patch(self, patch: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Applies localized adaptive unsharp masking and CLAHE for subtle microcalcifications.
        Falls back to original patch if enhancement fails.
        """
        try:
            patch_u8 = (np.clip(patch, 0.0, 1.0) * 255.0).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(patch_u8).astype(np.float32) / 255.0
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 1.0)
            unsharp = np.clip(enhanced + 0.5 * (enhanced - blurred), 0.0, 1.0)
            return unsharp.astype(np.float32), "enhanced"
        except Exception:
            return patch.astype(np.float32), "local_fallback"

    def generate_candidate_crops(
        self,
        image: np.ndarray,
        proposal: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Extracts 4 multi-scale crops for a single candidate proposal.
        """
        ymin, xmin, ymax, xmax = proposal['bbox']
        cy, cx = proposal['center']
        bw = max(16, xmax - xmin)
        bh = max(16, ymax - ymin)
        base_size = float(max(bw, bh))

        local_crop, info_l = self.extract_scaled_crop(image, (cy, cx), base_size, self.local_scale, self.target_crop_size)
        medium_crop, info_m = self.extract_scaled_crop(image, (cy, cx), base_size, self.medium_scale, self.target_crop_size)
        large_crop, info_lg = self.extract_scaled_crop(image, (cy, cx), base_size, self.large_scale, self.target_crop_size)
        context_crop, info_c = self.extract_scaled_crop(image, (cy, cx), base_size, self.context_scale, self.target_crop_size)

        crops = {
            'local': local_crop,
            'medium': medium_crop,
            'large': large_crop,
            'context': context_crop
        }
        meta = {
            'base_size': base_size,
            'center': (cy, cx),
            'scales': {'local': info_l, 'medium': info_m, 'large': info_lg, 'context': info_c}
        }
        return crops, meta
