"""
================================================================================
V5 Multi-Scale & Top-K Proposal ROI Generator
================================================================================
Generates deterministic multi-scale ROI crops for Top-K proposals alongside the
full-mammogram contextual canvas.

Extracts for each of K candidate proposals:
1. Local Crop ($s_{local}$, default 1.2x)
2. Medium Crop ($s_{medium}$, default 1.8x)
3. Large Crop ($s_{large}$, default 2.5x)
4. Surrounding Context Crop ($s_{context}$, default 4.0x)

Plus:
5. Full-Mammogram Breast Canvas (default 224x224 or 384x384)
6. Optional High-Resolution Calcification Enhanced Patch
"""

import os
import math
import numpy as np
import cv2
import torch
from typing import Dict, Any, List, Tuple, Optional
from classification.v5.topk_proposals import TopKProposalGenerator


class MultiScaleROIGeneratorV5:
    """
    V5 Multi-Scale & Top-K ROI Generator with full-mammogram context extraction.
    """
    def __init__(
        self,
        segmentor: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        device: torch.device = torch.device('cpu'),
        local_scale: float = 1.2,
        medium_scale: float = 1.8,
        large_scale: float = 2.5,
        context_scale: float = 4.0,
        target_crop_size: int = 224,
        full_image_size: int = 224,
        default_k: int = 3,
        calcification_high_res: bool = False
    ):
        self.device = device
        self.local_scale = local_scale
        self.medium_scale = medium_scale
        self.large_scale = large_scale
        self.context_scale = context_scale
        self.target_crop_size = target_crop_size
        self.full_image_size = full_image_size
        self.default_k = default_k
        self.calcification_high_res = calcification_high_res

        self.proposal_generator = TopKProposalGenerator(default_k=default_k)

        if segmentor is not None:
            self.segmentor = segmentor.to(device)
            self.segmentor.eval()
        elif checkpoint_path is not None and os.path.exists(checkpoint_path):
            from segmentation.ukan_v2b import MultiScaleUKAN
            self.segmentor = MultiScaleUKAN().to(device)
            chk = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in chk:
                self.segmentor.load_state_dict(chk['model_state_dict'])
            else:
                self.segmentor.load_state_dict(chk)
            self.segmentor.eval()
        else:
            self.segmentor = None

    def _extract_scaled_crop(
        self,
        image: np.ndarray,
        center: Tuple[float, float],
        base_size: float,
        scale: float,
        target_size: int
    ) -> np.ndarray:
        """
        Extracts a square crop centered at (cy, cx) with size = base_size * scale.
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

        return patch.astype(np.float32)

    def enhance_calcification_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Applies localized adaptive unsharp masking and CLAHE for subtle microcalcifications.
        """
        patch_u8 = (np.clip(patch, 0.0, 1.0) * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(patch_u8).astype(np.float32) / 255.0
        # Mild unsharp mask
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 1.0)
        unsharp = np.clip(enhanced + 0.5 * (enhanced - blurred), 0.0, 1.0)
        return unsharp.astype(np.float32)

    def generate_candidate_crops(
        self,
        image: np.ndarray,
        proposal: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Extracts 4 multi-scale crops for a single candidate proposal.
        """
        ymin, xmin, ymax, xmax = proposal['bbox']
        cy, cx = proposal['center']
        bw = max(16, xmax - xmin)
        bh = max(16, ymax - ymin)
        base_size = float(max(bw, bh))

        local_crop = self._extract_scaled_crop(image, (cy, cx), base_size, self.local_scale, self.target_crop_size)
        medium_crop = self._extract_scaled_crop(image, (cy, cx), base_size, self.medium_scale, self.target_crop_size)
        large_crop = self._extract_scaled_crop(image, (cy, cx), base_size, self.large_scale, self.target_crop_size)
        context_crop = self._extract_scaled_crop(image, (cy, cx), base_size, self.context_scale, self.target_crop_size)

        if self.calcification_high_res:
            local_crop = self.enhance_calcification_patch(local_crop)

        return {
            'local': local_crop,
            'medium': medium_crop,
            'large': large_crop,
            'context': context_crop
        }

    def generate_all_rois(
        self,
        image_canvas: np.ndarray,
        prob_map: Optional[np.ndarray] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Full V5 feature generation:
        - Full mammogram canvas
        - Top-K proposals
        - 4 multi-scale crops per proposal
        """
        k = k if k is not None else self.default_k

        if prob_map is None:
            if self.segmentor is None:
                raise ValueError("Segmentor model required if prob_map is not provided")
            inp = torch.from_numpy(image_canvas).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                logits = self.segmentor(inp)
                prob_map = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)

        # Full mammogram image resized to target resolution
        full_image = cv2.resize(image_canvas, (self.full_image_size, self.full_image_size), interpolation=cv2.INTER_AREA)

        # Generate Top-K proposals
        proposals = self.proposal_generator.generate_proposals(image_canvas, prob_map, k=k)

        candidates_crops = []
        for prop in proposals:
            crops = self.generate_candidate_crops(image_canvas, prop)
            candidates_crops.append(crops)

        return {
            'full_image': full_image.astype(np.float32),
            'proposals': proposals,
            'candidates_crops': candidates_crops,
            'prob_map': prob_map
        }
