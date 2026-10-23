import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List

from segmentation.ukan_v2b import MultiScaleUKAN
from classification.v4.roi_quality import extract_roi_quality_features, QUALITY_FEATURE_NAMES

class MultiScaleROIGeneratorV4:
    """
    V4 Multi-Scale ROI Generator.
    Integrates frozen V2-B MultiScaleUKAN with inference-only 16-feature ROI Quality Scoring
    and 4 explicit proposal states:
      - machine_roi (Primary ROI: bbox + 10% margin)
      - expanded_machine_roi (Expanded ROI: bbox + 30% margin when quality < 0.30 or small)
      - context_roi (2.0x bbox context)
      - fallback_context (deterministic tissue-centroid fallback when mask is empty)
    """
    def __init__(
        self,
        checkpoint_path: str = 'models/v2b/segmentation/ukan_v2b_best.pth',
        device: Optional[torch.device] = None,
        prob_threshold: float = 0.50,
        min_area: int = 50,
        target_crop_size: int = 224
    ):
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.prob_threshold = prob_threshold
        self.min_area = min_area
        self.target_crop_size = target_crop_size
        
        # Load and freeze V2-B U-KAN
        self.segmentor = MultiScaleUKAN(in_channels=1, out_channels=1).to(self.device)
        chk = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(chk, dict) and 'model_state_dict' in chk:
            self.segmentor.load_state_dict(chk['model_state_dict'])
        elif isinstance(chk, dict) and 'state_dict' in chk:
            self.segmentor.load_state_dict(chk['state_dict'])
        else:
            self.segmentor.load_state_dict(chk)
            
        self.segmentor.eval()
        for p in self.segmentor.parameters():
            p.requires_grad = False
            
    @torch.no_grad()
    def segment_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Runs single-pass deterministic inference through frozen V2-B U-KAN.
        Input: image_tensor [512, 512] or [1, 512, 512] or [1, 1, 512, 512]
        Returns: prob_map (np.ndarray of shape [512, 512], float32)
        """
        if image_tensor.dim() == 2:
            inp = image_tensor.unsqueeze(0).unsqueeze(0)
        elif image_tensor.dim() == 3:
            inp = image_tensor.unsqueeze(0)
        else:
            inp = image_tensor
            
        inp = inp.to(self.device, dtype=torch.float32)
        out = self.segmentor(inp)
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
        return probs

    def _get_tissue_centroid(self, image_512: np.ndarray) -> Tuple[int, int]:
        """
        Computes the breast tissue centroid using Otsu thresholding on the mammogram.
        Deterministic fallback when segmentation is completely empty.
        """
        img_u8 = (np.clip(image_512, 0.0, 1.0) * 255).astype(np.uint8)
        _, tissue_mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        y_indices, x_indices = np.where(tissue_mask > 0)
        if len(y_indices) > 0:
            cy = int(np.mean(y_indices))
            cx = int(np.mean(x_indices))
        else:
            cy, cx = 256, 256
        return cy, cx

    def _extract_crop(self, image_512: np.ndarray, cy: int, cx: int, box_size: int, target_size: int) -> np.ndarray:
        """
        Extracts a square bounding box centered at (cy, cx), clamped to [0, 512], and resizes to target_size.
        """
        H, W = image_512.shape
        half = box_size // 2
        y1 = max(0, cy - half)
        y2 = min(H, cy + half)
        x1 = max(0, cx - half)
        x2 = min(W, cx + half)
        
        crop = image_512[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            crop = np.zeros((target_size, target_size), dtype=np.float32)
        else:
            crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        return crop.astype(np.float32)

    def generate_rois(
        self,
        image_512: np.ndarray,
        prob_map: Optional[np.ndarray] = None,
        local_target_size: int = 224,
        context_target_size: int = 224
    ) -> Dict[str, Any]:
        """
        Generates 4-scale ROI crops and extracts 16-dimensional quality features.
        Zero ground truth mask access.
        """
        if prob_map is None:
            tensor_img = torch.from_numpy(image_512).float()
            prob_map = self.segment_image(tensor_img)
            
        quality_vector, roi_quality_score, feat_dict = extract_roi_quality_features(
            prob_map,
            prob_threshold=self.prob_threshold,
            min_component_area=10
        )
        
        binary_mask = (prob_map >= self.prob_threshold).astype(np.uint8)
        mask_pixel_count = int(np.sum(binary_mask))
        
        # Determine ROI State
        if mask_pixel_count == 0:
            # Completely empty segmentation -> deterministic fallback_context
            roi_source = 'fallback_context'
            cy, cx = self._get_tissue_centroid(image_512)
            base_box_size = 224
        else:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            valid_indices = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 10]
            if len(valid_indices) == 0:
                valid_indices = [i for i in range(1, num_labels)]
                
            primary_idx = max(valid_indices, key=lambda i: stats[i, cv2.CC_STAT_AREA])
            w = stats[primary_idx, cv2.CC_STAT_WIDTH]
            h = stats[primary_idx, cv2.CC_STAT_HEIGHT]
            cx_f, cy_f = centroids[primary_idx]
            cx, cy = int(cx_f), int(cy_f)
            
            raw_box_size = max(w, h, 32)
            
            # Decide between machine_roi vs expanded_machine_roi
            is_poor_quality = (
                roi_quality_score < 0.30 or
                mask_pixel_count < 200 or
                feat_dict['boundary_touch_ratio'] > 0.05 or
                feat_dict['mask_confidence'] < 0.60
            )
            if is_poor_quality:
                roi_source = 'expanded_machine_roi'
                # +30% expansion margin
                base_box_size = int(raw_box_size * 1.30)
            else:
                roi_source = 'machine_roi'
                # +10% standard margin
                base_box_size = int(raw_box_size * 1.10)
                
        # Calculate multi-scale crop box dimensions
        local_box_size = max(base_box_size, 32)
        med_box_size = int(local_box_size * 1.50)
        large_box_size = int(local_box_size * 2.00)
        context_box_size = max(int(local_box_size * 2.50), 300)
        
        local_crop = self._extract_crop(image_512, cy, cx, local_box_size, local_target_size)
        medium_crop = self._extract_crop(image_512, cy, cx, med_box_size, context_target_size)
        large_crop = self._extract_crop(image_512, cy, cx, large_box_size, context_target_size)
        context_crop = self._extract_crop(image_512, cy, cx, context_box_size, context_target_size)
        
        return {
            'local_crop': local_crop,
            'medium_crop': medium_crop,
            'large_crop': large_crop,
            'context_crop': context_crop,
            'quality_vector': quality_vector,
            'roi_quality_score': roi_quality_score,
            'roi_source': roi_source,
            'fallback_used': (roi_source == 'fallback_context'),
            'mask_pixel_count': mask_pixel_count,
            'center_y': cy,
            'center_x': cx,
            'local_box_size': local_box_size,
            'feature_dict': feat_dict
        }
