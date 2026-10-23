"""
================================================================================
V7 Multi-Proposal Generator Suite (Phase 5) - High Performance Implementation
================================================================================
Implements independent and combinable proposal sources for breast cancer lesion
localization without any ground-truth or label leakage:
- Source A: V6 U-KAN Baseline Proposals (512x512)
- Source B: High-Resolution Breast-ROI U-KAN Proposals (768x768)
- Source C: Calcification-Specific Detectors (Top-Hat, High-Pass, DoG, Contrast)
- Source D: Local Contrast & High-Frequency Saliency Detectors
- Source E: Contextual Candidate Bank (Control / Recovery Baseline)
- Source F: Random Crop Baseline (Fixed-Seed Control)
- Source G: Fixed Spatial Grid Baseline
- Label-Free NMS and Unified Ranking
================================================================================
"""

import os
import sys
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from segmentation.v6.ukan_v6 import MultiScaleUKANV6, build_ukan_v6

logger = logging.getLogger("V7_Proposal_Generators")


# ==============================================================================
# 1. DETERMINISTIC BREAST TISSUE BOUNDING BOX EXTRACTOR
# ==============================================================================
def extract_breast_bounding_box(
    image_gray: np.ndarray,
    min_intensity: float = 0.03,
    margin_ratio: float = 0.02
) -> Tuple[Tuple[int, int, int, int], np.ndarray]:
    """
    Extracts the bounding box and binary mask of breast tissue from a 2D mammogram.
    image_gray: float32 in [0, 1] or uint8 in [0, 255]
    Returns: (bbox: (ymin, xmin, ymax, xmax), small_tissue_mask: uint8 2D at working scale)
    """
    H, W = image_gray.shape
    if image_gray.dtype != np.uint8:
        small_img = cv2.resize((image_gray * 255).astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA)
    else:
        small_img = cv2.resize(image_gray, (512, 512), interpolation=cv2.INTER_AREA)
        
    blurred = cv2.GaussianBlur(small_img, (15, 15), 0)
    _, small_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    small_thresh[small_img < int(min_intensity * 255)] = 0
    
    # Largest connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(small_thresh, connectivity=8)
    if num_labels > 1:
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        small_mask = (labels == largest_idx).astype(np.uint8)
    else:
        small_mask = (small_img > int(min_intensity * 255)).astype(np.uint8)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    small_mask = cv2.morphologyEx(small_mask, cv2.MORPH_CLOSE, kernel)
    
    if np.sum(small_mask) > 50:
        ys, xs = np.where(small_mask > 0)
        ymin = int(np.min(ys) * H / 512.0)
        ymax = int(np.max(ys) * H / 512.0)
        xmin = int(np.min(xs) * W / 512.0)
        xmax = int(np.max(xs) * W / 512.0)
        
        pad_y = int((ymax - ymin) * margin_ratio)
        pad_x = int((xmax - xmin) * margin_ratio)
        ymin = max(0, ymin - pad_y)
        ymax = min(H, ymax + pad_y)
        xmin = max(0, xmin - pad_x)
        xmax = min(W, xmax + pad_x)
    else:
        ymin, ymax, xmin, xmax = 0, H, 0, W
        small_mask = np.ones((512, 512), dtype=np.uint8)
        
    return (ymin, xmin, ymax, xmax), small_mask


# ==============================================================================
# 2. LABEL-FREE PROPOSAL SCORER
# ==============================================================================
def score_proposal_box(
    image_gray: np.ndarray,
    bbox: Tuple[int, int, int, int],
    saliency_crop: Optional[np.ndarray] = None,
    source_name: str = 'generic',
    base_confidence: float = 0.5
) -> Dict[str, Any]:
    """
    Computes deterministic label-free quality features for a proposal bounding box.
    bbox: (ymin, xmin, ymax, xmax) in native image coordinates
    """
    H, W = image_gray.shape
    ymin, xmin, ymax, xmax = bbox
    ymin, ymax = max(0, ymin), min(H, ymax)
    xmin, xmax = max(0, xmin), min(W, xmax)
    
    box_h = ymax - ymin
    box_w = xmax - xmin
    box_area = box_h * box_w
    cy = float(ymin + box_h / 2.0)
    cx = float(xmin + box_w / 2.0)
    
    if box_area <= 0:
        return {
            'bbox': (0, 0, 0, 0),
            'center': (0.0, 0.0),
            'score': 0.0,
            'mean_intensity': 0.0,
            'peak_intensity': 0.0,
            'local_contrast': 0.0,
            'area': 0,
            'proposal_source': source_name
        }
        
    # Sample crop at modest resolution for speed
    crop = image_gray[ymin:ymax, xmin:xmax]
    mean_int = float(np.mean(crop))
    peak_int = float(np.max(crop))
    std_int = float(np.std(crop))
    
    # Aspect ratio / compactness
    aspect_ratio = float(box_w) / float(box_h + 1e-6)
    compactness = 1.0 - min(1.0, abs(float(np.log(aspect_ratio + 1e-6))))
    
    # Saliency / probability contribution
    if saliency_crop is not None:
        mean_sal = float(np.mean(saliency_crop))
        peak_sal = float(np.max(saliency_crop))
    else:
        mean_sal = base_confidence
        peak_sal = base_confidence
        
    # Unified label-free composite score
    norm_area = min(1.0, box_area / (0.20 * H * W + 1e-6))
    score = float(
        0.50 * mean_sal +
        0.20 * peak_sal +
        0.15 * min(1.0, mean_int * 1.5) +
        0.10 * compactness +
        0.05 * norm_area
    )
    
    return {
        'bbox': (ymin, xmin, ymax, xmax),
        'center': (cy, cx),
        'score': score,
        'mean_intensity': mean_int,
        'peak_intensity': peak_int,
        'std_intensity': std_int,
        'mean_sal': mean_sal,
        'peak_sal': peak_sal,
        'area': box_area,
        'compactness': compactness,
        'proposal_source': source_name
    }


# ==============================================================================
# 3. 2D BOUNDING BOX NMS (DEDUPLICATION)
# ==============================================================================
def box_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    
    inter_area = max(0, y2 - y1) * max(0, x2 - x1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    
    return float(inter_area / union_area) if union_area > 0 else 0.0


def non_max_suppression_proposals(
    proposals: List[Dict[str, Any]],
    iou_thresh: float = 0.40,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Applies greedy Non-Maximum Suppression to filter highly overlapping proposals.
    """
    if not proposals:
        return []
        
    sorted_props = sorted(proposals, key=lambda x: x['score'], reverse=True)
    kept = []
    
    for cand in sorted_props:
        cand_bbox = cand['bbox']
        overlap = False
        for k_prop in kept:
            if box_iou(cand_bbox, k_prop['bbox']) > iou_thresh:
                overlap = True
                break
        if not overlap:
            kept.append(cand)
            if len(kept) >= top_k:
                break
                
    return kept


# ==============================================================================
# 4. SOURCE A: V6 U-KAN PROPOSALS (512x512)
# ==============================================================================
class V6UKANProposalGenerator:
    def __init__(self, ukan_model: MultiScaleUKANV6, threshold: float = 0.50, min_area_512: int = 15):
        self.model = ukan_model
        self.threshold = threshold
        self.min_area_512 = min_area_512
        self.device = next(ukan_model.parameters()).device

    def generate(self, image_gray: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        H, W = image_gray.shape
        img_512 = cv2.resize(image_gray, (512, 512), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(img_512[np.newaxis, np.newaxis, :, :]).float().to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                logits = self.model(tensor)
                prob_map_512 = torch.sigmoid(logits).squeeze().float().cpu().numpy()
                
        bin_mask_512 = (prob_map_512 >= self.threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask_512, connectivity=8)
        
        proposals = []
        for label_idx in range(1, num_labels):
            area_512 = stats[label_idx, cv2.CC_STAT_AREA]
            if area_512 < self.min_area_512:
                continue
            xmin_512 = int(stats[label_idx, cv2.CC_STAT_LEFT])
            ymin_512 = int(stats[label_idx, cv2.CC_STAT_TOP])
            w_512 = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h_512 = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            
            # Map back to native H, W coordinates
            ymin = int(ymin_512 * H / 512.0)
            ymax = min(H, int((ymin_512 + h_512) * H / 512.0))
            xmin = int(xmin_512 * W / 512.0)
            xmax = min(W, int((xmin_512 + w_512) * W / 512.0))
            
            sal_crop = prob_map_512[ymin_512:ymin_512+h_512, xmin_512:xmin_512+w_512]
            p = score_proposal_box(image_gray, (ymin, xmin, ymax, xmax), sal_crop, source_name='v6_ukan_512')
            proposals.append(p)
            
        proposals.sort(key=lambda x: x['score'], reverse=True)
        return proposals[:top_k]


# ==============================================================================
# 5. SOURCE B: HIGH-RESOLUTION BREAST-ROI U-KAN PROPOSALS (768x768)
# ==============================================================================
class HighResUKANProposalGenerator:
    def __init__(self, ukan_model: MultiScaleUKANV6, target_res: int = 768, threshold: float = 0.40):
        self.model = ukan_model
        self.target_res = target_res
        self.threshold = threshold
        self.device = next(ukan_model.parameters()).device

    def generate(self, image_gray: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        H, W = image_gray.shape
        (b_ymin, b_xmin, b_ymax, b_xmax), _ = extract_breast_bounding_box(image_gray)
        
        breast_crop = image_gray[b_ymin:b_ymax, b_xmin:b_xmax]
        b_h, b_w = breast_crop.shape
        if b_h < 32 or b_w < 32:
            return []
            
        crop_res = cv2.resize(breast_crop, (self.target_res, self.target_res), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(crop_res[np.newaxis, np.newaxis, :, :]).float().to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                logits = self.model(tensor)
                prob_res = torch.sigmoid(logits).squeeze().float().cpu().numpy()
                
        bin_mask_res = (prob_res >= self.threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask_res, connectivity=8)
        
        proposals = []
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < 15:
                continue
            xmin_r = int(stats[label_idx, cv2.CC_STAT_LEFT])
            ymin_r = int(stats[label_idx, cv2.CC_STAT_TOP])
            w_r = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h_r = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            
            # Map back to breast crop -> native image coordinates
            ymin_c = int(ymin_r * b_h / float(self.target_res))
            ymax_c = int((ymin_r + h_r) * b_h / float(self.target_res))
            xmin_c = int(xmin_r * b_w / float(self.target_res))
            xmax_c = int((xmin_r + w_r) * b_w / float(self.target_res))
            
            ymin = max(0, b_ymin + ymin_c)
            ymax = min(H, b_ymin + ymax_c)
            xmin = max(0, b_xmin + xmin_c)
            xmax = min(W, b_xmin + xmax_c)
            
            sal_crop = prob_res[ymin_r:ymin_r+h_r, xmin_r:xmin_r+w_r]
            p = score_proposal_box(image_gray, (ymin, xmin, ymax, xmax), sal_crop, source_name='highres_ukan_768', base_confidence=0.7)
            proposals.append(p)
            
        proposals.sort(key=lambda x: x['score'], reverse=True)
        return proposals[:top_k]


# ==============================================================================
# 6. SOURCE C: FAST CALCIFICATION-SPECIFIC PROPOSAL GENERATORS
# ==============================================================================
class CalcificationProposalGenerator:
    """
    High-speed deterministic calcification specks and cluster detection on breast ROI.
    Modes: 'tophat', 'highpass', 'dog', 'contrast', 'all'
    """
    def __init__(self, mode: str = 'all', working_res: int = 1024):
        self.mode = mode
        self.working_res = working_res

    def _get_working_roi(self, image_gray: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray, float, float]:
        H, W = image_gray.shape
        (b_ymin, b_xmin, b_ymax, b_xmax), _ = extract_breast_bounding_box(image_gray)
        b_h = max(32, b_ymax - b_ymin)
        b_w = max(32, b_xmax - b_xmin)
        
        breast_crop = image_gray[b_ymin:b_ymax, b_xmin:b_xmax]
        scale_y = self.working_res / float(b_h)
        scale_x = self.working_res / float(b_w)
        
        crop_u8 = (cv2.resize(breast_crop, (self.working_res, self.working_res), interpolation=cv2.INTER_LINEAR) * 255).astype(np.uint8)
        _, tissue_mask = cv2.threshold(crop_u8, 10, 255, cv2.THRESH_BINARY)
        tissue_mask = cv2.erode(tissue_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        
        return crop_u8, (b_ymin, b_xmin, b_ymax, b_xmax), tissue_mask, scale_y, scale_x

    def _extract_clusters(
        self,
        image_gray: np.ndarray,
        bin_specks: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        scale_y: float,
        scale_x: float,
        source_name: str
    ) -> List[Dict[str, Any]]:
        H, W = image_gray.shape
        b_ymin, b_xmin, b_ymax, b_xmax = roi_bbox
        
        # Dilate specks into clusters
        cluster_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        clusters = cv2.dilate(bin_specks, cluster_kernel)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clusters, connectivity=8)
        props = []
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < 10 or area > 0.4 * self.working_res * self.working_res:
                continue
            xmin_w = int(stats[label_idx, cv2.CC_STAT_LEFT])
            ymin_w = int(stats[label_idx, cv2.CC_STAT_TOP])
            w_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h_w = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            
            # Map back to full native image
            ymin = max(0, b_ymin + int(ymin_w / scale_y))
            ymax = min(H, b_ymin + int((ymin_w + h_w) / scale_y))
            xmin = max(0, b_xmin + int(xmin_w / scale_x))
            xmax = min(W, b_xmin + int((xmin_w + w_w) / scale_x))
            
            # Add small padding
            pad_y = max(10, int((ymax - ymin) * 0.15))
            pad_x = max(10, int((xmax - xmin) * 0.15))
            ymin = max(0, ymin - pad_y)
            ymax = min(H, ymax + pad_y)
            xmin = max(0, xmin - pad_x)
            xmax = min(W, xmax + pad_x)
            
            p = score_proposal_box(image_gray, (ymin, xmin, ymax, xmax), None, source_name=source_name, base_confidence=0.62)
            props.append(p)
            
        return props

    def generate(self, image_gray: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        crop_u8, roi_bbox, tissue_mask, scale_y, scale_x = self._get_working_roi(image_gray)
        all_props = []
        
        # 1. Top-Hat
        if self.mode in ('tophat', 'all'):
            th_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tophat = cv2.morphologyEx(crop_u8, cv2.MORPH_TOPHAT, th_kernel)
            tophat_tissue = tophat[tissue_mask > 0]
            thresh_th = np.percentile(tophat_tissue, 99.2) if len(tophat_tissue) > 0 else 30
            bin_th = ((tophat >= max(20, thresh_th)) & (tissue_mask > 0)).astype(np.uint8)
            all_props.extend(self._extract_clusters(image_gray, bin_th, roi_bbox, scale_y, scale_x, 'calc_tophat'))
            
        # 2. High-Pass
        if self.mode in ('highpass', 'all'):
            blur = cv2.GaussianBlur(crop_u8, (15, 15), 3.0)
            hp = np.maximum(0, crop_u8.astype(np.int16) - blur.astype(np.int16)).astype(np.uint8)
            hp_tissue = hp[tissue_mask > 0]
            thresh_hp = np.percentile(hp_tissue, 99.0) if len(hp_tissue) > 0 else 25
            bin_hp = ((hp >= max(15, thresh_hp)) & (tissue_mask > 0)).astype(np.uint8)
            all_props.extend(self._extract_clusters(image_gray, bin_hp, roi_bbox, scale_y, scale_x, 'calc_highpass'))
            
        # 3. Difference-of-Gaussians (DoG)
        if self.mode in ('dog', 'all'):
            g1 = cv2.GaussianBlur(crop_u8, (5, 5), 1.0)
            g2 = cv2.GaussianBlur(crop_u8, (15, 15), 3.0)
            dog = np.maximum(0, g1.astype(np.int16) - g2.astype(np.int16)).astype(np.uint8)
            dog_tissue = dog[tissue_mask > 0]
            thresh_dog = np.percentile(dog_tissue, 99.0) if len(dog_tissue) > 0 else 20
            bin_dog = ((dog >= max(12, thresh_dog)) & (tissue_mask > 0)).astype(np.uint8)
            all_props.extend(self._extract_clusters(image_gray, bin_dog, roi_bbox, scale_y, scale_x, 'calc_dog'))
            
        # 4. Local Contrast / Variance
        if self.mode in ('contrast', 'all'):
            crop_f = crop_u8.astype(np.float32) / 255.0
            mean = cv2.blur(crop_f, (9, 9))
            sq_mean = cv2.blur(crop_f ** 2, (9, 9))
            std = np.sqrt(np.maximum(0.0, sq_mean - mean ** 2))
            std_tissue = std[tissue_mask > 0]
            thresh_std = np.percentile(std_tissue, 98.8) if len(std_tissue) > 0 else 0.05
            bin_std = ((std >= max(0.03, thresh_std)) & (tissue_mask > 0)).astype(np.uint8)
            all_props.extend(self._extract_clusters(image_gray, bin_std, roi_bbox, scale_y, scale_x, 'calc_contrast'))
            
        return non_max_suppression_proposals(all_props, iou_thresh=0.35, top_k=top_k)


# ==============================================================================
# 7. SOURCE D: LOCAL CONTRAST / HIGH-FREQUENCY SALIENCY PROPOSALS
# ==============================================================================
class SaliencyProposalGenerator:
    def __init__(self, working_res: int = 512):
        self.working_res = working_res

    def generate(self, image_gray: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        H, W = image_gray.shape
        (b_ymin, b_xmin, b_ymax, b_xmax), _ = extract_breast_bounding_box(image_gray)
        b_h = max(32, b_ymax - b_ymin)
        b_w = max(32, b_xmax - b_xmin)
        
        breast_crop = image_gray[b_ymin:b_ymax, b_xmin:b_xmax]
        scale_y = self.working_res / float(b_h)
        scale_x = self.working_res / float(b_w)
        
        crop_512 = cv2.resize(breast_crop, (self.working_res, self.working_res), interpolation=cv2.INTER_LINEAR)
        sobelx = cv2.Sobel(crop_512, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(crop_512, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        
        blurred_grad = cv2.GaussianBlur(grad_mag, (25, 25), 5.0)
        thresh = np.percentile(blurred_grad, 96.0)
        bin_mask = (blurred_grad >= max(0.03, thresh)).astype(np.uint8)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        props = []
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < 30:
                continue
            xmin_w = int(stats[label_idx, cv2.CC_STAT_LEFT])
            ymin_w = int(stats[label_idx, cv2.CC_STAT_TOP])
            w_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h_w = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            
            ymin = max(0, b_ymin + int(ymin_w / scale_y))
            ymax = min(H, b_ymin + int((ymin_w + h_w) / scale_y))
            xmin = max(0, b_xmin + int(xmin_w / scale_x))
            xmax = min(W, b_xmin + int((xmin_w + w_w) / scale_x))
            
            p = score_proposal_box(image_gray, (ymin, xmin, ymax, xmax), None, source_name='saliency_gradient', base_confidence=0.55)
            props.append(p)
            
        return non_max_suppression_proposals(props, iou_thresh=0.35, top_k=top_k)


# ==============================================================================
# 8. SOURCE E: CONTEXTUAL CANDIDATE BANK (RECOVERY BASELINE)
# ==============================================================================
class ContextualCandidateBank:
    def generate(self, image_gray: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        H, W = image_gray.shape
        (b_ymin, b_xmin, b_ymax, b_xmax), _ = extract_breast_bounding_box(image_gray)
        b_h = b_ymax - b_ymin
        b_w = b_xmax - b_xmin
        box_s_y = max(64, int(b_h * 0.35))
        box_s_x = max(64, int(b_w * 0.35))
        
        cy = (b_ymin + b_ymax) // 2
        cx = (b_xmin + b_xmax) // 2
        
        centers = [
            (cy, cx, 'context_center'),
            (b_ymin + b_h // 4, cx, 'context_upper'),
            (b_ymax - b_h // 4, cx, 'context_lower'),
            (cy, b_xmin + b_w // 4, 'context_left'),
            (cy, b_xmax - b_w // 4, 'context_right')
        ]
        
        props = []
        for c_y, c_x, s_name in centers:
            ymin = max(0, c_y - box_s_y // 2)
            ymax = min(H, c_y + box_s_y // 2)
            xmin = max(0, c_x - box_s_x // 2)
            xmax = min(W, c_x + box_s_x // 2)
            p = score_proposal_box(image_gray, (ymin, xmin, ymax, xmax), None, source_name=s_name, base_confidence=0.30)
            props.append(p)
            
        return props[:top_k]


# ==============================================================================
# 9. SOURCE F: RANDOM CROP CONTROL (FIXED SEED)
# ==============================================================================
class RandomCropBaseline:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(self, image_gray: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        H, W = image_gray.shape
        (b_ymin, b_xmin, b_ymax, b_xmax), _ = extract_breast_bounding_box(image_gray)
        b_h = max(64, b_ymax - b_ymin)
        b_w = max(64, b_xmax - b_xmin)
        box_s_y = max(64, int(b_h * 0.30))
        box_s_x = max(64, int(b_w * 0.30))
        
        props = []
        for i in range(top_k):
            rand_y = self.rng.randint(b_ymin, max(b_ymin + 1, b_ymax - box_s_y))
            rand_x = self.rng.randint(b_xmin, max(b_xmin + 1, b_xmax - box_s_x))
            ymin = rand_y
            ymax = min(H, rand_y + box_s_y)
            xmin = rand_x
            xmax = min(W, rand_x + box_s_x)
            p = score_proposal_box(image_gray, (ymin, xmin, ymax, xmax), None, source_name=f'random_crop_{i}', base_confidence=0.20)
            props.append(p)
        return props


# ==============================================================================
# 10. SOURCE G: FIXED SPATIAL GRID BASELINE
# ==============================================================================
class FixedGridBaseline:
    def generate(self, image_gray: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        H, W = image_gray.shape
        (b_ymin, b_xmin, b_ymax, b_xmax), _ = extract_breast_bounding_box(image_gray)
        b_h = b_ymax - b_ymin
        b_w = b_xmax - b_xmin
        
        box_h = int(b_h / 2.0)
        box_w = int(b_w / 2.0)
        
        grid_positions = [
            (b_ymin, b_xmin, 'grid_top_left'),
            (b_ymin, b_xmin + box_w, 'grid_top_right'),
            (b_ymin + box_h, b_xmin, 'grid_bottom_left'),
            (b_ymin + box_h, b_xmin + box_w, 'grid_bottom_right'),
            (b_ymin + box_h // 2, b_xmin + box_w // 2, 'grid_center')
        ]
        
        props = []
        for y, x, s_name in grid_positions:
            ymin = max(0, y)
            ymax = min(H, y + box_h)
            xmin = max(0, x)
            xmax = min(W, x + box_w)
            p = score_proposal_box(image_gray, (ymin, xmin, ymax, xmax), None, source_name=s_name, base_confidence=0.25)
            props.append(p)
        return props[:top_k]
