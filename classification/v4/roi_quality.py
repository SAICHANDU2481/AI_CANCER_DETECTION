import numpy as np
import cv2
import math
from typing import Dict, Any, Tuple, Optional, List

QUALITY_FEATURE_NAMES = [
    'predicted_mask_area',
    'normalized_mask_area',
    'bbox_width',
    'bbox_height',
    'bbox_aspect_ratio',
    'bbox_area',
    'mask_confidence',
    'mask_mean_probability',
    'mask_max_probability',
    'mask_probability_std',
    'mask_entropy',
    'connected_component_count',
    'component_compactness',
    'normalized_center_distance',
    'boundary_touch_ratio',
    'mask_fill_ratio'
]

def extract_roi_quality_features(
    prob_map: np.ndarray,
    prob_threshold: float = 0.50,
    min_component_area: int = 10
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """
    Extracts EXACTLY 16 inference-only quality features from the predicted 512x512 probability map.
    Zero access to ground-truth masks, pathology, or test information.
    
    Returns:
        quality_vector (np.ndarray of shape (16,), float32)
        roi_quality_score (float in [0, 1])
        feature_dict (Dict[str, float] of all 16 features)
    """
    # Ensure 2D 512x512 float32 array
    if prob_map.ndim == 3:
        prob_map = prob_map.squeeze()
    prob_map = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
    
    H, W = prob_map.shape
    max_center_dist = math.sqrt((H / 2.0)**2 + (W / 2.0)**2)
    
    # 1. Global map statistics
    mask_mean_prob = float(np.mean(prob_map))
    mask_max_prob = float(np.max(prob_map))
    mask_prob_std = float(np.std(prob_map))
    
    # Global entropy
    eps = 1e-7
    p_safe = np.clip(prob_map, eps, 1.0 - eps)
    entropy_map = -(p_safe * np.log2(p_safe) + (1.0 - p_safe) * np.log2(1.0 - p_safe))
    mask_entropy = float(np.mean(entropy_map))
    
    # Binary foreground mask
    binary_mask = (prob_map >= prob_threshold).astype(np.uint8)
    mask_area = float(np.sum(binary_mask))
    norm_mask_area = float(mask_area / (H * W))
    
    # Empty / Zero foreground fallback
    if mask_area == 0:
        feat_dict = {
            'predicted_mask_area': 0.0,
            'normalized_mask_area': 0.0,
            'bbox_width': 0.0,
            'bbox_height': 0.0,
            'bbox_aspect_ratio': 1.0,
            'bbox_area': 0.0,
            'mask_confidence': 0.0,
            'mask_mean_probability': mask_mean_prob,
            'mask_max_probability': mask_max_prob,
            'mask_probability_std': mask_prob_std,
            'mask_entropy': mask_entropy,
            'connected_component_count': 0.0,
            'component_compactness': 0.0,
            'normalized_center_distance': 1.0,
            'boundary_touch_ratio': 0.0,
            'mask_fill_ratio': 0.0
        }
        vec = np.array([feat_dict[k] for k in QUALITY_FEATURE_NAMES], dtype=np.float32)
        assert vec.shape == (16,), f"Expected 16 features, got {vec.shape}"
        return vec, 0.0, feat_dict
        
    # 2. Foreground confidence
    mask_confidence = float(np.mean(prob_map[binary_mask > 0]))
    
    # 3. Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Filter valid components by min_component_area
    valid_indices = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_component_area]
    cc_count = float(len(valid_indices))
    
    if len(valid_indices) == 0:
        # If all components are smaller than min_component_area, use the largest available foreground component
        valid_indices = [i for i in range(1, num_labels)]
        
    if len(valid_indices) > 0:
        # Find primary (largest) component
        primary_idx = max(valid_indices, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        x = stats[primary_idx, cv2.CC_STAT_LEFT]
        y = stats[primary_idx, cv2.CC_STAT_TOP]
        w = stats[primary_idx, cv2.CC_STAT_WIDTH]
        h = stats[primary_idx, cv2.CC_STAT_HEIGHT]
        c_area = stats[primary_idx, cv2.CC_STAT_AREA]
        cx, cy = centroids[primary_idx]
        
        # Primary component contour compactness
        comp_mask = (labels == primary_idx).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            perimeter = cv2.arcLength(contours[0], True)
            compactness = float(np.clip((4.0 * math.pi * c_area) / (perimeter**2 + 1e-6), 0.0, 1.0))
        else:
            compactness = 0.0
    else:
        y_indices, x_indices = np.where(binary_mask > 0)
        y, x = int(np.min(y_indices)), int(np.min(x_indices))
        h, w = int(np.max(y_indices) - y + 1), int(np.max(x_indices) - x + 1)
        cx, cy = float(x + w / 2.0), float(y + h / 2.0)
        compactness = 0.0
        
    bbox_width = float(w)
    bbox_height = float(h)
    bbox_aspect_ratio = float(bbox_width / (bbox_height + 1e-6))
    bbox_area = float(bbox_width * bbox_height)
    mask_fill_ratio = float(mask_area / (bbox_area + 1e-6))
    
    # 4. Center distance
    center_dist = math.sqrt((cx - (W / 2.0))**2 + (cy - (H / 2.0))**2)
    norm_center_dist = float(np.clip(center_dist / max_center_dist, 0.0, 1.0))
    
    # 5. Boundary touch ratio
    # Perimeter pixels of bbox touching image boundaries
    boundary_touch_pixels = 0
    if x == 0: boundary_touch_pixels += h
    if (x + w) >= W: boundary_touch_pixels += h
    if y == 0: boundary_touch_pixels += w
    if (y + h) >= H: boundary_touch_pixels += w
    total_bbox_perimeter = 2.0 * (w + h)
    boundary_touch_ratio = float(np.clip(boundary_touch_pixels / (total_bbox_perimeter + 1e-6), 0.0, 1.0))
    
    feat_dict = {
        'predicted_mask_area': mask_area,
        'normalized_mask_area': norm_mask_area,
        'bbox_width': bbox_width,
        'bbox_height': bbox_height,
        'bbox_aspect_ratio': bbox_aspect_ratio,
        'bbox_area': bbox_area,
        'mask_confidence': mask_confidence,
        'mask_mean_probability': mask_mean_prob,
        'mask_max_probability': mask_max_prob,
        'mask_probability_std': mask_prob_std,
        'mask_entropy': mask_entropy,
        'connected_component_count': cc_count,
        'component_compactness': compactness,
        'normalized_center_distance': norm_center_dist,
        'boundary_touch_ratio': boundary_touch_ratio,
        'mask_fill_ratio': mask_fill_ratio
    }
    
    # Deterministic Normalized ROI Quality Score in [0, 1]
    raw_score = (
        0.35 * mask_confidence +
        0.25 * compactness +
        0.20 * mask_fill_ratio +
        0.10 * (1.0 - boundary_touch_ratio) +
        0.10 * (1.0 - norm_center_dist)
    )
    # Penalize extreme smallness or fragmentation
    if mask_area < 50:
        raw_score *= (mask_area / 50.0)
    if cc_count > 3:
        raw_score *= 0.80
        
    roi_quality_score = float(np.clip(raw_score, 0.0, 1.0))
    
    vec = np.array([feat_dict[k] for k in QUALITY_FEATURE_NAMES], dtype=np.float32)
    # Check for NaN / Inf fallbacks
    vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    assert vec.shape == (16,), f"Expected 16 features, got {vec.shape}"
    
    return vec, roi_quality_score, feat_dict

class QualityFeatureNormalizer:
    """
    Standardizes the 16-dimensional quality vector using statistics computed strictly from the training cohort.
    """
    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = mean
        self.std = std
        
    def fit(self, quality_matrix: np.ndarray):
        assert quality_matrix.shape[-1] == 16, "Expected Nx16 matrix"
        self.mean = np.mean(quality_matrix, axis=0).astype(np.float32)
        self.std = np.std(quality_matrix, axis=0).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        
    def transform(self, quality_vec: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return quality_vec
        norm_vec = (quality_vec - self.mean) / self.std
        return np.nan_to_num(norm_vec, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)
