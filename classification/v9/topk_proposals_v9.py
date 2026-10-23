"""
================================================================================
V9 Multi-Source Proposal Generator & Label-Free Ranking Module
================================================================================
Implements strictly image-derived multi-source candidate proposal generation
and label-free feature ranking for mammographic lesion localization:

- Source A: V6 U-KAN at 512x512 (Multi-Threshold Connected Components)
- Source B: V6 U-KAN at 768x768 (Multi-Threshold CCs, Coordinated Scaled)
- Source C: High-Resolution Detail Pathway (Laplacian / Gradient Energy)
- Source D: Calcification-Sensitive Pathway (Multiscale Top-Hat + DoG)
- Source E: Breast-Region Constrained Spatial Filtering & Background Suppression
- Source F: Multi-Scale Cross-Resolution Agreement Feature

STRICT GOVERNANCE:
- 0 Ground-Truth information used during candidate generation or ranking
- 0 Pathology / label information used
- 100% Deterministic and reproducible
================================================================================
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2


def detect_breast_mask(img_512: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Deterministic, label-free breast parenchyma mask and bounding box detection.
    Returns: (binary_breast_mask, (ymin, xmin, ymax, xmax))
    """
    blurred = cv2.GaussianBlur(img_512, (9, 9), 0)
    u8 = (np.clip(blurred, 0, 1) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if num_labels <= 1:
        mask = (img_512 > 0.05).astype(np.uint8)
        return mask, (0, 0, 512, 512)
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = int(np.argmax(areas)) + 1
    x = int(stats[max_idx, cv2.CC_STAT_LEFT])
    y = int(stats[max_idx, cv2.CC_STAT_TOP])
    w = int(stats[max_idx, cv2.CC_STAT_WIDTH])
    h = int(stats[max_idx, cv2.CC_STAT_HEIGHT])
    
    breast_mask = (labels == max_idx).astype(np.uint8)
    
    pad_x = int(w * 0.05)
    pad_y = int(h * 0.05)
    ymin = max(0, y - pad_y)
    xmin = max(0, x - pad_x)
    ymax = min(512, y + h + pad_y)
    xmax = min(512, x + w + pad_x)
    
    if (ymax - ymin) < 64 or (xmax - xmin) < 64:
        return breast_mask, (0, 0, 512, 512)
        
    return breast_mask, (ymin, xmin, ymax, xmax)


def compute_box_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes (ymin, xmin, ymax, xmax)."""
    yA_min, xA_min, yA_max, xA_max = boxA
    yB_min, xB_min, yB_max, xB_max = boxB

    y_inter_min = max(yA_min, yB_min)
    x_inter_min = max(xA_min, xB_min)
    y_inter_max = min(yA_max, yB_max)
    x_inter_max = min(xmax_a := xA_max, xB_max)
    y_inter_min_clamp = max(yA_min, yB_min)
    y_inter_max_clamp = min(yA_max, yB_max)
    x_inter_min_clamp = max(xA_min, xB_min)
    x_inter_max_clamp = min(xA_max, xB_max)

    inter_h = max(0, y_inter_max_clamp - y_inter_min_clamp)
    inter_w = max(0, x_inter_max_clamp - x_inter_min_clamp)
    inter_area = inter_h * inter_w

    areaA = max(1, (yA_max - yA_min) * (xA_max - xA_min))
    areaB = max(1, (yB_max - yB_min) * (xB_max - xB_min))
    union_area = areaA + areaB - inter_area

    return float(inter_area / max(1, union_area))


def apply_safety_margin(bbox: Tuple[int, int, int, int], margin_ratio: float = 0.20, min_size: int = 48, max_dim: int = 512) -> Tuple[int, int, int, int]:
    """Adds a deterministic safety margin to candidate bounding box and clips to image boundaries."""
    ymin, xmin, ymax, xmax = bbox
    h = ymax - ymin
    w = xmax - xmin
    
    pad_h = max(int(h * margin_ratio), (min_size - h) // 2 if h < min_size else 0)
    pad_w = max(int(w * margin_ratio), (min_size - w) // 2 if w < min_size else 0)
    
    n_ymin = max(0, ymin - pad_h)
    n_xmin = max(0, xmin - pad_w)
    n_ymax = min(max_dim, ymax + pad_h)
    n_xmax = min(max_dim, xmax + pad_w)
    
    return (n_ymin, n_xmin, n_ymax, n_xmax)


class V9MultiSourceProposalGenerator:
    """
    Comprehensive V9 Proposal Generator & Label-Free Ranking Pipeline.
    Combines:
    - Multi-scale U-KAN (512, 768)
    - Multi-threshold Connected Components (p in [0.10, 0.20, 0.35, 0.50])
    - Calcification-sensitive morphological filters
    - High-resolution detail / Laplacian response
    - Breast-region consistency & distance constraints
    - Cross-scale spatial agreement
    - Deterministic label-free ranking & NMS deduplication
    """
    def __init__(
        self,
        min_area: int = 12,
        thresholds: List[float] = [0.10, 0.20, 0.35, 0.50],
        nms_iou_thresh: float = 0.50,
        margin_ratio: float = 0.20
    ):
        self.min_area = min_area
        self.thresholds = thresholds
        self.nms_iou_thresh = nms_iou_thresh
        self.margin_ratio = margin_ratio

    def extract_ukan_candidates(
        self,
        prob_map: np.ndarray,
        image_512: np.ndarray,
        source_name: str,
        scale_factor: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Extracts multi-threshold connected components from a U-KAN probability map.
        scale_factor: converts coordinates to 512x512 space (e.g. 512/768 for 768 map).
        """
        H, W = prob_map.shape
        candidates = []
        
        for thresh in self.thresholds:
            bin_map = (prob_map >= thresh).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_map, connectivity=8)
            
            for lbl in range(1, num_labels):
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area < self.min_area:
                    continue
                
                raw_xmin = int(stats[lbl, cv2.CC_STAT_LEFT])
                raw_ymin = int(stats[lbl, cv2.CC_STAT_TOP])
                raw_w = int(stats[lbl, cv2.CC_STAT_WIDTH])
                raw_h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
                
                # Scale coordinates to 512x512 space
                xmin = int(round(raw_xmin * scale_factor))
                ymin = int(round(raw_ymin * scale_factor))
                xmax = min(512, int(round((raw_xmin + raw_w) * scale_factor)))
                ymax = min(512, int(round((raw_ymin + raw_h) * scale_factor)))
                
                # Apply deterministic safety margin
                safe_box = apply_safety_margin((ymin, xmin, ymax, xmax), margin_ratio=self.margin_ratio, min_size=48, max_dim=512)
                
                comp_mask = (labels == lbl).astype(np.uint8)
                comp_probs = prob_map[comp_mask > 0]
                mean_p = float(np.mean(comp_probs)) if len(comp_probs) > 0 else 0.0
                max_p = float(np.max(comp_probs)) if len(comp_probs) > 0 else 0.0
                
                cy = float(centroids[lbl][1] * scale_factor)
                cx = float(centroids[lbl][0] * scale_factor)
                
                # Shape features
                contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                perimeter = cv2.arcLength(contours[0], True) if len(contours) > 0 else 0.0
                compactness = float(4.0 * np.pi * area / (perimeter ** 2 + 1e-6)) if perimeter > 0 else 0.0
                
                aspect_ratio = float(raw_w / max(1, raw_h))
                eccentricity = abs(1.0 - aspect_ratio) / (1.0 + aspect_ratio + 1e-6)
                
                candidates.append({
                    'source': f"{source_name}_t{int(thresh*100)}",
                    'base_source': source_name,
                    'threshold': thresh,
                    'bbox': safe_box,
                    'raw_bbox': (ymin, xmin, ymax, xmax),
                    'component_area': int(area * (scale_factor ** 2)),
                    'box_area': int((safe_box[2] - safe_box[0]) * (safe_box[3] - safe_box[1])),
                    'mean_probability': mean_p,
                    'max_probability': max_p,
                    'confidence': float(0.6 * mean_p + 0.4 * max_p),
                    'centroid': (cy, cx),
                    'width': safe_box[3] - safe_box[1],
                    'height': safe_box[2] - safe_box[0],
                    'aspect_ratio': aspect_ratio,
                    'compactness': compactness,
                    'eccentricity': eccentricity
                })
                
        return candidates

    def extract_calcification_candidates(
        self,
        image_512: np.ndarray,
        breast_mask: np.ndarray,
        k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Extracts high-frequency calcification clusters using multiscale Top-Hat & Difference-of-Gaussians.
        Operates strictly inside breast parenchyma to prevent border artifacts.
        """
        u8 = (np.clip(image_512, 0, 1) * 255).astype(np.uint8)
        
        # Multiscale Top-Hat
        k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        tophat_s = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, k_small).astype(np.float32) / 255.0
        tophat_m = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, k_med).astype(np.float32) / 255.0
        
        # Difference of Gaussians (DoG)
        g1 = cv2.GaussianBlur(image_512, (3, 3), 1.0)
        g2 = cv2.GaussianBlur(image_512, (9, 9), 3.0)
        dog = np.maximum(0, g1 - g2)
        
        # Combined calcification salience map
        calc_map = 0.45 * tophat_s + 0.35 * tophat_m + 0.20 * dog
        
        # Mask strictly inside breast parenchyma (erode breast mask slightly to avoid skin edge)
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        inner_breast = cv2.erode(breast_mask, erode_k)
        calc_masked = calc_map * inner_breast.astype(np.float32)
        
        # Threshold at high percentile of inner breast
        inner_vals = calc_masked[inner_breast > 0]
        if len(inner_vals) == 0:
            return []
        
        thresh_val = float(np.percentile(inner_vals, 98.2))
        bin_calc = (calc_masked >= thresh_val).astype(np.uint8)
        
        # Cluster nearby calcification spots
        cluster_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        bin_clustered = cv2.dilate(bin_calc, cluster_k)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clustered, connectivity=8)
        candidates = []
        
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 6 or area > 35000:
                continue
                
            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            w = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            
            cx = float(centroids[lbl][0])
            cy = float(centroids[lbl][1])
            
            bw = max(48, int(w * 1.6))
            bh = max(48, int(h * 1.6))
            ymin = max(0, int(cy - bh / 2.0))
            xmin = max(0, int(cx - bw / 2.0))
            ymax = min(512, ymin + bh)
            xmax = min(512, xmin + bw)
            
            safe_box = apply_safety_margin((ymin, xmin, ymax, xmax), margin_ratio=0.15, min_size=48, max_dim=512)
            
            comp_crop = calc_masked[safe_box[0]:safe_box[2], safe_box[1]:safe_box[3]]
            mean_c = float(np.mean(comp_crop)) if comp_crop.size > 0 else 0.0
            max_c = float(np.max(comp_crop)) if comp_crop.size > 0 else 0.0
            score = float(0.6 * max_c + 0.4 * mean_c * 5.0)
            
            candidates.append({
                'source': 'calc_multiscale_tophat',
                'base_source': 'calc_pathway',
                'threshold': thresh_val,
                'bbox': safe_box,
                'raw_bbox': (ymin, xmin, ymax, xmax),
                'component_area': int(area),
                'box_area': int((safe_box[2] - safe_box[0]) * (safe_box[3] - safe_box[1])),
                'mean_probability': mean_c,
                'max_probability': max_c,
                'confidence': score,
                'centroid': (cy, cx),
                'width': safe_box[3] - safe_box[1],
                'height': safe_box[2] - safe_box[0],
                'aspect_ratio': float(w / max(1, h)),
                'compactness': 0.8,
                'eccentricity': 0.2
            })
            
        candidates.sort(key=lambda c: c['confidence'], reverse=True)
        return candidates[:k]

    def extract_detail_candidates(
        self,
        image_512: np.ndarray,
        breast_mask: np.ndarray,
        k: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Extracts high-resolution structural detail / Laplacian energy clusters.
        Captures architectural distortion and spiculated masses.
        """
        g = cv2.GaussianBlur(image_512, (5, 5), 1.2)
        lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
        lap = lap / (np.max(lap) + 1e-8)
        
        # Mask strictly inside breast parenchyma
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        inner_breast = cv2.erode(breast_mask, erode_k)
        lap_masked = lap * inner_breast.astype(np.float32)
        
        inner_vals = lap_masked[inner_breast > 0]
        if len(inner_vals) == 0:
            return []
            
        thresh_val = float(np.percentile(inner_vals, 97.5))
        bin_lap = (lap_masked >= thresh_val).astype(np.uint8)
        
        dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bin_clustered = cv2.dilate(bin_lap, dil_k)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clustered, connectivity=8)
        candidates = []
        
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 15 or area > 40000:
                continue
                
            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            w = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            
            cx = float(centroids[lbl][0])
            cy = float(centroids[lbl][1])
            
            bw = max(56, int(w * 1.5))
            bh = max(56, int(h * 1.5))
            ymin = max(0, int(cy - bh / 2.0))
            xmin = max(0, int(cx - bw / 2.0))
            ymax = min(512, ymin + bh)
            xmax = min(512, xmin + bw)
            
            safe_box = apply_safety_margin((ymin, xmin, ymax, xmax), margin_ratio=0.15, min_size=48, max_dim=512)
            crop = lap_masked[safe_box[0]:safe_box[2], safe_box[1]:safe_box[3]]
            score = float(np.mean(crop) * 3.0 + np.max(crop)) if crop.size > 0 else 0.0
            
            candidates.append({
                'source': 'detail_laplacian_energy',
                'base_source': 'detail_pathway',
                'threshold': thresh_val,
                'bbox': safe_box,
                'raw_bbox': (ymin, xmin, ymax, xmax),
                'component_area': int(area),
                'box_area': int((safe_box[2] - safe_box[0]) * (safe_box[3] - safe_box[1])),
                'mean_probability': float(np.mean(crop)) if crop.size > 0 else 0.0,
                'max_probability': float(np.max(crop)) if crop.size > 0 else 0.0,
                'confidence': score,
                'centroid': (cy, cx),
                'width': safe_box[3] - safe_box[1],
                'height': safe_box[2] - safe_box[0],
                'aspect_ratio': float(w / max(1, h)),
                'compactness': 0.75,
                'eccentricity': 0.25
            })
            
        candidates.sort(key=lambda c: c['confidence'], reverse=True)
        return candidates[:k]

    def compute_label_free_features(
        self,
        candidate: Dict[str, Any],
        image_512: np.ndarray,
        p_512_map: np.ndarray,
        p_768_map_512: np.ndarray,
        breast_mask: np.ndarray,
        calc_salience_map: np.ndarray,
        detail_map: np.ndarray,
        other_candidates: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Computes the complete set of strictly image-derived features for ranking:
        A. Mean U-KAN probability
        B. Maximum U-KAN probability
        C. Connected-component area
        D. Component compactness
        E. Component eccentricity
        F. Local contrast
        G. Local intensity statistics
        H. High-frequency/detail response
        I. Calcification response
        J. Distance from breast boundary
        K. Breast-region consistency
        L. Multi-scale agreement
        M. Candidate size plausibility
        N. Edge/texture statistics
        """
        ymin, xmin, ymax, xmax = candidate['bbox']
        H, W = image_512.shape
        
        # 1. Probability Maps in Box
        box_p512 = p_512_map[ymin:ymax, xmin:xmax]
        box_p768 = p_768_map_512[ymin:ymax, xmin:xmax]
        mean_p512 = float(np.mean(box_p512)) if box_p512.size > 0 else 0.0
        max_p512 = float(np.max(box_p512)) if box_p512.size > 0 else 0.0
        mean_p768 = float(np.mean(box_p768)) if box_p768.size > 0 else 0.0
        max_p768 = float(np.max(box_p768)) if box_p768.size > 0 else 0.0
        
        mean_prob = 0.55 * mean_p512 + 0.45 * mean_p768
        max_prob = 0.55 * max_p512 + 0.45 * max_p768
        
        # 2. Local Contrast vs. Surrounding Margin
        pad_y = max(12, (ymax - ymin) // 2)
        pad_x = max(12, (xmax - xmin) // 2)
        s_ymin, s_ymax = max(0, ymin - pad_y), min(H, ymax + pad_y)
        s_xmin, s_xmax = max(0, xmin - pad_x), min(W, xmax + pad_x)
        
        box_img = image_512[ymin:ymax, xmin:xmax]
        surr_img = image_512[s_ymin:s_ymax, s_xmin:s_xmax]
        surr_mask = np.ones((s_ymax - s_ymin, s_xmax - s_xmin), dtype=bool)
        surr_mask[(ymin - s_ymin):(ymax - s_ymin), (xmin - s_xmin):(xmax - s_xmin)] = False
        
        box_mean_val = float(np.mean(box_img)) if box_img.size > 0 else 0.0
        box_std_val = float(np.std(box_img)) if box_img.size > 0 else 0.0
        surr_mean_val = float(np.mean(surr_img[surr_mask])) if np.sum(surr_mask) > 0 else box_mean_val
        local_contrast = float(box_mean_val - surr_mean_val)
        norm_contrast = float(np.clip((local_contrast + 0.2) / 0.5, 0.0, 1.0))
        
        # 3. High-Frequency & Calcification Responses
        box_calc = calc_salience_map[ymin:ymax, xmin:xmax]
        calc_response = float(np.mean(box_calc) * 2.0 + np.max(box_calc)) if box_calc.size > 0 else 0.0
        norm_calc = float(np.clip(calc_response / 1.5, 0.0, 1.0))
        
        box_detail = detail_map[ymin:ymax, xmin:xmax]
        detail_response = float(np.mean(box_detail) * 2.0 + np.max(box_detail)) if box_detail.size > 0 else 0.0
        norm_detail = float(np.clip(detail_response / 1.5, 0.0, 1.0))
        
        # 4. Breast-Region Consistency & Distance from Boundary
        box_breast = breast_mask[ymin:ymax, xmin:xmax]
        breast_consistency = float(np.mean(box_breast)) if box_breast.size > 0 else 0.0
        
        dist_transform = cv2.distanceTransform(breast_mask, cv2.DIST_L2, 5)
        cy, cx = candidate['centroid']
        cy_i, cx_i = int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))
        dist_boundary = float(dist_transform[cy_i, cx_i])
        norm_dist_boundary = float(np.clip(dist_boundary / 64.0, 0.0, 1.0))
        
        # 5. Multi-Scale Agreement Feature (IoU with any detection from other scale)
        is_source_512 = '512' in candidate['source']
        target_prefix = '768' if is_source_512 else '512'
        best_agreement_iou = 0.0
        for oc in other_candidates:
            if oc is not candidate and target_prefix in oc.get('source', ''):
                iou = compute_box_iou(candidate['bbox'], oc['bbox'])
                if iou > best_agreement_iou:
                    best_agreement_iou = iou
        multiscale_agreement = float(best_agreement_iou)
        
        # 6. Size Plausibility (penalizes trivial tiny <300px or huge >80000px boxes)
        box_area = candidate['box_area']
        log_area = math.log(max(10, box_area))
        target_log_area = math.log(12000.0)
        size_plausibility = float(math.exp(-((log_area - target_log_area) ** 2) / (2.0 * (1.8 ** 2))))
        
        # 7. Shape Features
        compactness = float(candidate.get('compactness', 0.8))
        eccentricity = float(candidate.get('eccentricity', 0.2))
        
        # 8. Composite Label-Free Ranking Score
        # STRICT FORMULA PREDECLARED BEFORE GT EVALUATION:
        composite_score = (
            0.35 * mean_prob +
            0.20 * max_prob +
            0.15 * multiscale_agreement +
            0.10 * norm_calc +
            0.08 * norm_contrast +
            0.05 * norm_detail +
            0.04 * size_plausibility +
            0.03 * breast_consistency
        )
        
        return {
            'ranking_score': float(composite_score),
            'mean_probability': mean_prob,
            'max_probability': max_prob,
            'multiscale_agreement': multiscale_agreement,
            'calc_response': norm_calc,
            'local_contrast': norm_contrast,
            'detail_response': norm_detail,
            'breast_consistency': breast_consistency,
            'dist_boundary': norm_dist_boundary,
            'size_plausibility': size_plausibility,
            'compactness': compactness,
            'eccentricity': eccentricity,
            'box_mean_val': box_mean_val,
            'box_std_val': box_std_val
        }

    def deduplicate_proposals(
        self,
        candidate_list: List[Dict[str, Any]],
        k: int = 5,
        iou_thresh: float = 0.50
    ) -> List[Dict[str, Any]]:
        """
        Deterministic label-free Non-Maximum Suppression (NMS).
        Greedily selects top candidates maximizing score while enforcing spatial diversity.
        """
        if len(candidate_list) == 0:
            return []
            
        sorted_cands = sorted(candidate_list, key=lambda c: c.get('ranking_score', c.get('confidence', 0.0)), reverse=True)
        selected = []
        
        for cand in sorted_cands:
            box = cand['bbox']
            overlap = False
            for s in selected:
                if compute_box_iou(box, s['bbox']) > iou_thresh:
                    overlap = True
                    break
            if not overlap:
                selected.append(cand)
            if len(selected) == k:
                break
                
        # If fewer than k, fill with remaining non-identical
        if len(selected) < k:
            for cand in sorted_cands:
                if cand not in selected:
                    selected.append(cand)
                if len(selected) == k:
                    break
                    
        return selected

    def generate_all_arms(
        self,
        image_512: np.ndarray,
        image_768: np.ndarray,
        p_512_map: np.ndarray,
        p_768_map: np.ndarray,
        k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generates candidate proposals for all predefined arms:
        - Arm 1 (Single-Scale 512 Baseline): Single threshold p=0.15 CCs
        - Arm 2 (Single-Scale 768 Baseline): Single threshold p=0.15 CCs
        - Arm 3 (Multi-Threshold 512 CC): p in [0.10, 0.20, 0.35, 0.50]
        - Arm 4 (Multi-Scale 512+768 Naive NMS)
        - Arm 5 (Calcification Pathway Alone)
        - Arm 6 (High-Res Detail Pathway Alone)
        - Arm 7 (Multi-Scale Multi-Threshold + Breast)
        - Arm 8 (Multi-Scale + Calc + Detail Naive)
        - Arm 9 (V9 Full Multi-Source + Ranking)
        - Arm 10 (Random Baseline)
        - Arm 11 (Fixed Grid Baseline)
        """
        p_768_to_512 = cv2.resize(p_768_map, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        
        # 1. Breast Mask & Parenchyma Detection
        breast_mask, breast_bbox = detect_breast_mask(image_512)
        
        # 2. Salience Maps for Detail & Calcification
        u8_512 = (np.clip(image_512, 0, 1) * 255).astype(np.uint8)
        k_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tophat_map = cv2.morphologyEx(u8_512, cv2.MORPH_TOPHAT, k_tophat).astype(np.float32) / 255.0
        
        g_lap = cv2.GaussianBlur(image_512, (5, 5), 1.2)
        lap_map = np.abs(cv2.Laplacian(g_lap, cv2.CV_32F, ksize=3))
        lap_map = lap_map / (np.max(lap_map) + 1e-8)
        
        # --- SOURCE EXTRACTIONS ---
        
        # Source A: 512 Multi-Threshold CCs
        cands_512 = self.extract_ukan_candidates(p_512_map, image_512, 'ukan_512', scale_factor=1.0)
        
        # Source B: 768 Multi-Threshold CCs
        cands_768 = self.extract_ukan_candidates(p_768_map, image_512, 'ukan_768', scale_factor=512.0 / 768.0)
        
        # Source C: Detail Pathway
        cands_detail = self.extract_detail_candidates(image_512, breast_mask, k=6)
        
        # Source D: Calcification Pathway
        cands_calc = self.extract_calcification_candidates(image_512, breast_mask, k=8)
        
        # --- ABLATION ARMS GENERATION ---
        
        # Arm 1: V6 512 Single Threshold (p >= 0.15) Baseline
        bin_512_single = (p_512_map >= 0.15).astype(np.uint8)
        num_l, lbls, st, cents = cv2.connectedComponentsWithStats(bin_512_single, connectivity=8)
        cands_arm1 = []
        for l in range(1, num_l):
            if st[l, cv2.CC_STAT_AREA] >= self.min_area:
                xmin, ymin, w, h = st[l, cv2.CC_STAT_LEFT], st[l, cv2.CC_STAT_TOP], st[l, cv2.CC_STAT_WIDTH], st[l, cv2.CC_STAT_HEIGHT]
                sb = apply_safety_margin((ymin, xmin, min(512, ymin+h), min(512, xmin+w)), margin_ratio=0.10)
                cp = p_512_map[lbls == l]
                cands_arm1.append({
                    'source': 'v6_512_single',
                    'bbox': sb,
                    'ranking_score': float(np.mean(cp)),
                    'confidence': float(np.mean(cp)),
                    'area': (sb[2]-sb[0])*(sb[3]-sb[1]),
                    'centroid': (float(cents[l][1]), float(cents[l][0]))
                })
        cands_arm1.sort(key=lambda c: c['confidence'], reverse=True)
        props_arm1 = self._pad_with_fallback(cands_arm1[:k], image_512, p_512_map, k)
        
        # Arm 2: V6 768 Single Threshold (p >= 0.15) Baseline
        bin_768_single = (p_768_to_512 >= 0.15).astype(np.uint8)
        num_l7, lbls7, st7, cents7 = cv2.connectedComponentsWithStats(bin_768_single, connectivity=8)
        cands_arm2 = []
        for l in range(1, num_l7):
            if st7[l, cv2.CC_STAT_AREA] >= self.min_area:
                xmin, ymin, w, h = st7[l, cv2.CC_STAT_LEFT], st7[l, cv2.CC_STAT_TOP], st7[l, cv2.CC_STAT_WIDTH], st7[l, cv2.CC_STAT_HEIGHT]
                sb = apply_safety_margin((ymin, xmin, min(512, ymin+h), min(512, xmin+w)), margin_ratio=0.10)
                cp = p_768_to_512[lbls7 == l]
                cands_arm2.append({
                    'source': 'v6_768_single',
                    'bbox': sb,
                    'ranking_score': float(np.mean(cp)),
                    'confidence': float(np.mean(cp)),
                    'area': (sb[2]-sb[0])*(sb[3]-sb[1]),
                    'centroid': (float(cents7[l][1]), float(cents7[l][0]))
                })
        cands_arm2.sort(key=lambda c: c['confidence'], reverse=True)
        props_arm2 = self._pad_with_fallback(cands_arm2[:k], image_512, p_768_to_512, k)
        
        # Arm 3: Multi-Threshold 512 CCs
        props_arm3 = self.deduplicate_proposals(cands_512, k=k, iou_thresh=0.55)
        props_arm3 = self._pad_with_fallback(props_arm3, image_512, p_512_map, k)
        
        # Arm 4: Multi-Scale 512 + 768 Naive NMS (V8 Phase 1 Arm E)
        comb_4 = list(cands_arm1) + list(cands_arm2)
        props_arm4 = self.deduplicate_proposals(comb_4, k=k, iou_thresh=0.65)
        props_arm4 = self._pad_with_fallback(props_arm4, image_512, p_512_map, k)
        
        # Arm 5: Calcification Pathway Alone
        props_arm5 = self.deduplicate_proposals(cands_calc, k=k, iou_thresh=0.50)
        props_arm5 = self._pad_with_fallback(props_arm5, image_512, p_512_map, k)
        
        # Arm 6: Detail Pathway Alone
        props_arm6 = self.deduplicate_proposals(cands_detail, k=k, iou_thresh=0.50)
        props_arm6 = self._pad_with_fallback(props_arm6, image_512, p_512_map, k)
        
        # Arm 7: Multi-Scale + Multi-Threshold + Breast Constraint
        all_ukan_cands = cands_512 + cands_768
        cands_arm7 = []
        for c in all_ukan_cands:
            cy, cx = int(c['centroid'][0]), int(c['centroid'][1])
            if 0 <= cy < 512 and 0 <= cx < 512 and breast_mask[cy, cx] > 0:
                cands_arm7.append(c)
        props_arm7 = self.deduplicate_proposals(cands_arm7, k=k, iou_thresh=0.50)
        props_arm7 = self._pad_with_fallback(props_arm7, image_512, p_512_map, k)
        
        # Arm 8: Multi-Scale + Calcification + Detail Pathways (Naive merge)
        comb_8 = cands_512 + cands_768 + cands_calc + cands_detail
        props_arm8 = self.deduplicate_proposals(comb_8, k=k, iou_thresh=0.55)
        props_arm8 = self._pad_with_fallback(props_arm8, image_512, p_512_map, k)
        
        # Arm 9: Primary V9 Full Multi-Source + Label-Free 14-Feature Ranking + NMS
        all_sources_cands = cands_512 + cands_768 + cands_calc + cands_detail
        for cand in all_sources_cands:
            feat_dict = self.compute_label_free_features(
                candidate=cand,
                image_512=image_512,
                p_512_map=p_512_map,
                p_768_map_512=p_768_to_512,
                breast_mask=breast_mask,
                calc_salience_map=tophat_map,
                detail_map=lap_map,
                other_candidates=all_sources_cands
            )
            cand.update(feat_dict)
            
        valid_arm9 = [c for c in all_sources_cands if c.get('breast_consistency', 1.0) >= 0.20]
        if len(valid_arm9) == 0:
            valid_arm9 = all_sources_cands
            
        props_arm9 = self.deduplicate_proposals(valid_arm9, k=k, iou_thresh=self.nms_iou_thresh)
        props_arm9 = self._pad_with_fallback(props_arm9, image_512, p_512_map, k)
        
        # Arm 10: Random Proposal Baseline
        rng = np.random.RandomState(42 + int(np.sum(image_512) * 1000) % 100000)
        props_arm10 = []
        for i in range(k):
            bw = rng.randint(48, 160)
            bh = rng.randint(48, 160)
            xmin = rng.randint(16, 512 - bw - 16)
            ymin = rng.randint(16, 512 - bh - 16)
            props_arm10.append({
                'source': 'random_baseline',
                'bbox': (ymin, xmin, ymin + bh, xmin + bw),
                'ranking_score': float(1.0 / (i + 1)),
                'confidence': float(1.0 / (i + 1)),
                'area': bw * bh,
                'centroid': (ymin + bh / 2.0, xmin + bw / 2.0)
            })
            
        # Arm 11: Fixed-Grid Proposal Baseline
        grid_boxes = [
            (128, 128, 384, 384), # Center (256x256)
            (32, 32, 256, 256),   # Top-Left
            (32, 256, 256, 480),  # Top-Right
            (256, 32, 480, 256),  # Bottom-Left
            (256, 256, 480, 480)  # Bottom-Right
        ]
        props_arm11 = []
        for i, gbox in enumerate(grid_boxes[:k]):
            ymin, xmin, ymax, xmax = gbox
            props_arm11.append({
                'source': 'fixed_grid_baseline',
                'bbox': gbox,
                'ranking_score': float(1.0 / (i + 1)),
                'confidence': float(1.0 / (i + 1)),
                'area': (ymax - ymin) * (xmax - xmin),
                'centroid': ((ymin + ymax) / 2.0, (xmin + xmax) / 2.0)
            })

        return {
            'Arm 1 (Single-Scale 512 Baseline)': props_arm1,
            'Arm 2 (Single-Scale 768 Baseline)': props_arm2,
            'Arm 3 (Multi-Threshold 512 CC)': props_arm3,
            'Arm 4 (Multi-Scale 512+768 Naive NMS)': props_arm4,
            'Arm 5 (Calcification Pathway Alone)': props_arm5,
            'Arm 6 (High-Res Detail Pathway Alone)': props_arm6,
            'Arm 7 (Multi-Scale Multi-Threshold + Breast)': props_arm7,
            'Arm 8 (Multi-Scale + Calc + Detail Naive)': props_arm8,
            'Arm 9 (V9 Full Multi-Source + Ranking)': props_arm9,
            'Arm 10 (Random Baseline)': props_arm10,
            'Arm 11 (Fixed Grid Baseline)': props_arm11
        }

    def _pad_with_fallback(
        self,
        current_props: List[Dict[str, Any]],
        image_512: np.ndarray,
        prob_map_512: np.ndarray,
        k: int
    ) -> List[Dict[str, Any]]:
        """Ensures exactly K proposals exist by filling with deterministic breast context fallbacks if needed."""
        if len(current_props) >= k:
            return current_props[:k]
            
        final_props = list(current_props)
        needed = k - len(final_props)
        H, W = 512, 512
        box_size = 128
        
        # Strategy 1: Probability Peak
        if needed > 0 and len(final_props) == 0:
            flat_idx = int(np.argmax(prob_map_512))
            peak_y, peak_x = divmod(flat_idx, W)
            half_s = box_size // 2
            ymin = max(0, peak_y - half_s)
            ymax = min(H, peak_y + half_s)
            xmin = max(0, peak_x - half_s)
            xmax = min(W, peak_x + half_s)
            final_props.append({
                'source': 'prob_peak_fallback',
                'bbox': (ymin, xmin, ymax, xmax),
                'ranking_score': float(prob_map_512[peak_y, peak_x]),
                'confidence': float(prob_map_512[peak_y, peak_x]),
                'area': (ymax - ymin) * (xmax - xmin),
                'centroid': (float(peak_y), float(peak_x))
            })
            needed -= 1
            
        # Strategy 2: Parenchyma Centroids & Quadrant offsets
        tissue_mask = image_512 > 0.05
        if np.sum(tissue_mask) > 100:
            tissue_ys, tissue_xs = np.where(tissue_mask)
            cy, cx = int(np.median(tissue_ys)), int(np.median(tissue_xs))
        else:
            cy, cx = 256, 256
            
        offsets = [
            (0, 0),
            (-H // 6, -W // 6),
            (H // 6, W // 6),
            (-H // 6, W // 6),
            (H // 6, -W // 6),
            (0, -W // 5),
            (0, W // 5)
        ]
        
        off_idx = 0
        while len(final_props) < k and off_idx < len(offsets):
            dy, dx = offsets[off_idx]
            off_idx += 1
            f_cy = max(box_size // 2, min(H - box_size // 2, cy + dy))
            f_cx = max(box_size // 2, min(W - box_size // 2, cx + dx))
            half_s = box_size // 2
            ymin = max(0, f_cy - half_s)
            ymax = min(H, f_cy + half_s)
            xmin = max(0, f_cx - half_s)
            xmax = min(W, f_cx + half_s)
            
            final_props.append({
                'source': f'parenchyma_context_fallback_{off_idx}',
                'bbox': (ymin, xmin, ymax, xmax),
                'ranking_score': 0.01,
                'confidence': 0.01,
                'area': (ymax - ymin) * (xmax - xmin),
                'centroid': (float(f_cy), float(f_cx))
            })
            
        return final_props[:k]
