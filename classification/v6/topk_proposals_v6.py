"""
================================================================================
V6 Top-K Lesion Proposal Generator
================================================================================
Generates and ranks Top-K candidate lesion proposals from upstream V6 U-KAN
probability maps using strictly inference-available information.

Strict Governance:
- Zero ground-truth mask usage for proposal ranking/generation
- Zero pathology / label usage
- Deterministic ranking based on:
  1. Mean predicted probability in component
  2. Component pixel area
  3. Component peak confidence
  4. Component compactness
  5. Local contrast against surrounding parenchyma
================================================================================
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple


class TopKProposalGeneratorV6:
    """
    Extracts, scores, and selects Top-K candidate lesion proposals from a 2D probability map.
    """
    def __init__(
        self,
        min_area: int = 15,
        threshold: float = 0.50,
        default_k: int = 3,
        fallback_box_size: int = 128
    ):
        self.min_area = min_area
        self.threshold = threshold
        self.default_k = default_k
        self.fallback_box_size = fallback_box_size

    def score_component(
        self,
        component_mask: np.ndarray,
        prob_map: np.ndarray,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Dict[str, Any]:
        """
        Computes deterministic scoring metrics using inference-only signals.
        bbox: (ymin, xmin, ymax, xmax)
        """
        ymin, xmin, ymax, xmax = bbox
        area = int(np.sum(component_mask > 0))
        if area == 0:
            return {
                'score': 0.0,
                'mean_prob': 0.0,
                'area': 0,
                'peak_prob': 0.0,
                'compactness': 0.0,
                'local_contrast': 0.0,
                'bbox': bbox
            }

        comp_probs = prob_map[component_mask > 0]
        mean_prob = float(np.mean(comp_probs))
        peak_prob = float(np.max(comp_probs))
        std_prob = float(np.std(comp_probs))

        # Compactness: 4 * pi * Area / (Perimeter^2)
        contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if len(contours) > 0 else 0.0
        compactness = float(4.0 * np.pi * area / (perimeter ** 2 + 1e-6)) if perimeter > 0 else 0.0

        # Local contrast against surrounding margin
        H, W = image.shape
        pad_y = max(10, (ymax - ymin) // 2)
        pad_x = max(10, (xmax - xmin) // 2)
        surr_ymin, surr_ymax = max(0, ymin - pad_y), min(H, ymax + pad_y)
        surr_xmin, surr_xmax = max(0, xmin - pad_x), min(W, xmax + pad_x)

        surr_patch = image[surr_ymin:surr_ymax, surr_xmin:surr_xmax]
        comp_patch_mask = component_mask[surr_ymin:surr_ymax, surr_xmin:surr_xmax] > 0
        lesion_mean_val = float(np.mean(surr_patch[comp_patch_mask])) if np.sum(comp_patch_mask) > 0 else 0.0
        bg_mean_val = float(np.mean(surr_patch[~comp_patch_mask])) if np.sum(~comp_patch_mask) > 0 else 0.0
        local_contrast = float(lesion_mean_val - bg_mean_val)

        # Composite proposal score: mean probability + size weight + contrast weight
        norm_area = min(1.0, area / 2000.0)
        norm_contrast = max(0.0, min(1.0, local_contrast + 0.5))
        composite_score = float(0.55 * mean_prob + 0.25 * peak_prob + 0.10 * norm_area + 0.10 * norm_contrast)

        return {
            'score': composite_score,
            'mean_prob': mean_prob,
            'peak_prob': peak_prob,
            'std_prob': std_prob,
            'area': area,
            'compactness': compactness,
            'local_contrast': local_contrast,
            'bbox': bbox
        }

    def generate_proposals(
        self,
        image: np.ndarray,
        prob_map: np.ndarray,
        k: int = 3
    ) -> Tuple[List[Dict[str, Any]], int, bool]:
        """
        Generates exactly K proposal bounding boxes and quality dictionaries.
        Returns: (proposals_list, num_detected_components, fallback_flag)
        """
        H, W = prob_map.shape
        bin_mask = (prob_map >= self.threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

        candidate_list = []
        raw_cc_count = 0
        for label_idx in range(1, num_labels):
            comp_area = stats[label_idx, cv2.CC_STAT_AREA]
            if comp_area < self.min_area:
                continue
            raw_cc_count += 1

            comp_mask = (labels == label_idx).astype(np.uint8)
            xmin = int(stats[label_idx, cv2.CC_STAT_LEFT])
            ymin = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            xmax = min(W, xmin + w)
            ymax = min(H, ymin + h)

            metrics = self.score_component(comp_mask, prob_map, image, (ymin, xmin, ymax, xmax))
            metrics['proposal_source'] = 'machine_cc'
            metrics['center'] = (float(centroids[label_idx][1]), float(centroids[label_idx][0]))
            candidate_list.append(metrics)

        # Sort candidates deterministically by composite score in descending order
        candidate_list.sort(key=lambda x: x['score'], reverse=True)

        # If candidates >= k, return top-K
        if len(candidate_list) >= k:
            return candidate_list[:k], raw_cc_count, False

        # Fallback needed
        fallback_flag = True
        final_proposals = list(candidate_list)
        needed = k - len(final_proposals)

        # Strategy 1: Sub-threshold high probability peaks
        if needed > 0 and len(final_proposals) == 0:
            flat_idx = int(np.argmax(prob_map))
            peak_y, peak_x = divmod(flat_idx, W)
            if prob_map[peak_y, peak_x] > 0.15:
                half_s = self.fallback_box_size // 2
                ymin = max(0, peak_y - half_s)
                ymax = min(H, peak_y + half_s)
                xmin = max(0, peak_x - half_s)
                xmax = min(W, peak_x + half_s)
                final_proposals.append({
                    'score': float(prob_map[peak_y, peak_x]),
                    'mean_prob': float(np.mean(prob_map[ymin:ymax, xmin:xmax])),
                    'peak_prob': float(prob_map[peak_y, peak_x]),
                    'std_prob': 0.0,
                    'area': int((ymax - ymin) * (xmax - xmin)),
                    'compactness': 1.0,
                    'local_contrast': 0.0,
                    'bbox': (ymin, xmin, ymax, xmax),
                    'proposal_source': 'prob_peak_fallback',
                    'center': (float(peak_y), float(peak_x))
                })
                needed -= 1

        # Strategy 2: High-density breast tissue centroids
        tissue_mask = image > 0.05
        if np.sum(tissue_mask) > 100:
            tissue_ys, tissue_xs = np.where(tissue_mask)
            cy, cx = int(np.median(tissue_ys)), int(np.median(tissue_xs))
        else:
            cy, cx = H // 2, W // 2

        # Quadrant offsets for diverse contextual fallbacks
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
        while len(final_proposals) < k and off_idx < len(offsets):
            dy, dx = offsets[off_idx]
            off_idx += 1
            f_cy = max(self.fallback_box_size // 2, min(H - self.fallback_box_size // 2, cy + dy))
            f_cx = max(self.fallback_box_size // 2, min(W - self.fallback_box_size // 2, cx + dx))
            half_s = self.fallback_box_size // 2
            ymin = max(0, f_cy - half_s)
            ymax = min(H, f_cy + half_s)
            xmin = max(0, f_cx - half_s)
            xmax = min(W, f_cx + half_s)

            final_proposals.append({
                'score': 0.0,
                'mean_prob': float(np.mean(prob_map[ymin:ymax, xmin:xmax])),
                'peak_prob': float(np.max(prob_map[ymin:ymax, xmin:xmax])),
                'std_prob': 0.0,
                'area': int((ymax - ymin) * (xmax - xmin)),
                'compactness': 1.0,
                'local_contrast': 0.0,
                'bbox': (ymin, xmin, ymax, xmax),
                'proposal_source': f'tissue_context_fallback_{off_idx}',
                'center': (float(f_cy), float(f_cx))
            })

        return final_proposals[:k], raw_cc_count, fallback_flag
