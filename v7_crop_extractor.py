"""
================================================================================
V7 Phase 6: High-Performance Multi-Scale ROI Crop Extractor & Cache Builder
================================================================================
Reads Arm-F proposals from Phase 5 proposal cache (top-5 candidate bboxes),
extracts 4 multi-scale crops (1.2x, 1.8x, 2.5x, 4.0x) for each candidate,
extracts the full-image global canvas (224x224), and creates the calcification-
enhanced patch. Saves compact uint8 .npz archives for ultra-fast I/O.
================================================================================
"""

import os
import sys
import json
import logging
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
import pandas as pd

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from v7_stage0_base import resolve_image_path, VAL_CSV, TRAIN_CSV

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("V7_Crop_Extractor")

TRAIN_PROPOSALS_JSON = os.path.join(ROOT, "data", "cache", "v7_proposals", "train", "v7_train_proposals.json")
VAL_PROPOSALS_JSON = os.path.join(ROOT, "data", "cache", "v7_proposals", "val", "v7_val_proposals.json")

TRAIN_ROI_CACHE_DIR = os.path.join(ROOT, "data", "cache", "v7_roi_crops", "train")
VAL_ROI_CACHE_DIR = os.path.join(ROOT, "data", "cache", "v7_roi_crops", "val")

os.makedirs(TRAIN_ROI_CACHE_DIR, exist_ok=True)
os.makedirs(VAL_ROI_CACHE_DIR, exist_ok=True)


def extract_scaled_crop_224_u8(
    image_u8: np.ndarray,
    center: Tuple[float, float],
    base_size: float,
    scale: float,
    target_size: int = 224
) -> np.ndarray:
    """
    Extracts a square uint8 crop centered at (cy, cx) with size = base_size * scale.
    """
    H, W = image_u8.shape
    cy, cx = center
    crop_s = max(24.0, base_size * scale)
    half_s = crop_s / 2.0

    ymin = int(round(cy - half_s))
    ymax = int(round(cy + half_s))
    xmin = int(round(cx - half_s))
    xmax = int(round(cx + half_s))

    pad_top = max(0, -ymin)
    pad_bottom = max(0, ymax - H)
    pad_left = max(0, -xmin)
    pad_right = max(0, xmax - W)

    src_ymin = max(0, ymin)
    src_ymax = min(H, ymax)
    src_xmin = max(0, xmin)
    src_xmax = min(W, xmax)

    patch = image_u8[src_ymin:src_ymax, src_xmin:src_xmax]

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    if patch.shape[0] != target_size or patch.shape[1] != target_size:
        patch = cv2.resize(patch, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return patch.astype(np.uint8)


def enhance_calcification_patch_u8(patch_u8: np.ndarray) -> np.ndarray:
    """
    Applies localized adaptive unsharp masking and CLAHE for microcalcifications on uint8 patch.
    """
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(patch_u8)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 1.0)
        unsharp = np.clip(enhanced.astype(np.int16) + 0.5 * (enhanced.astype(np.int16) - blurred.astype(np.int16)), 0, 255).astype(np.uint8)
        return unsharp
    except Exception:
        return patch_u8


def process_dataset_crops(df: pd.DataFrame, proposals_cache: Dict[str, Any], output_dir: str, name: str):
    logger.info(f"Extracting multi-scale crops for {name} ({len(df)} cases)...")
    t0 = time.time()
    
    scales = {
        'local': 1.2,
        'medium': 1.8,
        'large': 2.5,
        'context': 4.0
    }
    
    count = 0
    for idx, row in df.iterrows():
        case_id = str(row['case_id'])
        out_path = os.path.join(output_dir, f"{case_id}.npz")
        
        # Check if already cached
        if os.path.exists(out_path):
            count += 1
            continue
            
        img_p = resolve_image_path(str(row['image_path']))
        if not os.path.exists(img_p):
            logger.warning(f"Image not found: {img_p}")
            continue
            
        img_u8 = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img_u8 is None:
            continue
            
        H, W = img_u8.shape
        
        # 1. Full image canvas (224x224)
        full_image_224 = cv2.resize(img_u8, (224, 224), interpolation=cv2.INTER_AREA)
        
        # 2. Get proposals for this scan
        case_prop_info = proposals_cache.get(case_id, {})
        if 'proposals' in case_prop_info:
            arm_f_bboxes = case_prop_info['proposals'].get('Arm F (V6 + High-Res + Calc)', [])
            if not arm_f_bboxes:
                arm_f_bboxes = case_prop_info['proposals'].get('Arm G (All + Context)', [])
            if not arm_f_bboxes:
                arm_f_bboxes = case_prop_info['proposals'].get('Source B (High-Res U-KAN 768)', [])
        else:
            arm_f_bboxes = case_prop_info.get('proposals_arm_f', [])
            if not arm_f_bboxes:
                arm_f_bboxes = case_prop_info.get('proposals_arm_g', [])
                
        # If no proposal found, create default center crop
        if not arm_f_bboxes:
            cy, cx = H // 2, W // 2
            bs = min(H, W) // 3
            arm_f_bboxes = [[cy - bs//2, cx - bs//2, cy + bs//2, cx + bs//2]]
            
        # Extract crops for up to K=5 candidates
        save_dict = {'full_image': full_image_224}
        
        cand1_local = None
        for k_idx in range(5):
            if k_idx < len(arm_f_bboxes):
                ymin, xmin, ymax, xmax = arm_f_bboxes[k_idx]
                bw = max(16, xmax - xmin)
                bh = max(16, ymax - ymin)
                base_size = float(max(bw, bh))
                cy = float(ymin + bh / 2.0)
                cx = float(xmin + bw / 2.0)
            else:
                # Pad with candidate 1 (or center crop)
                ymin, xmin, ymax, xmax = arm_f_bboxes[0]
                bw = max(16, xmax - xmin)
                bh = max(16, ymax - ymin)
                base_size = float(max(bw, bh))
                cy = float(ymin + bh / 2.0)
                cx = float(xmin + bw / 2.0)
                
            for s_name, s_val in scales.items():
                crop_patch = extract_scaled_crop_224_u8(img_u8, (cy, cx), base_size, s_val, target_size=224)
                save_dict[f'cand{k_idx+1}_{s_name}'] = crop_patch
                if k_idx == 0 and s_name == 'local':
                    cand1_local = crop_patch
                    
        # 3. Calcification-Enhanced Patch (from candidate 1 local)
        if cand1_local is not None:
            save_dict['calc_enhanced'] = enhance_calcification_patch_u8(cand1_local)
        else:
            save_dict['calc_enhanced'] = full_image_224
            
        np.savez_compressed(out_path, **save_dict)
        count += 1
        
        if count % 500 == 0 or count == len(df):
            logger.info(f"Cached {count}/{len(df)} {name} cases ({time.time() - t0:.1f}s)")
            
    logger.info(f"Completed {name} ROI crop extraction in {time.time() - t0:.1f}s.")


def main():
    logger.info("Loading Phase 5 Proposal Caches...")
    with open(VAL_PROPOSALS_JSON, "r") as f:
        val_proposals = json.load(f)
    with open(TRAIN_PROPOSALS_JSON, "r") as f:
        train_proposals = json.load(f)
        
    val_df = pd.read_csv(VAL_CSV)
    train_df = pd.read_csv(TRAIN_CSV)
    
    # Process Validation & Train
    process_dataset_crops(val_df, val_proposals, VAL_ROI_CACHE_DIR, "VALIDATION")
    process_dataset_crops(train_df, train_proposals, TRAIN_ROI_CACHE_DIR, "TRAIN")
    
    logger.info("All V7 Phase 6 multi-scale ROI crops successfully extracted and cached.")


if __name__ == '__main__':
    main()
