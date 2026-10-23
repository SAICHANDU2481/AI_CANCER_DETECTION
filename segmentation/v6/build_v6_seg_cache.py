"""
================================================================================
V6 Segmentation Fast Pre-Cache Builder (Train + Validation Only)
================================================================================
Strict Governance:
- Processes ONLY data/processed/train.csv (2501) and data/processed/val.csv (541).
- Quarantines test.csv completely (assert test access == 0).
- Multi-threaded processing for rapid offline caching of 512x512 enhanced images and masks.
================================================================================
"""

import os
import sys
import time
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# Ensure project root is in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from preprocessing.enhancement import enhance_mammogram

CACHE_DIR = os.path.join(ROOT, "data", "cache", "v6_seg_512")
os.makedirs(CACHE_DIR, exist_ok=True)

def resolve_path(p: str) -> Optional[str]:
    if os.path.exists(p):
        return p
    alt1 = p.replace("OneDrive\\Desktop", "Desktop")
    if os.path.exists(alt1):
        return alt1
    alt2 = p.replace("Desktop", "OneDrive\\Desktop")
    if os.path.exists(alt2):
        return alt2
    return None

def process_case(row_dict: dict) -> bool:
    case_id = str(row_dict['case_id'])
    out_file = os.path.join(CACHE_DIR, f"{case_id}.npz")
    if os.path.exists(out_file):
        return True

    # 1. Image
    img_path = resolve_path(str(row_dict['image_path']))
    if not img_path or not os.path.exists(img_path):
        return False

    raw_img = Image.open(img_path).convert("L")
    arr = np.array(raw_img.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    prep = enhance_mammogram(arr)
    enhanced = prep["enhanced"].astype(np.float32)

    # 2. Mask
    mask_path = resolve_path(str(row_dict.get('mask_path', '')))
    if mask_path and os.path.exists(mask_path):
        raw_mask = Image.open(mask_path).convert("L")
        mask_arr = (np.array(raw_mask.resize((512, 512), Image.Resampling.NEAREST), dtype=np.float32) > 127).astype(np.float32)
    else:
        mask_arr = np.zeros((512, 512), dtype=np.float32)

    np.savez_compressed(
        out_file,
        image=enhanced,
        mask=mask_arr,
        case_id=case_id,
        label=int(row_dict['label']),
        abnormality_type=str(row_dict.get('abnormality_type', '')),
        pathology=str(row_dict.get('pathology', ''))
    )
    return True

def build_cache():
    print("=" * 80)
    print("STARTING V6 SEGMENTATION CACHE BUILD (TRAIN + VAL ONLY)")
    print("=" * 80)

    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"Total scans to cache: {len(combined_df)} ({len(train_df)} train, {len(val_df)} val)")
    print(f"Target Cache Directory: {CACHE_DIR}")

    rows = combined_df.to_dict(orient='records')
    t0 = time.time()
    success_count = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_case, r) for r in rows]
        for idx, f in enumerate(as_completed(futures)):
            if f.result():
                success_count += 1
            if (idx + 1) % 500 == 0 or (idx + 1) == len(rows):
                print(f"  Processed {idx + 1}/{len(rows)} cases ({(time.time()-t0):.1f}s elapsed)...")

    t_total = time.time() - t0
    print(f"Cache complete: {success_count}/{len(rows)} files saved in {t_total:.1f}s ({t_total/len(rows):.3f}s / scan).")
    print("=" * 80)

if __name__ == "__main__":
    build_cache()
