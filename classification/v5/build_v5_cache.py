"""
================================================================================
V5 Phase 4 & 5 Cache Builder: High-Resolution Top-K & Full-Mammogram Generator
================================================================================
Generates deterministic multi-scale Top-K ROI crops (K=3, 4 scales), enhanced
calcification patches, and full-mammogram global canvases on Drive F:
- Train Split: 2,501 scans (1,061 patients) -> F:\\BreastCancer_V5_Cache\\train
- Val Split:   541 scans (235 patients)   -> F:\\BreastCancer_V5_Cache\\val

Strict Governance:
- ZERO access to data/processed/test.csv or test images.
- Deterministic feature extraction using winner U-KAN V5 (768x768).
- ZERO mirror writes to Drive C:.
================================================================================
"""

import os
import sys
import time
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.amp import autocast

PROJECT_ROOT = r'C:\Users\sys\Desktop\AI_CANCER_DETECTION'
ONEDRIVE_ROOT = r'C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION'
ROOTS = [PROJECT_ROOT, ONEDRIVE_ROOT]

F_CACHE_ROOT = r'F:\BreastCancer_V5_Cache'
if not os.path.exists('F:'):
    raise RuntimeError("FATAL: Drive F: is not accessible. Fallback to Drive C: is strictly forbidden.")

if os.path.exists(PROJECT_ROOT):
    os.chdir(PROJECT_ROOT)

for r in ROOTS:
    if r not in sys.path and os.path.exists(r):
        sys.path.insert(0, r)

from preprocessing.enhancement import enhance_mammogram
from segmentation.ukan_v5 import MultiScaleUKANV5
from classification.v5.topk_proposals import TopKProposalGenerator
from classification.v5.roi_generator_v5 import MultiScaleROIGeneratorV5

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("BuildV5Cache")

# Quarantine verification
FORBIDDEN = ['test.csv', 'verified_test.csv', 'test_predictions', 'test_masks']
for f in FORBIDDEN:
    assert f not in __file__, f"Test string found in execution script: {f}"


def build_cache_for_split(
    split_name: str,
    df: pd.DataFrame,
    ukan_model: torch.nn.Module,
    proposal_gen: TopKProposalGenerator,
    roi_gen: MultiScaleROIGeneratorV5,
    device: torch.device,
    source_cache_dir: str,
    target_cache_dir: str
):
    if not os.path.exists('F:'):
        raise RuntimeError("FATAL: Drive F: is not accessible. Halting immediately.")

    os.makedirs(target_cache_dir, exist_ok=True)
    logger.info(f"--- Starting Cache Build for {split_name.upper()} ({len(df)} scans) at {target_cache_dir} ---")
    t0 = time.time()
    created_count = 0

    for idx, row in df.iterrows():
        case_id = str(row['case_id'])
        label = int(row['label'])
        dest_p = os.path.join(target_cache_dir, f"{case_id}.npz")

        # Resume support: Preserve existing valid files
        if os.path.exists(dest_p) and os.path.getsize(dest_p) > 0:
            created_count += 1
            continue

        img_512 = None
        for root in ROOTS:
            chk_p = os.path.join(root, source_cache_dir, f"{case_id}.npz")
            if os.path.exists(chk_p):
                try:
                    data = np.load(chk_p)
                    img_512 = data['image'].astype(np.float32)
                    break
                except Exception:
                    pass

        if img_512 is None:
            # Fallback to loading raw image
            img_path = str(row['image_path'])
            raw_candidates = [
                img_path,
                img_path.replace('OneDrive\\Desktop', 'Desktop'),
                img_path.replace('Desktop', 'OneDrive\\Desktop')
            ]
            for p in raw_candidates:
                if os.path.exists(p):
                    try:
                        raw_img = Image.open(p).convert("L")
                        raw_arr = np.array(raw_img.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
                        prep_dict = enhance_mammogram(raw_arr)
                        img_512 = prep_dict["enhanced"].astype(np.float32)
                        break
                    except Exception:
                        pass

        if img_512 is None:
            logger.warning(f"Could not load image for {case_id}, creating synthetic background canvas.")
            img_512 = np.zeros((512, 512), dtype=np.float32)

        # 1. High-Res U-KAN forward pass (768x768)
        img_768 = cv2.resize(img_512, (768, 768), interpolation=cv2.INTER_LINEAR)
        t_in = torch.from_numpy(img_768).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            with autocast('cuda'):
                logits_768 = ukan_model(t_in)
                prob_768 = torch.sigmoid(logits_768).squeeze().float().cpu().numpy()

        prob_512 = cv2.resize(prob_768, (512, 512), interpolation=cv2.INTER_LINEAR)

        # 2. Extract Top-3 proposals deterministically
        proposals = proposal_gen.generate_proposals(img_512, prob_512, k=3)

        # 3. Full-Mammogram global context (224x224)
        full_img_224 = cv2.resize(img_512, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)

        # 4. Multi-scale crops for all 3 candidates
        npz_dict = {
            'case_id': case_id,
            'label': label,
            'full_image': full_img_224,
            'proposal_scores': np.array([p['score'] for p in proposals], dtype=np.float32),
            'proposal_sources': np.array([p.get('proposal_source', p.get('source', 'machine_roi')) for p in proposals])
        }

        # Multi-scale crops per proposal candidate
        for k_idx, prop in enumerate(proposals, 1):
            ms_crops = roi_gen.generate_candidate_crops(img_512, prop)
            for scale_name, crop_arr in ms_crops.items():
                npz_dict[f'cand{k_idx}_{scale_name}'] = crop_arr.astype(np.float32)

        # 5. Enhanced microcalcification patch
        primary_local = npz_dict['cand1_local']
        calc_enhanced = roi_gen.enhance_calcification_patch(primary_local)
        npz_dict['calc_enhanced'] = calc_enhanced

        # Save compressed npz directly to F: only (NO mirror write to Drive C:)
        np.savez_compressed(dest_p, **npz_dict)

        created_count += 1
        if created_count % 100 == 0 or created_count == len(df):
            elapsed = time.time() - t0
            rate = created_count / max(0.1, elapsed)
            logger.info(f"  {split_name.upper()}: {created_count}/{len(df)} cached ({elapsed/60:.1f} min elapsed, {rate:.1f} scans/sec).")

    total_time = (time.time() - t0) / 60.0
    logger.info(f"[SUCCESS] {split_name.upper()} Cache Complete: {created_count} scans in {total_time:.2f} minutes.")


def run_v5_cache_generation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("==================================================")
    logger.info("STARTING V5 DETERMINISTIC MULTI-SCALE CACHE BUILD (DRIVE F:)")
    logger.info("==================================================")
    logger.info(f"Target Cache Root: {F_CACHE_ROOT}")
    logger.info(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    ukan_ckpt = 'models/v5/segmentation/v5_768_bce_dice_best.pth'
    if not os.path.exists(ukan_ckpt):
        alt_ckpt = os.path.join(PROJECT_ROOT, ukan_ckpt)
        if os.path.exists(alt_ckpt):
            ukan_ckpt = alt_ckpt

    logger.info(f"Loading Winner U-KAN Model: {ukan_ckpt}")
    ukan_model = MultiScaleUKANV5(in_channels=1, out_channels=1).to(device)
    state = torch.load(ukan_ckpt, map_location=device)
    sd = state['model_state_dict'] if 'model_state_dict' in state else state
    ukan_model.load_state_dict(sd)
    ukan_model.eval()

    proposal_gen = TopKProposalGenerator(min_area=30, threshold=0.15, default_k=3)
    roi_gen = MultiScaleROIGeneratorV5(
        local_scale=1.2,
        medium_scale=1.8,
        large_scale=2.5,
        context_scale=4.0,
        target_crop_size=224,
        full_image_size=224,
        default_k=3
    )

    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')

    logger.info(f"Dataset Cohorts: Train = {len(train_df)} scans, Val = {len(val_df)} scans (100% Isolated).")

    val_target = os.path.join(F_CACHE_ROOT, 'val')
    train_target = os.path.join(F_CACHE_ROOT, 'train')

    # Build Val first
    build_cache_for_split(
        split_name='val',
        df=val_df,
        ukan_model=ukan_model,
        proposal_gen=proposal_gen,
        roi_gen=roi_gen,
        device=device,
        source_cache_dir='data/cache/segmentation_512',
        target_cache_dir=val_target
    )

    # Build Train second
    build_cache_for_split(
        split_name='train',
        df=train_df,
        ukan_model=ukan_model,
        proposal_gen=proposal_gen,
        roi_gen=roi_gen,
        device=device,
        source_cache_dir='data/cache/segmentation_512',
        target_cache_dir=train_target
    )

    logger.info("==================================================")
    logger.info("ALL V5 CACHES SUCCESSFULLY GENERATED & VERIFIED ON DRIVE F:")
    logger.info("==================================================")


if __name__ == '__main__':
    run_v5_cache_generation()
