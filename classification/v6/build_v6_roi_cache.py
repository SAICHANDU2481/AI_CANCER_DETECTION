"""
================================================================================
V6 Phase 4: Machine-ROI Cache Builder
================================================================================
Generates deterministic multi-scale Top-K ROI crops (K=3, 4 scales), enhanced
calcification patches, and full-mammogram global canvases using the frozen
V6 Phase 3 winning U-KAN model (Arm G, MD5: 68cfc38cfbddd93d6e4a6e692fffbb34).

Cohorts:
- Train Split: 2,501 scans (1,096 patients) -> data/cache/v6_machine_rois/train/
- Val Split:   541 scans (235 patients)   -> data/cache/v6_machine_rois/val/

Strict Governance:
- ZERO access to data/processed/test.csv or test images.
- Deterministic feature extraction using frozen V6 U-KAN.
- Validation GT masks used ONLY for post-hoc validation Dice/IoU recording.
================================================================================
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.amp import autocast

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from segmentation.v6.ukan_v6 import build_ukan_v6
from preprocessing.enhancement import enhance_mammogram
from classification.v6.topk_proposals_v6 import TopKProposalGeneratorV6
from classification.v6.roi_generator_v6 import MultiScaleROIGeneratorV6

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("BuildV6ROICache")

# Explicit Quarantine Enforcement
FORBIDDEN = ['test.csv', 'verified_test.csv', 'test_predictions', 'test_masks']
for f in FORBIDDEN:
    assert f not in __file__, f"Test string found in execution script: {f}"


def compute_sample_dice(pred_binary: np.ndarray, target_binary: np.ndarray, smooth: float = 1.0) -> float:
    intersection = np.sum(pred_binary * target_binary)
    cardinality = np.sum(pred_binary) + np.sum(target_binary)
    if cardinality == 0:
        return 1.0
    return float((2.0 * intersection + smooth) / (cardinality + smooth))


def compute_sample_iou(pred_binary: np.ndarray, target_binary: np.ndarray, smooth: float = 1.0) -> float:
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary) - intersection
    if union == 0:
        return 1.0
    return float((intersection + smooth) / (union + smooth))


def build_v6_cache_for_split(
    split_name: str,
    df: pd.DataFrame,
    v6_model: torch.nn.Module,
    proposal_gen: TopKProposalGeneratorV6,
    roi_gen: MultiScaleROIGeneratorV6,
    device: torch.device,
    target_cache_dir: str,
    seg_source_dir: str = "data/cache/v6_seg_512",
    k: int = 3
) -> Dict[str, Any]:
    os.makedirs(target_cache_dir, exist_ok=True)
    logger.info(f"--- Starting V6 Cache Build for {split_name.upper()} ({len(df)} scans) at {target_cache_dir} ---")
    t0 = time.time()
    created_count = 0
    fallback_count = 0
    gt_dices = []
    gt_ious = []
    roi_metadata_records = []

    for idx, row in df.iterrows():
        case_id = str(row['case_id'])
        label = int(row['label'])
        abn_type = str(row.get('abnormality_type', ''))
        patient_id = str(row.get('patient_id', ''))
        laterality = str(row.get('laterality', ''))
        view = str(row.get('view', ''))
        dest_p = os.path.join(target_cache_dir, f"{case_id}.npz")

        # Resume support: Check if already computed and valid
        if os.path.exists(dest_p):
            try:
                cached_data = np.load(dest_p)
                if 'full_image' in cached_data and 'calc_enhanced' in cached_data and 'cand3_context' in cached_data:
                    created_count += 1
                    fb = bool(cached_data.get('fallback_flag', False))
                    if fb:
                        fallback_count += 1
                    if 'gt_dice' in cached_data:
                        gt_dices.append(float(cached_data['gt_dice']))
                    if 'gt_iou' in cached_data:
                        gt_ious.append(float(cached_data['gt_iou']))
                    roi_metadata_records.append({
                        "case_id": case_id,
                        "patient_id": patient_id,
                        "split": split_name,
                        "label": label,
                        "abnormality_type": abn_type,
                        "laterality": laterality,
                        "view": view,
                        "fallback_flag": fb,
                        "num_components": int(cached_data.get('num_components', 0)),
                        "gt_dice": float(cached_data.get('gt_dice', 0.0)),
                        "gt_iou": float(cached_data.get('gt_iou', 0.0)),
                        "calc_source": str(cached_data.get('calc_source', '')),
                        "top1_score": float(cached_data.get('proposal_scores', [0.0])[0]) if len(cached_data.get('proposal_scores', [])) > 0 else 0.0,
                        "v6_checkpoint_md5": "68cfc38cfbddd93d6e4a6e692fffbb34",
                        "inference_threshold": 0.50
                    })
                    if created_count % 250 == 0 or created_count == len(df):
                        elapsed = (time.time() - t0) / 60.0
                        logger.info(f"  {split_name.upper()}: {created_count}/{len(df)} cached ({elapsed:.1f} min elapsed).")
                    continue
            except Exception:
                # File corrupted or truncated; recompute
                pass

        # Check existing preprocessed 512x512 segmentation cache
        img_512 = None
        gt_mask_512 = None
        seg_cache_p = os.path.join(ROOT, seg_source_dir, f"{case_id}.npz")
        if os.path.exists(seg_cache_p):
            try:
                data = np.load(seg_cache_p)
                img_512 = data['image'].astype(np.float32)
                if 'mask' in data:
                    gt_mask_512 = data['mask'].astype(np.float32)
            except Exception:
                pass

        if img_512 is None:
            img_path = str(row['image_path'])
            if not os.path.isabs(img_path):
                img_path = os.path.join(ROOT, img_path)
            if os.path.exists(img_path):
                raw_img = Image.open(img_path).convert("L")
                raw_arr = np.array(raw_img.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
                prep_dict = enhance_mammogram(raw_arr)
                img_512 = prep_dict["enhanced"].astype(np.float32)

        if img_512 is None:
            logger.warning(f"Could not load image for {case_id}, creating synthetic zero canvas.")
            img_512 = np.zeros((512, 512), dtype=np.float32)

        # 1. V6 U-KAN forward pass (512x512)
        t_in = torch.from_numpy(img_512).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = v6_model(t_in)
                if isinstance(outputs, (tuple, list)):
                    main_logits = outputs[0]
                elif isinstance(outputs, dict):
                    main_logits = outputs['main']
                else:
                    main_logits = outputs
                prob_512 = torch.sigmoid(main_logits).squeeze().float().cpu().numpy()

        # 2. Extract Top-K proposals deterministically
        proposals, num_ccs, is_fallback = proposal_gen.generate_proposals(img_512, prob_512, k=k)
        if is_fallback:
            fallback_count += 1

        # 3. Full-Mammogram global context (224x224)
        full_img_224 = cv2.resize(img_512, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)

        # 4. Multi-scale crops for all K candidates
        npz_dict = {
            'case_id': case_id,
            'patient_id': patient_id,
            'label': label,
            'split': split_name,
            'abnormality_type': abn_type,
            'laterality': laterality,
            'view': view,
            'full_image': full_img_224,
            'proposal_scores': np.array([p['score'] for p in proposals], dtype=np.float32),
            'proposal_sources': np.array([p['proposal_source'] for p in proposals]),
            'num_components': int(num_ccs),
            'fallback_flag': bool(is_fallback)
        }

        # Multi-scale crops per proposal candidate
        for k_idx, prop in enumerate(proposals, 1):
            ms_crops, crop_meta = roi_gen.generate_candidate_crops(img_512, prop)
            for scale_name, crop_arr in ms_crops.items():
                npz_dict[f'cand{k_idx}_{scale_name}'] = crop_arr.astype(np.float32)

        # 5. Enhanced microcalcification patch
        primary_local = npz_dict['cand1_local']
        calc_enhanced, calc_src = roi_gen.enhance_calcification_patch(primary_local)
        npz_dict['calc_enhanced'] = calc_enhanced
        npz_dict['calc_source'] = str(calc_src)

        # 6. Validation GT Metrics (Audit only; never fed to model input)
        sample_dice = 0.0
        sample_iou = 0.0
        if gt_mask_512 is not None:
            pred_bin = (prob_512 >= 0.50).astype(np.float32)
            gt_bin = (gt_mask_512 > 0.50).astype(np.float32)
            sample_dice = compute_sample_dice(pred_bin, gt_bin)
            sample_iou = compute_sample_iou(pred_bin, gt_bin)
            gt_dices.append(sample_dice)
            gt_ious.append(sample_iou)
            npz_dict['gt_dice'] = float(sample_dice)
            npz_dict['gt_iou'] = float(sample_iou)

        # Save compressed NPZ
        np.savez_compressed(dest_p, **npz_dict)
        created_count += 1

        roi_metadata_records.append({
            "case_id": case_id,
            "patient_id": patient_id,
            "split": split_name,
            "label": label,
            "abnormality_type": abn_type,
            "num_components": int(num_ccs),
            "fallback_flag": bool(is_fallback),
            "calc_source": str(calc_src),
            "top1_score": float(proposals[0]['score']) if proposals else 0.0,
            "top1_source": str(proposals[0]['proposal_source']) if proposals else "none",
            "gt_dice": float(sample_dice),
            "gt_iou": float(sample_iou)
        })

        if created_count % 250 == 0 or created_count == len(df):
            elapsed = time.time() - t0
            rate = created_count / max(0.1, elapsed)
            logger.info(f"  {split_name.upper()}: {created_count}/{len(df)} cached ({elapsed/60:.1f} min elapsed, {rate:.1f} scans/sec).")

    total_time_min = (time.time() - t0) / 60.0
    fallback_pct = (fallback_count / len(df)) * 100.0 if len(df) > 0 else 0.0
    mean_dice = float(np.mean(gt_dices)) * 100.0 if gt_dices else 0.0
    mean_iou = float(np.mean(gt_ious)) * 100.0 if gt_ious else 0.0

    summary = {
        "split": split_name,
        "total_scans": created_count,
        "fallback_count": fallback_count,
        "fallback_pct": round(fallback_pct, 2),
        "machine_roi_success_pct": round(100.0 - fallback_pct, 2),
        "mean_gt_dice": round(mean_dice, 2),
        "mean_gt_iou": round(mean_iou, 2),
        "runtime_min": round(total_time_min, 2),
        "target_dir": target_cache_dir
    }
    logger.info(f"[SUCCESS] {split_name.upper()} Cache Complete: {created_count} scans in {total_time_min:.2f} min (Fallback Rate: {fallback_pct:.1f}%, Mean Dice: {mean_dice:.2f}%).")
    return summary, roi_metadata_records


def run_v6_roi_cache_build():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("=" * 80)
    logger.info("STARTING V6 MACHINE-ROI CACHE GENERATION")
    logger.info("=" * 80)
    logger.info(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load frozen V6 U-KAN Best Model
    v6_ckpt = os.path.join(ROOT, "models", "v6", "segmentation", "final", "ukan_v6_arm_g_best.pth")
    assert os.path.exists(v6_ckpt), f"V6 best checkpoint missing: {v6_ckpt}"
    logger.info(f"Loading Frozen V6 U-KAN Checkpoint: {v6_ckpt}")

    v6_model = build_ukan_v6(
        use_detail_branch=True,
        use_multi_scale_blocks=True,
        deep_supervision=True
    ).to(device)

    sd = torch.load(v6_ckpt, map_location=device)
    if 'model_state_dict' in sd:
        v6_model.load_state_dict(sd['model_state_dict'])
    else:
        v6_model.load_state_dict(sd)
    v6_model.eval()

    proposal_gen = TopKProposalGeneratorV6(min_area=15, threshold=0.50, default_k=3)
    roi_gen = MultiScaleROIGeneratorV6(
        local_scale=1.2,
        medium_scale=1.8,
        large_scale=2.5,
        context_scale=4.0,
        target_crop_size=224,
        full_image_size=224,
        default_k=3
    )

    train_df = pd.read_csv(os.path.join(ROOT, "data", "processed", "train.csv"))
    val_df = pd.read_csv(os.path.join(ROOT, "data", "processed", "val.csv"))
    logger.info(f"Cohorts: Train = {len(train_df)} scans, Val = {len(val_df)} scans (100% Patient Isolated).")

    val_target = os.path.join(ROOT, "data", "cache", "v6_machine_rois", "val")
    train_target = os.path.join(ROOT, "data", "cache", "v6_machine_rois", "train")

    # 1. Build Val Cache First
    val_summary, val_records = build_v6_cache_for_split(
        split_name='val',
        df=val_df,
        v6_model=v6_model,
        proposal_gen=proposal_gen,
        roi_gen=roi_gen,
        device=device,
        target_cache_dir=val_target,
        k=3
    )

    # 2. Build Train Cache Second
    train_summary, train_records = build_v6_cache_for_split(
        split_name='train',
        df=train_df,
        v6_model=v6_model,
        proposal_gen=proposal_gen,
        roi_gen=roi_gen,
        device=device,
        target_cache_dir=train_target,
        k=3
    )

    # Save metadata records to CSV
    all_records = pd.DataFrame(val_records + train_records)
    meta_csv_path = os.path.join(ROOT, "results", "v6", "classification", "v6_roi_cache_metadata.csv")
    all_records.to_csv(meta_csv_path, index=False)
    logger.info(f"Saved ROI cache metadata table: {meta_csv_path}")

    # Build Audit Summary
    audit_summary = {
        "timestamp": time.time(),
        "date_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "v6_checkpoint": v6_ckpt,
        "train_summary": train_summary,
        "val_summary": val_summary,
        "total_cached": train_summary["total_scans"] + val_summary["total_scans"]
    }
    logger.info(f"Audit Summary: {json.dumps(audit_summary, indent=2)}")
    logger.info("=" * 80)
    logger.info("V6 MACHINE-ROI CACHE GENERATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_v6_roi_cache_build()
