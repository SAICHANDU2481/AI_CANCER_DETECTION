"""
================================================================================
V7 STAGE 0 SUITE: High-Performance Diagnostic Suite (3A - 3F)
================================================================================
"""

import os
import sys
import time
import json
import logging
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from v7_stage0_base import (
    DEVICE, RESULTS_DIR, DATA_INTEGRITY_DIR,
    FROZEN_VIT_PATH, TRAIN_CACHE_DIR, VAL_CACHE_DIR,
    TRAIN_CSV, VAL_CSV, resolve_image_path,
    load_frozen_v6_classifier, patient_clustered_bootstrap_metric, compute_metrics_from_arrays
)
from classification.v6.model_v6 import MultiScaleViTV6
from classification.v6.dataset_v6 import V6CachedMultiScaleDataset
from classification.v6.roi_generator_v6 import MultiScaleROIGeneratorV6

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("V7_Stage0_Suite")

NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

def to_tensor_eval(crop_2d: np.ndarray) -> torch.Tensor:
    t = np.repeat(crop_2d[np.newaxis, :, :], 3, axis=0)
    t = (t - NORM_MEAN) / NORM_STD
    return torch.from_numpy(t).float()


# ==============================================================================
# DIAGNOSTIC 3A: TRAIN VS VALIDATION ROI DISTRIBUTION
# ==============================================================================
def run_diagnostic_3a(train_df: pd.DataFrame, val_df: pd.DataFrame):
    logger.info("=== Starting Diagnostic 3A: Train vs Validation ROI Distribution ===")
    meta_csv = os.path.join(ROOT, "results", "v6", "classification", "v6_roi_cache_metadata.csv")
    if os.path.exists(meta_csv):
        meta_df = pd.read_csv(meta_csv)
    else:
        raise FileNotFoundError(f"Metadata CSV not found: {meta_csv}")
        
    train_meta = meta_df[meta_df['split'] == 'train'].copy()
    val_meta = meta_df[meta_df['split'] == 'val'].copy()
    
    tr_total = len(train_meta)
    tr_fall = int(train_meta['fallback_flag'].sum())
    tr_mach = tr_total - tr_fall
    tr_mach_rate = tr_mach / tr_total if tr_total > 0 else 0.0
    tr_fall_rate = tr_fall / tr_total if tr_total > 0 else 0.0
    tr_mean_dice = float(train_meta['gt_dice'].mean())
    tr_mean_iou = float(train_meta['gt_iou'].mean())
    
    val_total = len(val_meta)
    val_fall = int(val_meta['fallback_flag'].sum())
    val_mach = val_total - val_fall
    val_mach_rate = val_mach / val_total if val_total > 0 else 0.0
    val_fall_rate = val_fall / val_total if val_total > 0 else 0.0
    val_mean_dice = float(val_meta['gt_dice'].mean())
    val_mean_iou = float(val_meta['gt_iou'].mean())
    
    ks_dice = stats.ks_2samp(train_meta['gt_dice'], val_meta['gt_dice'])
    ks_iou = stats.ks_2samp(train_meta['gt_iou'], val_meta['gt_iou'])
    
    contingency = [[tr_mach, tr_fall], [val_mach, val_fall]]
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency)
    
    res = {
        'train': {
            'total_records': tr_total,
            'machine_roi_count': tr_mach,
            'machine_roi_rate': tr_mach_rate,
            'fallback_count': tr_fall,
            'fallback_rate': tr_fall_rate,
            'mean_dice': tr_mean_dice,
            'mean_iou': tr_mean_iou,
            'source_breakdown': train_meta['top1_source'].fillna('fallback').value_counts().to_dict()
        },
        'validation': {
            'total_records': val_total,
            'machine_roi_count': val_mach,
            'machine_roi_rate': val_mach_rate,
            'fallback_count': val_fall,
            'fallback_rate': val_fall_rate,
            'mean_dice': val_mean_dice,
            'mean_iou': val_mean_iou,
            'source_breakdown': val_meta['top1_source'].fillna('fallback').value_counts().to_dict()
        },
        'comparison': {
            'chi2_stat': float(chi2_stat),
            'chi2_p_value': float(chi2_p),
            'ks_dice_stat': float(ks_dice.statistic),
            'ks_dice_p_value': float(ks_dice.pvalue),
            'ks_iou_stat': float(ks_iou.statistic),
            'ks_iou_p_value': float(ks_iou.pvalue),
            'mismatch_detected': bool(chi2_p < 0.05 or ks_dice.pvalue < 0.05)
        }
    }
    
    out_path = os.path.join(RESULTS_DIR, "stage0_train_val_roi_distribution.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"Diagnostic 3A complete. Output saved to {out_path}")
    return res, train_meta, val_meta


# ==============================================================================
# SINGLE-PASS VALIDATION EXTRACTION FOR 3B & 3C
# ==============================================================================
def extract_validation_features_and_preds(val_df: pd.DataFrame, model: MultiScaleViTV6):
    logger.info("Extracting validation embeddings and baseline predictions in a single pass (AMP FP16)...")
    val_ds = V6CachedMultiScaleDataset(
        val_df,
        cache_dir=VAL_CACHE_DIR,
        is_training=False,
        k=3,
        use_full_image=True,
        use_calc_enhanced=True
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    
    all_case_ids = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    list_full_feats = []
    list_calc_feats = []
    dict_scale_feats = {s: [] for s in model.scales_used}
    
    with torch.no_grad():
        for batch in val_loader:
            case_ids = batch['case_id']
            y = batch['label'].to(DEVICE)
            full_img = batch['full_image'].to(DEVICE)
            calc_patch = batch['calc_enhanced'].to(DEVICE)
            cand_dict = {
                'local': batch['local'].to(DEVICE),
                'medium': batch['medium'].to(DEVICE),
                'large': batch['large'].to(DEVICE),
                'context': batch['context'].to(DEVICE)
            }
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                # Extract individual feature representations
                batch_scale_feats = []
                for s in model.scales_used:
                    cand_input = cand_dict[s]
                    f = model.aggregate_candidates(cand_input, scale_name=s)
                    batch_scale_feats.append(f)
                    dict_scale_feats[s].append(f.cpu())
                    
                full_feat = model.encode_tokens(full_img)
                calc_feat = model.encode_tokens(calc_patch)
                list_full_feats.append(full_feat.cpu())
                list_calc_feats.append(calc_feat.cpu())
                
                # Baseline forward
                parts = [full_feat, torch.cat(batch_scale_feats, dim=-1), calc_feat]
                joint_feat = torch.cat(parts, dim=-1)
                normed = model.fusion_norm(joint_feat)
                logits = model.fusion_mlp(normed)
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
            all_case_ids.extend(case_ids)
            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())
            all_y_prob.extend(probs[:, 1].cpu().numpy())
            
    feature_cache = {
        'case_ids': all_case_ids,
        'y_true': np.array(all_y_true),
        'y_pred': np.array(all_y_pred),
        'y_prob': np.array(all_y_prob),
        'full_feat': torch.cat(list_full_feats, dim=0),
        'calc_feat': torch.cat(list_calc_feats, dim=0),
        'scale_feats': {s: torch.cat(dict_scale_feats[s], dim=0) for s in model.scales_used}
    }
    logger.info("Validation feature extraction complete.")
    return feature_cache


# ==============================================================================
# DIAGNOSTIC 3B: ERROR STRATIFICATION
# ==============================================================================
def run_diagnostic_3b(val_df: pd.DataFrame, feature_cache: dict, val_meta: pd.DataFrame):
    logger.info("=== Starting Diagnostic 3B: Validation Error Stratification ===")
    
    val_preds_df = val_df.copy()
    val_preds_df['pred'] = feature_cache['y_pred']
    val_preds_df['prob'] = feature_cache['y_prob']
    val_preds_df['error'] = (val_preds_df['pred'] != val_preds_df['label']).astype(int)
    
    meta_sub = val_meta[['case_id', 'fallback_flag', 'gt_dice', 'gt_iou', 'top1_score']].copy()
    meta_sub.rename(columns={'fallback_flag': 'fallback', 'gt_dice': 'dice', 'gt_iou': 'iou'}, inplace=True)
    val_preds_df = val_preds_df.merge(meta_sub, on='case_id', how='left')
    val_preds_df['roi_status'] = val_preds_df['fallback'].apply(lambda x: 'Fallback' if x else 'Machine ROI')
    
    overall_metrics = compute_metrics_from_arrays(val_preds_df['label'].values, val_preds_df['pred'].values, val_preds_df['prob'].values)
    overall_bootstrap = patient_clustered_bootstrap_metric(val_preds_df, pred_col='pred', prob_col='prob', target_col='label')
    
    val_preds_df['dice_bin'] = pd.cut(
        val_preds_df['dice'].fillna(0.0),
        bins=[-0.01, 0.10, 0.30, 0.50, 1.01],
        labels=['Dice < 0.10', 'Dice 0.10-0.30', 'Dice 0.30-0.50', 'Dice >= 0.50']
    )
    
    sizes = []
    for idx, row in val_preds_df.iterrows():
        mp = resolve_image_path(str(row.get('mask_path', '')))
        if os.path.exists(mp):
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            sizes.append(int(np.sum(m > 127)))
        else:
            sizes.append(0)
    val_preds_df['lesion_area'] = sizes
    val_preds_df['size_bin'] = pd.cut(
        val_preds_df['lesion_area'],
        bins=[-1, 2000, 10000, 1e8],
        labels=['Small (<2k)', 'Medium (2k-10k)', 'Large (>10k)']
    )
    
    strat_rows = []
    def stratify_group(sub_df: pd.DataFrame, strat_name: str, strat_val: str):
        if len(sub_df) == 0:
            return
        m = compute_metrics_from_arrays(sub_df['label'].values, sub_df['pred'].values, sub_df['prob'].values)
        boot = patient_clustered_bootstrap_metric(sub_df, pred_col='pred', prob_col='prob', target_col='label') if len(sub_df['label'].unique()) > 1 else {}
        
        row = {
            'stratum_type': strat_name,
            'stratum_value': strat_val,
            'n_records': len(sub_df),
            'n_patients': int(sub_df['patient_id'].nunique()),
            'accuracy': m['accuracy'],
            'macro_f1': m['macro_f1'],
            'binary_f1': m['binary_f1'],
            'recall': m['recall'],
            'specificity': m['specificity'],
            'roc_auc': m['roc_auc'],
            'pr_auc': m['pr_auc'],
            'tn': m['tn'], 'fp': m['fp'], 'fn': m['fn'], 'tp': m['tp'],
            'acc_ci_lower': boot.get('acc', {}).get('ci_lower', m['accuracy']),
            'acc_ci_upper': boot.get('acc', {}).get('ci_upper', m['accuracy']),
            'macro_f1_ci_lower': boot.get('macro_f1', {}).get('ci_lower', m['macro_f1']),
            'macro_f1_ci_upper': boot.get('macro_f1', {}).get('ci_upper', m['macro_f1']),
            'rec_ci_lower': boot.get('rec', {}).get('ci_lower', m['recall']),
            'rec_ci_upper': boot.get('rec', {}).get('ci_upper', m['recall']),
            'spec_ci_lower': boot.get('spec', {}).get('ci_lower', m['specificity']),
            'spec_ci_upper': boot.get('spec', {}).get('ci_upper', m['specificity']),
        }
        strat_rows.append(row)

    stratify_group(val_preds_df, 'Overall', 'All Validation')
    for ab in val_preds_df['abnormality_type'].dropna().unique():
        stratify_group(val_preds_df[val_preds_df['abnormality_type'] == ab], 'Abnormality', str(ab))
    if 'image_view' in val_preds_df.columns:
        for v in val_preds_df['image_view'].dropna().unique():
            stratify_group(val_preds_df[val_preds_df['image_view'] == v], 'View', str(v))
    for s in ['Machine ROI', 'Fallback']:
        stratify_group(val_preds_df[val_preds_df['roi_status'] == s], 'ROI_Status', s)
    for db in ['Dice < 0.10', 'Dice 0.10-0.30', 'Dice 0.30-0.50', 'Dice >= 0.50']:
        stratify_group(val_preds_df[val_preds_df['dice_bin'] == db], 'Dice_Quality', str(db))
    for sb in ['Small (<2k)', 'Medium (2k-10k)', 'Large (>10k)']:
        stratify_group(val_preds_df[val_preds_df['size_bin'] == sb], 'Lesion_Size', str(sb))

    strat_df = pd.DataFrame(strat_rows)
    out_csv = os.path.join(RESULTS_DIR, "stage0_val_error_stratification.csv")
    strat_df.to_csv(out_csv, index=False)
    
    out_json = os.path.join(RESULTS_DIR, "stage0_val_overall_metrics.json")
    with open(out_json, "w") as f:
        json.dump({'deterministic': overall_metrics, 'bootstrap_ci': overall_bootstrap}, f, indent=2)
        
    logger.info(f"Diagnostic 3B complete. Saved to {out_csv} and {out_json}")
    return val_preds_df, strat_df, overall_metrics, overall_bootstrap


# ==============================================================================
# DIAGNOSTIC 3C: FAST CLASSIFIER PATHWAY ABLATIONS
# ==============================================================================
def run_diagnostic_3c(feature_cache: dict, model: MultiScaleViTV6):
    logger.info("=== Starting Diagnostic 3C: Frozen ViT Pathway Ablations ===")
    
    y_true = feature_cache['y_true']
    N = len(y_true)
    full_feat = feature_cache['full_feat'].to(DEVICE)
    calc_feat = feature_cache['calc_feat'].to(DEVICE)
    roi_fused = torch.cat([feature_cache['scale_feats'][s].to(DEVICE) for s in model.scales_used], dim=-1)
    
    results = {}
    
    modes = {
        'full_pipeline': (full_feat, roi_fused, calc_feat),
        'roi_only': (torch.zeros_like(full_feat), roi_fused, torch.zeros_like(calc_feat)),
        'full_mammogram_only': (full_feat, torch.zeros_like(roi_fused), torch.zeros_like(calc_feat)),
        'calc_patch_only': (torch.zeros_like(full_feat), torch.zeros_like(roi_fused), calc_feat),
        'random_crop_control': (full_feat, torch.randn_like(roi_fused) * 0.1, calc_feat)
    }
    
    with torch.no_grad():
        for m_name, (f_full, f_roi, f_calc) in modes.items():
            joint_feat = torch.cat([f_full, f_roi, f_calc], dim=-1)
            normed = model.fusion_norm(joint_feat)
            logits = model.fusion_mlp(normed)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            y_p = preds.cpu().numpy()
            y_pr = probs[:, 1].cpu().numpy()
            results[m_name] = compute_metrics_from_arrays(y_true, y_p, y_pr)
            
    out_path = os.path.join(RESULTS_DIR, "stage0_pathway_ablations.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Diagnostic 3C complete. Output saved to {out_path}")
    return results


# ==============================================================================
# DIAGNOSTIC 3D: ORACLE ROI DIAGNOSTIC (CEILING BENCHMARK)
# ==============================================================================
class OracleValidationDataset(Dataset):
    def __init__(self, val_df: pd.DataFrame, roi_gen: MultiScaleROIGeneratorV6):
        self.val_df = val_df.reset_index(drop=True)
        self.roi_gen = roi_gen
        
    def __len__(self):
        return len(self.val_df)
        
    def __getitem__(self, idx):
        row = self.val_df.iloc[idx]
        case_id = str(row['case_id'])
        img_path = resolve_image_path(str(row['image_path']))
        mask_path = resolve_image_path(str(row.get('mask_path', '')))
        label = int(row['label'])
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is not None:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            img_gray = np.zeros((512, 512), dtype=np.float32)
            
        mask_bgr = cv2.imread(mask_path)
        if mask_bgr is not None:
            mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY) > 127
        else:
            mask_gray = np.zeros_like(img_gray, dtype=bool)
            
        H, W = img_gray.shape
        if np.sum(mask_gray) > 0:
            ys, xs = np.where(mask_gray)
            ymin, ymax = int(np.min(ys)), int(np.max(ys))
            xmin, xmax = int(np.min(xs)), int(np.max(xs))
            cy, cx = float(np.mean(ys)), float(np.mean(xs))
        else:
            ymin, ymax = H // 4, 3 * H // 4
            xmin, xmax = W // 4, 3 * W // 4
            cy, cx = float(H / 2), float(W / 2)
            
        oracle_proposal = {
            'bbox': (ymin, xmin, ymax, xmax),
            'center': (cy, cx),
            'score': 1.0,
            'proposal_source': 'oracle_ground_truth'
        }
        
        oracle_crops, meta = self.roi_gen.generate_candidate_crops(img_gray, oracle_proposal)
        full_canvas = cv2.resize(img_gray, (224, 224), interpolation=cv2.INTER_LINEAR)
        full_tensor = to_tensor_eval(full_canvas)
        
        calc_crop, _ = self.roi_gen.enhance_calcification_patch(oracle_crops['local'])
        calc_tensor = to_tensor_eval(calc_crop)
        
        scale_dict = {}
        for s in ('local', 'medium', 'large', 'context'):
            t = to_tensor_eval(oracle_crops[s])
            scale_dict[s] = torch.stack([t, t, t], dim=0) # (3, 3, 224, 224)
            
        return {
            'case_id': case_id,
            'patient_id': str(row['patient_id']),
            'label': label,
            'full_image': full_tensor,
            'calc_enhanced': calc_tensor,
            'local': scale_dict['local'],
            'medium': scale_dict['medium'],
            'large': scale_dict['large'],
            'context': scale_dict['context']
        }


def run_diagnostic_3d(val_df: pd.DataFrame, model: MultiScaleViTV6):
    logger.info("=== Starting Diagnostic 3D: Oracle ROI Diagnostic (GT Ceiling) ===")
    roi_gen = MultiScaleROIGeneratorV6(
        local_scale=1.2,
        medium_scale=1.8,
        large_scale=2.5,
        context_scale=4.0,
        target_crop_size=224,
        full_image_size=224,
        default_k=3
    )
    
    oracle_ds = OracleValidationDataset(val_df, roi_gen)
    oracle_loader = DataLoader(oracle_ds, batch_size=16, shuffle=False, num_workers=0)
    
    oracle_records = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    with torch.no_grad():
        for batch in oracle_loader:
            case_ids = batch['case_id']
            patient_ids = batch['patient_id']
            y = batch['label'].to(DEVICE)
            full_img = batch['full_image'].to(DEVICE)
            calc_patch = batch['calc_enhanced'].to(DEVICE)
            cand_dict = {
                'local': batch['local'].to(DEVICE),
                'medium': batch['medium'].to(DEVICE),
                'large': batch['large'].to(DEVICE),
                'context': batch['context'].to(DEVICE)
            }
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(cand_dict, full_image_tensor=full_img, calc_patch_tensor=calc_patch)
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())
            all_y_prob.extend(probs[:, 1].cpu().numpy())
            
            for cid, pid, yt, yp, ypr in zip(case_ids, patient_ids, y.cpu().numpy(), preds.cpu().numpy(), probs[:, 1].cpu().numpy()):
                oracle_records.append({
                    'case_id': cid,
                    'patient_id': pid,
                    'label': int(yt),
                    'oracle_pred': int(yp),
                    'oracle_prob': float(ypr),
                    'oracle_error': int(yp != yt)
                })
                
    oracle_df = pd.DataFrame(oracle_records)
    oracle_metrics = compute_metrics_from_arrays(np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob))
    oracle_bootstrap = patient_clustered_bootstrap_metric(oracle_df, pred_col='oracle_pred', prob_col='oracle_prob', target_col='label')
    
    out_json = os.path.join(RESULTS_DIR, "stage0_oracle_roi_metrics.json")
    with open(out_json, "w") as f:
        json.dump({'deterministic': oracle_metrics, 'bootstrap_ci': oracle_bootstrap}, f, indent=2)
    logger.info(f"Diagnostic 3D complete. Output saved to {out_json}")
    return oracle_metrics, oracle_bootstrap, oracle_df


# ==============================================================================
# DIAGNOSTIC 3E: DATA INTEGRITY AUDIT
# ==============================================================================
def run_diagnostic_3e(train_df: pd.DataFrame, val_df: pd.DataFrame):
    logger.info("=== Starting Diagnostic 3E: Data Integrity Audit ===")
    
    def audit_split(df: pd.DataFrame, split_name: str):
        shape_mismatches = 0
        empty_masks = 0
        missing_images = 0
        missing_masks = 0
        total_records = len(df)
        
        base_scan_labels = {}
        for idx, row in df.iterrows():
            img_p = resolve_image_path(str(row['image_path']))
            mask_p = resolve_image_path(str(row.get('mask_path', '')))
            label = int(row['label'])
            
            if not os.path.exists(img_p):
                missing_images += 1
                continue
            if not os.path.exists(mask_p):
                missing_masks += 1
                continue
                
            img_size = Image.open(img_p).size
            mask_size = Image.open(mask_p).size
            if img_size != mask_size:
                shape_mismatches += 1
                
            mask_arr = np.array(Image.open(mask_p).convert('L'))
            if np.sum(mask_arr > 127) == 0:
                empty_masks += 1
                
            base_key = f"{row['patient_id']}_{row.get('left_or_right_breast','')}_{row.get('image_view','')}"
            if base_key not in base_scan_labels:
                base_scan_labels[base_key] = []
            base_scan_labels[base_key].append(label)
            
        multi_ab_count = sum(1 for k, v in base_scan_labels.items() if len(v) > 1)
        conflict_count = sum(1 for k, v in base_scan_labels.items() if len(set(v)) > 1)
        
        return {
            'total_records': total_records,
            'missing_images': missing_images,
            'missing_masks': missing_masks,
            'shape_mismatches': shape_mismatches,
            'empty_masks': empty_masks,
            'multi_abnormality_scans': multi_ab_count,
            'conflicting_label_scans': conflict_count
        }

    train_audit = audit_split(train_df, 'train')
    val_audit = audit_split(val_df, 'validation')
    
    logger.info("Generating 50 validation visual overlay samples...")
    sample_indices = random.Random(42).sample(range(len(val_df)), min(50, len(val_df)))
    
    for i, idx in enumerate(sample_indices):
        row = val_df.iloc[idx]
        case_id = str(row['case_id'])
        img_p = resolve_image_path(str(row['image_path']))
        mask_p = resolve_image_path(str(row.get('mask_path', '')))
        
        if not os.path.exists(img_p) or not os.path.exists(mask_p):
            continue
            
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title(f"Image: {case_id}\nLabel: {row['pathology']}")
        ax[0].axis('off')
        
        ax[1].imshow(img, cmap='gray')
        if np.sum(mask > 127) > 0:
            contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                poly = patches.Polygon(cnt.squeeze(1), fill=False, edgecolor='red', linewidth=2)
                ax[1].add_patch(poly)
        ax[1].set_title(f"GT Mask Overlay\nType: {row.get('abnormality_type', '')}")
        ax[1].axis('off')
        
        fig.tight_layout()
        save_p = os.path.join(DATA_INTEGRITY_DIR, f"overlay_val_{i:02d}_{case_id}.png")
        fig.savefig(save_p, dpi=100)
        plt.close(fig)
        
    audit_summary = {
        'train_split': train_audit,
        'validation_split': val_audit,
        'overlays_generated': min(50, len(val_df)),
        'overlays_dir': DATA_INTEGRITY_DIR
    }
    
    out_json = os.path.join(RESULTS_DIR, "stage0_data_integrity_audit.json")
    with open(out_json, "w") as f:
        json.dump(audit_summary, f, indent=2)
    logger.info(f"Diagnostic 3E complete. Output saved to {out_json}")
    return audit_summary


# ==============================================================================
# DIAGNOSTIC 3F: TRAIN VS VALIDATION DISTRIBUTION ANALYSIS
# ==============================================================================
def run_diagnostic_3f(train_df: pd.DataFrame, val_df: pd.DataFrame):
    logger.info("=== Starting Diagnostic 3F: Train vs Validation Distribution Analysis ===")
    
    tr_ab = train_df['abnormality_type'].value_counts(normalize=True).to_dict()
    val_ab = val_df['abnormality_type'].value_counts(normalize=True).to_dict()
    
    tr_label = train_df['label'].value_counts(normalize=True).to_dict()
    val_label = val_df['label'].value_counts(normalize=True).to_dict()
    
    tr_view = train_df['image_view'].value_counts(normalize=True).to_dict() if 'image_view' in train_df.columns else {}
    val_view = val_df['image_view'].value_counts(normalize=True).to_dict() if 'image_view' in val_df.columns else {}
    
    def compute_gt_lesion_areas(df: pd.DataFrame):
        areas = []
        for idx, row in df.iterrows():
            mask_p = resolve_image_path(str(row.get('mask_path', '')))
            if os.path.exists(mask_p):
                mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
                areas.append(int(np.sum(mask > 127)))
            else:
                areas.append(0)
        return areas
        
    tr_areas = compute_gt_lesion_areas(train_df)
    val_areas = compute_gt_lesion_areas(val_df)
    
    ks_areas = stats.ks_2samp(tr_areas, val_areas)
    
    summary = {
        'abnormality_distribution': {
            'train': tr_ab,
            'validation': val_ab
        },
        'class_distribution': {
            'train': tr_label,
            'validation': val_label
        },
        'view_distribution': {
            'train': tr_view,
            'validation': val_view
        },
        'lesion_size_pixels': {
            'train_mean': float(np.mean(tr_areas)),
            'train_median': float(np.median(tr_areas)),
            'val_mean': float(np.mean(val_areas)),
            'val_median': float(np.median(val_areas)),
            'ks_stat': float(ks_areas.statistic),
            'ks_p_value': float(ks_areas.pvalue)
        }
    }
    
    out_json = os.path.join(RESULTS_DIR, "stage0_train_vs_val_comparison.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Diagnostic 3F complete. Output saved to {out_json}")
    return summary


def main():
    logger.info("Starting Full V7 Stage 0 Diagnostic Suite...")
    t0 = time.time()
    
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    
    logger.info(f"Loaded Train: {len(train_df)} rows ({train_df['patient_id'].nunique()} pts)")
    logger.info(f"Loaded Validation: {len(val_df)} rows ({val_df['patient_id'].nunique()} pts)")
    
    # 3A: ROI Distribution
    res_3a, tr_meta, val_meta = run_diagnostic_3a(train_df, val_df)
    
    # Load frozen classifier
    model = load_frozen_v6_classifier()
    
    # Single-pass feature extraction for 3B and 3C
    feature_cache = extract_validation_features_and_preds(val_df, model)
    
    # 3B: Validation Stratification
    val_preds_df, strat_df, overall_m, overall_boot = run_diagnostic_3b(val_df, feature_cache, val_meta)
    
    # 3C: Fast Pathway Ablations
    res_3c = run_diagnostic_3c(feature_cache, model)
    
    # 3D: Batched Oracle ROI Diagnostic
    oracle_m, oracle_boot, oracle_df = run_diagnostic_3d(val_df, model)
    
    # 3E: Data Integrity Audit
    res_3e = run_diagnostic_3e(train_df, val_df)
    
    # 3F: Distribution Divergence
    res_3f = run_diagnostic_3f(train_df, val_df)
    
    elapsed = time.time() - t0
    logger.info(f"=== All Stage 0 Diagnostics Completed Successfully in {elapsed:.1f}s ===")


if __name__ == '__main__':
    main()
