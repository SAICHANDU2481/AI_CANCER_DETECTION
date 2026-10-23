import os
import sys
import json
import time
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve,
    roc_curve, auc, confusion_matrix, precision_score, recall_score,
    classification_report
)

PROJECT_ROOT = r'C:\Users\sys\Desktop\AI_CANCER_DETECTION'
ONEDRIVE_ROOT = r'C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION'
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from segmentation.ukan_v2b import MultiScaleUKAN
from classification.v4.roi_generator_v4 import MultiScaleROIGeneratorV4
from classification.v4.model_v4 import MultiScaleViTV4
from classification.v4.roi_quality import QUALITY_FEATURE_NAMES

def get_md5(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def compute_bootstrap_ci(y_true, y_pred, metric_fn, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        scores.append(metric_fn(y_true[indices], y_pred[indices]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

def compute_bootstrap_auc_ci(y_true, y_prob, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        scores.append(roc_auc_score(y_true[indices], y_prob[indices]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

def run_v4_test_evaluation():
    print("==================================================")
    print("V4-A FINAL LOCKED TEST EVALUATION — ONE-TIME EXECUTION")
    print("==================================================")
    
    t_start = time.time()
    
    # 1. Verification of Checksums & Integrity
    print("\n--- 1. Pre-Evaluation Checksum & Model Verification ---")
    chk_ukan = 'models/v2b/segmentation/ukan_v2b_best.pth'
    chk_v4 = 'models/v4/vit_v4_best.pth'
    test_csv_path = 'data/processed/test.csv'
    
    md5_ukan = get_md5(chk_ukan)
    md5_v4 = get_md5(chk_v4)
    print(f"Frozen V2-B U-KAN Checkpoint: {chk_ukan} -> MD5: {md5_ukan}")
    print(f"Locked V4-A Best Checkpoint:  {chk_v4} -> MD5: {md5_v4}")
    
    assert md5_ukan == 'a9a18b540cfd23114313b3e6d4276595', f"Mismatch in U-KAN MD5: {md5_ukan}"
    assert md5_v4 == 'd016fdc756758658e03d43f02fd4482c', f"Mismatch in V4-A MD5: {md5_v4}"
    
    test_df_raw = pd.read_csv(test_csv_path)
    # Filter to canonical 525-case benchmark cohort (excluding duplicate annotation)
    test_df = test_df_raw[test_df_raw['case_id'] != 'CALCIFICATION_P_01563_RIGHT_MLO_ab2'].reset_index(drop=True)
    
    print(f"\nLocked Test Cohort: {len(test_df)} valid scans, {test_df['patient_id'].nunique()} unique patients")
    print(f"Benign: {(test_df['label'] == 0).sum()}, Malignant: {(test_df['label'] == 1).sum()}")
    
    assert len(test_df) == 525, f"Expected 525 scans, got {len(test_df)}"
    assert test_df['patient_id'].nunique() == 235, f"Expected 235 patients, got {test_df['patient_id'].nunique()}"
    assert (test_df['label'] == 0).sum() == 310, f"Expected 310 benign, got {(test_df['label'] == 0).sum()}"
    assert (test_df['label'] == 1).sum() == 215, f"Expected 215 malignant, got {(test_df['label'] == 1).sum()}"
    
    # Patient isolation check against train and val
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    train_pts = set(train_df['patient_id'])
    val_pts = set(val_df['patient_id'])
    test_pts = set(test_df['patient_id'])
    
    assert len(test_pts.intersection(train_pts)) == 0, "Patient leakage detected between Test and Train!"
    assert len(test_pts.intersection(val_pts)) == 0, "Patient leakage detected between Test and Val!"
    print("Patient-Level Isolation: 100% VERIFIED (0 overlap with Train/Val)")
    
    # 2. Setup Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    generator = MultiScaleROIGeneratorV4(checkpoint_path=chk_ukan, device=device)
    
    v4_model = MultiScaleViTV4(
        scales_used=('local', 'medium', 'large', 'context'),
        use_quality_vector=False,
        enable_gating=False,
        pretrained=False
    ).to(device)
    
    chk_v4_data = torch.load(chk_v4, map_location=device)
    v4_model.load_state_dict(chk_v4_data['model_state_dict'])
    v4_model.eval()
    
    # 3. Deterministic Inference on 525 Test Scans
    print("\n--- 2. Deterministic Test Inference Pipeline ---")
    predictions_records = []
    
    test_crops_dir = 'data/cache/v4/test'
    os.makedirs(test_crops_dir, exist_ok=True)
    os.makedirs(os.path.join(ONEDRIVE_ROOT, test_crops_dir), exist_ok=True)
    
    norm_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    
    def prep_tensor(crop_2d):
        t = torch.from_numpy(crop_2d).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
        return (t - norm_mean) / norm_std
        
    with torch.inference_mode():
        for idx, row in test_df.iterrows():
            case_id = row['case_id']
            patient_id = row['patient_id']
            ab_type = row.get('abnormality_type', '')
            pathology = row.get('pathology', '')
            y_true = int(row['label'])
            
            # Load preprocessed 512x512 image & mask from segmentation cache
            cache_path = os.path.join('data/cache/segmentation_512', f"{case_id}.npz")
            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"Missing segmentation cache file: {cache_path}")
            data = np.load(cache_path)
            image_512 = data['image'].astype(np.float32)
            gt_mask = data['mask'].astype(np.float32)
            
            # Segmentation Inference via Frozen U-KAN
            inp = torch.from_numpy(image_512).float().unsqueeze(0).unsqueeze(0).to(device)
            with autocast('cuda', dtype=torch.float16):
                logits_seg = generator.segmentor(inp)
                prob_map = torch.sigmoid(logits_seg).squeeze().cpu().numpy().astype(np.float32)
            del inp, logits_seg
            
            # Multi-Scale ROI Extraction on CPU/numpy
            roi_res = generator.generate_rois(image_512, prob_map=prob_map)
            
            # Ground-truth evaluation metrics for post-hoc analysis ONLY (never fed to model)
            pred_binary = (prob_map >= 0.50).astype(np.uint8)
            gt_bin = (gt_mask > 0.50).astype(np.uint8)
            inter = np.sum((pred_binary > 0) & (gt_bin > 0))
            sum_masks = np.sum(pred_binary > 0) + np.sum(gt_bin > 0)
            union = np.sum((pred_binary > 0) | (gt_bin > 0))
            seg_dice = float(2.0 * inter / sum_masks) if sum_masks > 0 else (1.0 if np.sum(gt_bin > 0) == 0 and np.sum(pred_binary > 0) == 0 else 0.0)
            seg_iou = float(inter / union) if union > 0 else (1.0 if np.sum(gt_bin > 0) == 0 and np.sum(pred_binary > 0) == 0 else 0.0)
                
            # Classifier Forward Pass
            loc_t = prep_tensor(roi_res['local_crop'])
            med_t = prep_tensor(roi_res['medium_crop'])
            lrg_t = prep_tensor(roi_res['large_crop'])
            ctx_t = prep_tensor(roi_res['context_crop'])
            
            with autocast('cuda'):
                logits_clf = v4_model(loc_t, med_t, lrg_t, ctx_t)
                probs = torch.softmax(logits_clf, dim=-1)
                p_mal = float(probs[0, 1].item())
                
            y_pred = 1 if p_mal >= 0.50 else 0
            
            # Save test crop to cache for complete artifact traceability
            npz_path = os.path.join(test_crops_dir, f"{case_id}.npz")
            np.savez_compressed(
                npz_path,
                local=roi_res['local_crop'].astype(np.float32),
                medium=roi_res['medium_crop'].astype(np.float32),
                large=roi_res['large_crop'].astype(np.float32),
                context=roi_res['context_crop'].astype(np.float32),
                quality_vector=roi_res['quality_vector'].astype(np.float32),
                roi_quality_score=np.float32(roi_res['roi_quality_score'])
            )
            
            feat_dict = roi_res['feature_dict']
            record = {
                'patient_id': patient_id,
                'case_id': case_id,
                'abnormality_type': ab_type,
                'pathology': pathology,
                'true_label': y_true,
                'predicted_label': y_pred,
                'p_malignant': p_mal,
                'roi_source': roi_res['roi_source'],
                'fallback_used': roi_res['fallback_used'],
                'dice_if_available': seg_dice,
                'iou_if_available': seg_iou,
                'roi_area': float(feat_dict['bbox_area']),
                'quality_score_if_available': float(roi_res['roi_quality_score'])
            }
            for qf in QUALITY_FEATURE_NAMES:
                record[qf] = float(feat_dict[qf])
            predictions_records.append(record)
            
            del loc_t, med_t, lrg_t, ctx_t, logits_clf, probs
            if (idx + 1) % 20 == 0:
                torch.cuda.empty_cache()
                
            if (idx + 1) % 50 == 0 or (idx + 1) == len(test_df):
                print(f"  Processed [{idx+1}/{len(test_df)}] test scans ({time.time()-t_start:.1f}s)")
            
    pred_df = pd.DataFrame(predictions_records)
    
    # Save predictions CSV
    os.makedirs('results/v4/test', exist_ok=True)
    os.makedirs(os.path.join(ONEDRIVE_ROOT, 'results/v4/test'), exist_ok=True)
    
    pred_csv_path = 'results/v4/test/v4_test_predictions.csv'
    pred_df.to_csv(pred_csv_path, index=False)
    pred_df.to_csv(os.path.join(ONEDRIVE_ROOT, pred_csv_path), index=False)
    
    # 4. Compute Metrics
    print("\n--- 3. Primary Metrics & Statistical Calculations ---")
    y_true_all = pred_df['true_label'].values
    y_pred_all = pred_df['predicted_label'].values
    y_prob_all = pred_df['p_malignant'].values
    
    acc = accuracy_score(y_true_all, y_pred_all)
    macro_f1 = f1_score(y_true_all, y_pred_all, average='macro')
    binary_f1 = f1_score(y_true_all, y_pred_all, average='binary')
    roc_auc = roc_auc_score(y_true_all, y_prob_all)
    
    p_curve, r_curve, _ = precision_recall_curve(y_true_all, y_prob_all)
    pr_auc = auc(r_curve, p_curve)
    
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    mal_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    mal_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    
    benign_prec = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    benign_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    benign_f1 = float(2 * benign_prec * benign_recall / (benign_prec + benign_recall)) if (benign_prec + benign_recall) > 0 else 0.0
    mal_f1 = float(2 * mal_prec * mal_recall / (mal_prec + mal_recall)) if (mal_prec + mal_recall) > 0 else 0.0
    
    # 95% Bootstrap CIs
    acc_ci_low, acc_ci_high = compute_bootstrap_ci(y_true_all, y_pred_all, accuracy_score)
    f1_ci_low, f1_ci_high = compute_bootstrap_ci(y_true_all, y_pred_all, lambda yt, yp: f1_score(yt, yp, average='macro'))
    auc_ci_low, auc_ci_high = compute_bootstrap_auc_ci(y_true_all, y_prob_all)
    
    print(f"Accuracy:         {acc*100:.2f}% (95% CI: [{acc_ci_low*100:.2f}%, {acc_ci_high*100:.2f}%])")
    print(f"Macro-F1:         {macro_f1*100:.2f}% (95% CI: [{f1_ci_low*100:.2f}%, {f1_ci_high*100:.2f}%])")
    print(f"Binary F1:        {binary_f1*100:.2f}%")
    print(f"ROC-AUC:          {roc_auc:.4f} (95% CI: [{auc_ci_low:.4f}, {auc_ci_high:.4f}])")
    print(f"PR-AUC:           {pr_auc:.4f}")
    print(f"Malignant Recall: {mal_recall*100:.2f}% ({tp}/{tp+fn})")
    print(f"Specificity:      {spec*100:.2f}% ({tn}/{tn+fp})")
    print(f"Precision:        {mal_prec*100:.2f}% ({tp}/{tp+fp})")
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    # 5. Stratified Subgroup Analysis
    print("\n--- 4. Stratified Subgroup Analysis ---")
    subgroups = {}
    
    # A. Abnormality Type (Mass vs Calcification)
    for ab in ['MASS', 'CALCIFICATION']:
        sub_df = pred_df[pred_df['abnormality_type'].str.upper() == ab]
        if len(sub_df) > 0:
            yt = sub_df['true_label'].values
            yp = sub_df['predicted_label'].values
            ypr = sub_df['p_malignant'].values
            cm_sub = confusion_matrix(yt, yp, labels=[0, 1])
            tn_s, fp_s, fn_s, tp_s = cm_sub.ravel() if cm_sub.size == 4 else (0,0,0,0)
            try:
                sub_auc = float(roc_auc_score(yt, ypr))
            except Exception:
                sub_auc = 0.5
            subgroups[f"abnormality_{ab.lower()}"] = {
                'count': len(sub_df),
                'benign_count': int((yt == 0).sum()),
                'malignant_count': int((yt == 1).sum()),
                'accuracy_pct': float(accuracy_score(yt, yp) * 100),
                'macro_f1_pct': float(f1_score(yt, yp, average='macro', zero_division=0) * 100),
                'roc_auc': sub_auc,
                'malignant_recall_pct': float(tp_s / (tp_s + fn_s) * 100) if (tp_s + fn_s) > 0 else 0.0,
                'specificity_pct': float(tn_s / (tn_s + fp_s) * 100) if (tn_s + fp_s) > 0 else 0.0,
                'precision_pct': float(tp_s / (tp_s + fp_s) * 100) if (tp_s + fp_s) > 0 else 0.0,
                'tp': int(tp_s), 'tn': int(tn_s), 'fp': int(fp_s), 'fn': int(fn_s)
            }
            print(f"  {ab}: N={len(sub_df)}, Acc={subgroups[f'abnormality_{ab.lower()}']['accuracy_pct']:.2f}%, F1={subgroups[f'abnormality_{ab.lower()}']['macro_f1_pct']:.2f}%, MalRecall={subgroups[f'abnormality_{ab.lower()}']['malignant_recall_pct']:.2f}%, Spec={subgroups[f'abnormality_{ab.lower()}']['specificity_pct']:.2f}%")
            
    # B. ROI Source
    for src in ['machine_roi', 'expanded_machine_roi', 'context_roi', 'fallback_context']:
        sub_df = pred_df[pred_df['roi_source'] == src]
        if len(sub_df) > 0:
            yt = sub_df['true_label'].values
            yp = sub_df['predicted_label'].values
            ypr = sub_df['p_malignant'].values
            cm_sub = confusion_matrix(yt, yp, labels=[0, 1])
            tn_s, fp_s, fn_s, tp_s = cm_sub.ravel() if cm_sub.size == 4 else (0,0,0,0)
            try:
                sub_auc = float(roc_auc_score(yt, ypr))
            except Exception:
                sub_auc = 0.5
            subgroups[f"roi_source_{src}"] = {
                'count': len(sub_df),
                'rate_pct': float(len(sub_df) / len(pred_df) * 100),
                'benign_count': int((yt == 0).sum()),
                'malignant_count': int((yt == 1).sum()),
                'accuracy_pct': float(accuracy_score(yt, yp) * 100),
                'macro_f1_pct': float(f1_score(yt, yp, average='macro', zero_division=0) * 100),
                'roc_auc': sub_auc,
                'malignant_recall_pct': float(tp_s / (tp_s + fn_s) * 100) if (tp_s + fn_s) > 0 else 0.0,
                'specificity_pct': float(tn_s / (tn_s + fp_s) * 100) if (tn_s + fp_s) > 0 else 0.0,
                'tp': int(tp_s), 'tn': int(tn_s), 'fp': int(fp_s), 'fn': int(fn_s)
            }
            print(f"  ROI Source {src}: N={len(sub_df)} ({len(sub_df)/len(pred_df)*100:.2f}%), Acc={subgroups[f'roi_source_{src}']['accuracy_pct']:.2f}%, F1={subgroups[f'roi_source_{src}']['macro_f1_pct']:.2f}%")
            
    # C. Segmentation Quality
    pred_df['dice_stratum'] = pd.cut(
        pred_df['dice_if_available'],
        bins=[-0.01, 0.10, 0.50, 1.01],
        labels=['dice_lt_0.10', 'dice_0.10_to_0.50', 'dice_ge_0.50']
    )
    for stratum in ['dice_lt_0.10', 'dice_0.10_to_0.50', 'dice_ge_0.50']:
        sub_df = pred_df[pred_df['dice_stratum'] == stratum]
        if len(sub_df) > 0:
            yt = sub_df['true_label'].values
            yp = sub_df['predicted_label'].values
            cm_sub = confusion_matrix(yt, yp, labels=[0, 1])
            tn_s, fp_s, fn_s, tp_s = cm_sub.ravel() if cm_sub.size == 4 else (0,0,0,0)
            subgroups[f"seg_{stratum}"] = {
                'count': len(sub_df),
                'rate_pct': float(len(sub_df) / len(pred_df) * 100),
                'accuracy_pct': float(accuracy_score(yt, yp) * 100),
                'macro_f1_pct': float(f1_score(yt, yp, average='macro', zero_division=0) * 100),
                'malignant_recall_pct': float(tp_s / (tp_s + fn_s) * 100) if (tp_s + fn_s) > 0 else 0.0,
                'specificity_pct': float(tn_s / (tn_s + fp_s) * 100) if (tn_s + fp_s) > 0 else 0.0,
                'tp': int(tp_s), 'tn': int(tn_s), 'fp': int(fp_s), 'fn': int(fn_s)
            }
            print(f"  Segmentation Stratum {stratum}: N={len(sub_df)}, Acc={subgroups[f'seg_{stratum}']['accuracy_pct']:.2f}%, F1={subgroups[f'seg_{stratum}']['macro_f1_pct']:.2f}%")
            
    # Save Subgroups JSON
    subgroup_json_path = 'results/v4/test/v4_test_subgroup_metrics.json'
    with open(subgroup_json_path, 'w', encoding='utf-8') as f:
        json.dump(subgroups, f, indent=2)
    with open(os.path.join(ONEDRIVE_ROOT, subgroup_json_path), 'w', encoding='utf-8') as f:
        json.dump(subgroups, f, indent=2)
        
    # 6. Comprehensive Error Taxonomy Analysis
    print("\n--- 5. Test Error Taxonomy & Malignant False Negative Breakdown ---")
    pred_df['is_error'] = pred_df['true_label'] != pred_df['predicted_label']
    
    def classify_test_error(row):
        if not row['is_error']:
            return 'Correct'
        dice = row['dice_if_available']
        src = row['roi_source']
        fb = row['fallback_used']
        if dice < 0.10:
            return 'Severe segmentation failure (Dice < 0.10)'
        elif 0.10 <= dice < 0.50:
            return 'Partial lesion localization (0.10 <= Dice < 0.50)'
        elif fb or src in ['context_roi', 'fallback_context']:
            return 'Fallback / context case'
        elif dice >= 0.50:
            return 'Good localization but classifier failure'
        else:
            return 'Other'
            
    pred_df['error_category'] = pred_df.apply(classify_test_error, axis=1)
    
    # Save error analysis CSV
    err_csv_path = 'results/v4/test/v4_test_error_analysis.csv'
    pred_df.to_csv(err_csv_path, index=False)
    pred_df.to_csv(os.path.join(ONEDRIVE_ROOT, err_csv_path), index=False)
    
    errors_df = pred_df[pred_df['is_error']].copy()
    fn_df = pred_df[(pred_df['true_label'] == 1) & (pred_df['predicted_label'] == 0)].copy()
    
    fn_severe = int((fn_df['dice_if_available'] < 0.10).sum())
    fn_partial = int(((fn_df['dice_if_available'] >= 0.10) & (fn_df['dice_if_available'] < 0.50)).sum())
    fn_fallback = int(((fn_df['fallback_used'] == True) & (fn_df['dice_if_available'] >= 0.10)).sum())
    fn_good_clf = int(((fn_df['dice_if_available'] >= 0.50) & (fn_df['fallback_used'] == False)).sum())
    loc_assoc_fn = fn_severe + fn_partial + fn_fallback
    
    print(f"Total Errors: {len(errors_df)} / 525 ({len(errors_df)/525*100:.2f}%)")
    for cat, cnt in errors_df['error_category'].value_counts().items():
        print(f"  {cat}: {cnt} ({cnt/len(errors_df)*100:.2f}%)")
        
    print(f"\nMalignant False Negatives: {len(fn_df)} / 215 ({len(fn_df)/215*100:.2f}%)")
    print(f"  Severe Localization FN (Dice < 0.10):        {fn_severe} ({fn_severe/len(fn_df)*100:.2f}%)")
    print(f"  Partial Localization FN (0.10 <= Dice < 0.50): {fn_partial} ({fn_partial/len(fn_df)*100:.2f}%)")
    print(f"  Fallback / Context FN:                       {fn_fallback} ({fn_fallback/len(fn_df)*100:.2f}%)")
    print(f"  Good Localization Classifier FN (Dice >= 0.5):{fn_good_clf} ({fn_good_clf/len(fn_df)*100:.2f}%)")
    print(f"  Total Localization/Fallback-Associated FN:    {loc_assoc_fn}/{len(fn_df)} ({loc_assoc_fn/len(fn_df)*100:.2f}%)")
    print(f"  [Comparison with Locked V3-B Benchmark: V3-B FN = 108, Loc-Assoc = 80/108 (74.07%)]")
    
    # 7. Generate Evaluation Figures
    print("\n--- 6. Generating Evaluation Plots ---")
    
    # A. Confusion Matrix Plot
    plt.figure(figsize=(6, 5), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('V4-A Final Locked Test Confusion Matrix\n(Multi-Scale Visual Fusion)', fontsize=12, pad=12)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Benign (0)', 'Malignant (1)'], fontsize=10)
    plt.yticks(tick_marks, ['Benign (0)', 'Malignant (1)'], fontsize=10)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14, weight='bold')
    plt.ylabel('True Pathology Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    cm_plot_path = 'results/v4/test/v4_test_confusion_matrix.png'
    plt.savefig(cm_plot_path)
    plt.savefig(os.path.join(ONEDRIVE_ROOT, cm_plot_path))
    plt.close()
    
    # B. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'V4-A Multi-Scale ViT (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--', label='Chance (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Malignant Recall)', fontsize=11)
    plt.title('V4-A Final Test ROC Curve', fontsize=12, pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_plot_path = 'results/v4/test/v4_test_roc_curve.png'
    plt.savefig(roc_plot_path)
    plt.savefig(os.path.join(ONEDRIVE_ROOT, roc_plot_path))
    plt.close()
    
    # C. PR Curve Plot
    plt.figure(figsize=(6, 5), dpi=300)
    plt.plot(r_curve, p_curve, color='#2ca02c', lw=2.5, label=f'V4-A Multi-Scale ViT (PR-AUC = {pr_auc:.4f})')
    baseline_pr = float(np.sum(y_true_all == 1) / len(y_true_all))
    plt.axhline(y=baseline_pr, color='grey', linestyle='--', lw=1.5, label=f'Prevalence (P = {baseline_pr:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=11)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=11)
    plt.title('V4-A Final Test Precision-Recall Curve', fontsize=12, pad=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_plot_path = 'results/v4/test/v4_test_pr_curve.png'
    plt.savefig(pr_plot_path)
    plt.savefig(os.path.join(ONEDRIVE_ROOT, pr_plot_path))
    plt.close()
    
    # 8. Save Metrics JSON
    metrics_summary = {
        'model_name': 'V4-A Multi-Scale Visual Fusion ViT',
        'checkpoint_path': chk_v4,
        'checkpoint_md5': md5_v4,
        'ukan_checkpoint_path': chk_ukan,
        'ukan_checkpoint_md5': md5_ukan,
        'test_scans': len(test_df),
        'test_patients': int(test_df['patient_id'].nunique()),
        'primary_metrics': {
            'accuracy': float(acc),
            'accuracy_pct': float(acc * 100),
            'accuracy_95ci': [acc_ci_low, acc_ci_high],
            'macro_f1': float(macro_f1),
            'macro_f1_pct': float(macro_f1 * 100),
            'macro_f1_95ci': [f1_ci_low, f1_ci_high],
            'binary_f1': float(binary_f1),
            'binary_f1_pct': float(binary_f1 * 100),
            'roc_auc': float(roc_auc),
            'roc_auc_95ci': [auc_ci_low, auc_ci_high],
            'pr_auc': float(pr_auc),
            'malignant_recall': float(mal_recall),
            'malignant_recall_pct': float(mal_recall * 100),
            'specificity': float(spec),
            'specificity_pct': float(spec * 100),
            'precision': float(mal_prec),
            'precision_pct': float(mal_prec * 100)
        },
        'per_class_metrics': {
            'benign': {
                'precision': benign_prec,
                'recall': benign_recall,
                'f1_score': benign_f1,
                'support': int((y_true_all == 0).sum())
            },
            'malignant': {
                'precision': mal_prec,
                'recall': mal_recall,
                'f1_score': mal_f1,
                'support': int((y_true_all == 1).sum())
            }
        },
        'confusion_matrix': {
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
        },
        'error_taxonomy': {
            'total_errors': len(errors_df),
            'error_rate_pct': float(len(errors_df) / len(test_df) * 100),
            'categories': errors_df['error_category'].value_counts().to_dict(),
            'malignant_fn_breakdown': {
                'total_malignant_fn': len(fn_df),
                'severe_localization_fn': fn_severe,
                'partial_localization_fn': fn_partial,
                'fallback_context_fn': fn_fallback,
                'good_localization_classifier_fn': fn_good_clf,
                'localization_associated_fn_count': loc_assoc_fn,
                'localization_associated_fn_pct': float(loc_assoc_fn / len(fn_df) * 100) if len(fn_df) > 0 else 0.0,
                'v3b_locked_reference_fn_count': 108,
                'v3b_locked_reference_loc_associated_count': 80,
                'v3b_locked_reference_loc_associated_pct': 74.07
            }
        },
        'execution_time_seconds': float(time.time() - t_start)
    }
    
    metrics_json_path = 'results/v4/test/v4_test_metrics.json'
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2)
    with open(os.path.join(ONEDRIVE_ROOT, metrics_json_path), 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2)
        
    # 9. Create Final Test Manifest
    manifest = {
        'manifest_version': '1.0.0',
        'experiment_name': 'V4-A Final Locked Test Evaluation',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'evaluation_status': 'COMPLETED_AND_LOCKED',
        'model_architecture': 'MultiScaleViTV4',
        'scales_used': ['local', 'medium', 'large', 'context'],
        'quality_vector_enabled': False,
        'dynamic_gating_enabled': False,
        'checkpoint_path': chk_v4,
        'checkpoint_md5': md5_v4,
        'segmentation_model': {
            'name': 'MultiScaleUKAN (V2-B)',
            'checkpoint_path': chk_ukan,
            'md5_checksum': md5_ukan
        },
        'test_cohort': {
            'csv_path': test_csv_path,
            'total_scans': len(test_df),
            'unique_patients': int(test_df['patient_id'].nunique()),
            'benign_scans': int((test_df['label'] == 0).sum()),
            'malignant_scans': int((test_df['label'] == 1).sum()),
            'patient_isolation_status': 'VERIFIED_ZERO_LEAKAGE'
        },
        'final_test_metrics': {
            'accuracy_pct': float(acc * 100),
            'accuracy_95ci': [acc_ci_low, acc_ci_high],
            'macro_f1_pct': float(macro_f1 * 100),
            'macro_f1_95ci': [f1_ci_low, f1_ci_high],
            'binary_f1_pct': float(binary_f1 * 100),
            'roc_auc': float(roc_auc),
            'roc_auc_95ci': [auc_ci_low, auc_ci_high],
            'pr_auc': float(pr_auc),
            'malignant_recall_pct': float(mal_recall * 100),
            'specificity_pct': float(spec * 100),
            'precision_pct': float(mal_prec * 100),
            'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
        },
        'output_files_checksums': {
            'v4_test_predictions.csv': get_md5(pred_csv_path),
            'v4_test_metrics.json': get_md5(metrics_json_path),
            'v4_test_error_analysis.csv': get_md5(err_csv_path),
            'v4_test_subgroup_metrics.json': get_md5(subgroup_json_path),
            'v4_test_confusion_matrix.png': get_md5(cm_plot_path),
            'v4_test_roc_curve.png': get_md5(roc_plot_path),
            'v4_test_pr_curve.png': get_md5(pr_plot_path)
        }
    }
    
    test_manifest_path = 'results/v4/V4_FINAL_TEST_MANIFEST.json'
    with open(test_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(ONEDRIVE_ROOT, test_manifest_path), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nFinal Test Manifest generated at: {test_manifest_path}")
    print("V4-A Final Locked Test Evaluation Complete!")
    return metrics_summary, subgroups, pred_df

if __name__ == '__main__':
    run_v4_test_evaluation()
