import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve,
    auc, confusion_matrix, precision_score, recall_score
)

PROJECT_ROOT = r'C:\Users\sys\Desktop\AI_CANCER_DETECTION'
ONEDRIVE_ROOT = r'C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION'
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from classification.v4.dataset_v4 import MultiScaleCancerDatasetV4
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

def run_detailed_analysis():
    print("==================================================")
    print("V4-A DETAILED VALIDATION ANALYSIS & ERROR TAXONOMY")
    print("==================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Validation Dataset and ROI Quality Data
    val_csv_path = 'data/processed/val.csv'
    val_df = pd.read_csv(val_csv_path)
    print(f"Validation Cohort: {len(val_df)} scans, {val_df['patient_id'].nunique()} patients")
    print(f"Validation Class Distribution: Benign = {(val_df['label']==0).sum()}, Malignant = {(val_df['label']==1).sum()}")
    
    val_q_path = 'data/cache/v4/roi_quality_val.csv'
    val_q_df = pd.read_csv(val_q_path)
    
    # Check ROI Statistics
    total_val = len(val_q_df)
    
    machine_roi_count = (val_q_df['roi_source'] == 'machine_roi').sum()
    expanded_roi_count = (val_q_df['roi_source'] == 'expanded_machine_roi').sum()
    context_roi_count = (val_q_df['roi_source'] == 'context_roi').sum()
    fallback_count = (val_q_df['fallback_used'] == True).sum()
    
    print("\n--- ROI Generation Statistics (Validation Cohort: 541 scans) ---")
    print(f"Machine ROI Rate:         {machine_roi_count}/{total_val} ({machine_roi_count/total_val*100:.2f}%)")
    print(f"Expanded ROI Rate:        {expanded_roi_count}/{total_val} ({expanded_roi_count/total_val*100:.2f}%)")
    print(f"Context ROI Rate:         {context_roi_count}/{total_val} ({context_roi_count/total_val*100:.2f}%)")
    print(f"Fallback ROI Rate:        {fallback_count}/{total_val} ({fallback_count/total_val*100:.2f}%)")
    
    q_scores = val_q_df['roi_quality_score'].values
    dices = val_q_df['seg_dice'].values if 'seg_dice' in val_q_df.columns else np.zeros(total_val)
    
    print("\n--- ROI Quality Score (S_quality) Distribution ---")
    print(f"Mean:   {np.mean(q_scores):.4f} +/- {np.std(q_scores):.4f}")
    print(f"Median: {np.median(q_scores):.4f}")
    print(f"Min:    {np.min(q_scores):.4f}")
    print(f"25th %: {np.percentile(q_scores, 25):.4f}")
    print(f"75th %: {np.percentile(q_scores, 75):.4f}")
    print(f"Max:    {np.max(q_scores):.4f}")
    
    print("\n--- Segmentation Dice Distribution ---")
    print(f"Mean:   {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
    print(f"Median: {np.median(dices):.4f}")
    print(f"Min:    {np.min(dices):.4f}")
    print(f"25th %: {np.percentile(dices, 25):.4f}")
    print(f"75th %: {np.percentile(dices, 75):.4f}")
    print(f"Max:    {np.max(dices):.4f}")
    
    # Save ROI Stats
    roi_stats = {
        'total_validation_scans': total_val,
        'machine_roi_count': int(machine_roi_count),
        'machine_roi_rate_pct': float(machine_roi_count/total_val*100),
        'expanded_roi_count': int(expanded_roi_count),
        'expanded_roi_rate_pct': float(expanded_roi_count/total_val*100),
        'context_roi_count': int(context_roi_count),
        'context_roi_rate_pct': float(context_roi_count/total_val*100),
        'fallback_roi_count': int(fallback_count),
        'fallback_roi_rate_pct': float(fallback_count/total_val*100),
        'roi_quality_score': {
            'mean': float(np.mean(q_scores)),
            'std': float(np.std(q_scores)),
            'median': float(np.median(q_scores)),
            'min': float(np.min(q_scores)),
            'p25': float(np.percentile(q_scores, 25)),
            'p75': float(np.percentile(q_scores, 75)),
            'max': float(np.max(q_scores))
        },
        'segmentation_dice': {
            'mean': float(np.mean(dices)),
            'std': float(np.std(dices)),
            'median': float(np.median(dices)),
            'min': float(np.min(dices)),
            'p25': float(np.percentile(dices, 25)),
            'p75': float(np.percentile(dices, 75)),
            'max': float(np.max(dices))
        }
    }
    
    with open('results/v4/validation/V4_ROI_STATISTICS.json', 'w') as f:
        json.dump(roi_stats, f, indent=2)
    with open(os.path.join(ONEDRIVE_ROOT, 'results/v4/validation/V4_ROI_STATISTICS.json'), 'w') as f:
        json.dump(roi_stats, f, indent=2)

    # 2. Re-evaluate models on val set to get exact predictions & per-case error taxonomy
    from classification.v4.roi_quality import QualityFeatureNormalizer
    with open('data/cache/v4/quality_norm_stats.json', 'r', encoding='utf-8') as f:
        norm_dict = json.load(f)
    q_norm = QualityFeatureNormalizer(
        mean=np.array(norm_dict['mean'], dtype=np.float32),
        std=np.array(norm_dict['std'], dtype=np.float32)
    )
    val_dataset = MultiScaleCancerDatasetV4(
        df=val_q_df,
        crops_dir='data/cache/v4/crops',
        is_training=False,
        quality_normalizer=q_norm
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Evaluate Arm E/F (Multi-scale visual winner) and Arm G (Quality vector) and Arm H
    # Check best checkpoints in models/v4/
    print("\n--- Checkpoints in models/v4/ ---")
    for f in os.listdir('models/v4'):
        if f.endswith('.pth'):
            chk_path = os.path.join('models/v4', f)
            print(f"  {f}: MD5 = {get_md5(chk_path)} ({os.path.getsize(chk_path)/(1024*1024):.1f} MB)")

    # Let's run evaluation on models/v4/vit_v4_best.pth
    chk_best_path = 'models/v4/vit_v4_best.pth'
    if os.path.exists(chk_best_path):
        chk = torch.load(chk_best_path, map_location=device)
        print(f"\nLoaded vit_v4_best.pth: Stage {chk.get('stage', '?')}, Epoch {chk.get('epoch', '?')}, Scales: {chk.get('scales_used')}, QualityVec: {chk.get('use_quality_vector')}, Gating: {chk.get('enable_gating')}")
        
        model = MultiScaleViTV4(
            scales_used=chk.get('scales_used', ('local', 'medium', 'large', 'context')),
            use_quality_vector=chk.get('use_quality_vector', True),
            enable_gating=chk.get('enable_gating', False),
            pretrained=False
        ).to(device)
        model.load_state_dict(chk['model_state_dict'])
        model.eval()
        
        all_targets = []
        all_probs = []
        all_case_ids = []
        
        with torch.no_grad():
            for batch in val_loader:
                loc = batch['local_crop'].to(device)
                med = batch['medium_crop'].to(device)
                lrg = batch['large_crop'].to(device)
                ctx = batch['context_crop'].to(device)
                qvec = batch['quality_vector'].to(device) if chk.get('use_quality_vector', True) else None
                qscore = batch['roi_quality_score'].to(device) if chk.get('enable_gating', False) else None
                
                with autocast('cuda'):
                    logits = model(loc, med, lrg, ctx, qvec, qscore)
                    probs = torch.softmax(logits, dim=-1)[:, 1]
                    
                all_probs.extend(probs.cpu().numpy().tolist())
                all_targets.extend(batch['label'].numpy().tolist())
                all_case_ids.extend(batch['case_id'])
                
        val_pred_df = pd.DataFrame({
            'case_id': all_case_ids,
            'y_true': all_targets,
            'y_prob': all_probs
        })
        val_pred_df['y_pred'] = (val_pred_df['y_prob'] >= 0.50).astype(int)
        
        # Merge with quality and metadata
        val_pred_df = val_pred_df.merge(val_q_df[['case_id', 'seg_dice', 'roi_source', 'fallback_used', 'roi_quality_score']], on='case_id', how='left')
        
        # Compute exact validation metrics
        y_true = val_pred_df['y_true'].values
        y_prob = val_pred_df['y_prob'].values
        y_pred = val_pred_df['y_pred'].values
        
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        binary_f1 = f1_score(y_true, y_pred, average='binary')
        roc_auc = roc_auc_score(y_true, y_prob)
        p_c, r_c, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(r_c, p_c)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn)
        spec = tn / (tn + fp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        print("\n--- vit_v4_best.pth Validation Performance ---")
        print(f"Accuracy:        {acc*100:.2f}%")
        print(f"Macro-F1:        {macro_f1*100:.2f}%")
        print(f"Binary F1:       {binary_f1*100:.2f}%")
        print(f"ROC-AUC:         {roc_auc:.4f}")
        print(f"PR-AUC:          {pr_auc:.4f}")
        print(f"Malignant Recall:{recall*100:.2f}% ({tp}/{tp+fn})")
        print(f"Specificity:     {spec*100:.2f}% ({tn}/{tn+fp})")
        print(f"Precision:       {prec*100:.2f}% ({tp}/{tp+fp})")
        print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        # Error Taxonomy Analysis
        val_pred_df['is_error'] = val_pred_df['y_true'] != val_pred_df['y_pred']
        errors_df = val_pred_df[val_pred_df['is_error']].copy()
        
        def classify_error(row):
            dice = row['seg_dice']
            src = row['roi_source']
            fb = row['fallback_used']
            if dice < 0.10:
                return 'Severe segmentation failure (Dice < 0.10)'
            elif 0.10 <= dice < 0.50:
                return 'Partial localization (0.10 <= Dice < 0.50)'
            elif fb or src in ['context_roi', 'fallback_context']:
                return 'Fallback / context ROI'
            elif dice >= 0.50:
                return 'Good localization but classifier failure'
            else:
                return 'Other'
                
        errors_df['error_category'] = errors_df.apply(classify_error, axis=1)
        val_pred_df['error_category'] = val_pred_df.apply(lambda r: classify_error(r) if r['is_error'] else 'Correct', axis=1)
        
        print("\n--- Validation Error Taxonomy (Total Errors = {}) ---".format(len(errors_df)))
        for cat, cnt in errors_df['error_category'].value_counts().items():
            print(f"  {cat}: {cnt} ({cnt/len(errors_df)*100:.2f}%)")
            
        # Malignant False Negatives Breakdown
        fn_df = val_pred_df[(val_pred_df['y_true'] == 1) & (val_pred_df['y_pred'] == 0)].copy()
        print(f"\n--- Malignant False Negatives Breakdown (Total FN = {len(fn_df)}) ---")
        
        fn_severe = (fn_df['seg_dice'] < 0.10).sum()
        fn_partial = ((fn_df['seg_dice'] >= 0.10) & (fn_df['seg_dice'] < 0.50)).sum()
        fn_fallback = ((fn_df['fallback_used'] == True) & (fn_df['seg_dice'] >= 0.10)).sum()
        fn_good_clf = ((fn_df['seg_dice'] >= 0.50) & (fn_df['fallback_used'] == False)).sum()
        
        print(f"  Severe Localization FN (Dice < 0.10):        {fn_severe} ({fn_severe/len(fn_df)*100:.2f}%)")
        print(f"  Partial Localization FN (0.10 <= Dice < 0.50): {fn_partial} ({fn_partial/len(fn_df)*100:.2f}%)")
        print(f"  Fallback / Context FN:                       {fn_fallback} ({fn_fallback/len(fn_df)*100:.2f}%)")
        print(f"  Good Localization Classifier FN (Dice >= 0.5):{fn_good_clf} ({fn_good_clf/len(fn_df)*100:.2f}%)")
        
        loc_assoc_fn = fn_severe + fn_partial + fn_fallback
        print(f"  Localization/Fallback-Associated FN:          {loc_assoc_fn}/{len(fn_df)} ({loc_assoc_fn/len(fn_df)*100:.2f}%)")
        print(f"  [Comparison with V3-B Locked Benchmark: V3-B was 80/108 = 74.07%]")
        
        # Save error analysis CSV and JSON
        val_pred_df.to_csv('results/v4/validation/V4_VALIDATION_PREDICTIONS.csv', index=False)
        val_pred_df.to_csv(os.path.join(ONEDRIVE_ROOT, 'results/v4/validation/V4_VALIDATION_PREDICTIONS.csv'), index=False)
        
        error_summary = {
            'total_validation_scans': total_val,
            'total_errors': len(errors_df),
            'accuracy_pct': float(acc * 100),
            'macro_f1_pct': float(macro_f1 * 100),
            'binary_f1_pct': float(binary_f1 * 100),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'malignant_recall_pct': float(recall * 100),
            'specificity_pct': float(spec * 100),
            'precision_pct': float(prec * 100),
            'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
            'error_categories': errors_df['error_category'].value_counts().to_dict(),
            'malignant_fn_breakdown': {
                'total_malignant_fn': int(len(fn_df)),
                'severe_localization_fn': int(fn_severe),
                'partial_localization_fn': int(fn_partial),
                'fallback_context_fn': int(fn_fallback),
                'good_localization_classifier_fn': int(fn_good_clf),
                'localization_associated_fn_count': int(loc_assoc_fn),
                'localization_associated_fn_pct': float(loc_assoc_fn / len(fn_df) * 100) if len(fn_df) > 0 else 0.0,
                'v3b_locked_comparison_pct': 74.07
            }
        }
        with open('results/v4/validation/V4_ERROR_ANALYSIS.json', 'w') as f:
            json.dump(error_summary, f, indent=2)
        with open(os.path.join(ONEDRIVE_ROOT, 'results/v4/validation/V4_ERROR_ANALYSIS.json'), 'w') as f:
            json.dump(error_summary, f, indent=2)
            
    print("\nDetailed Validation Analysis Complete!")

if __name__ == '__main__':
    run_detailed_analysis()
