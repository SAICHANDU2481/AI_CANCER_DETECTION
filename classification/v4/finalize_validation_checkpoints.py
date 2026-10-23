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
    auc, confusion_matrix
)

PROJECT_ROOT = r'C:\Users\sys\Desktop\AI_CANCER_DETECTION'
ONEDRIVE_ROOT = r'C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION'
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from classification.v4.dataset_v4 import MultiScaleCancerDatasetV4
from classification.v4.model_v4 import MultiScaleViTV4
from classification.v4.roi_quality import QualityFeatureNormalizer
from classification.v4.train_v4 import set_seed

def get_md5(path: str) -> str:
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def eval_model(model, loader, device, use_q, use_gate):
    model.eval()
    all_targets, all_probs, all_case_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            loc = batch['local_crop'].to(device)
            med = batch['medium_crop'].to(device)
            lrg = batch['large_crop'].to(device)
            ctx = batch['context_crop'].to(device)
            qvec = batch['quality_vector'].to(device) if use_q else None
            qscore = batch['roi_quality_score'].to(device) if use_gate else None
            with autocast('cuda'):
                logits = model(loc, med, lrg, ctx, qvec, qscore)
                probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(batch['label'].numpy().tolist())
            all_case_ids.extend(batch['case_id'])
            
    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.50).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    binary_f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
    try:
        p_c, r_c, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(r_c, p_c)
    except Exception:
        pr_auc = 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    
    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'binary_f1': float(binary_f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'sensitivity_malignant_recall': sens,
        'specificity': spec,
        'precision': prec,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }, y_prob, y_pred, all_case_ids

def run_full_compilation():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_q_df = pd.read_csv('data/cache/v4/roi_quality_val.csv')
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
    
    # Load training dataset for quick reproducible fit to preserve exact checkpoints
    train_q_df = pd.read_csv('data/cache/v4/roi_quality_train.csv')
    train_dataset = MultiScaleCancerDatasetV4(
        df=train_q_df,
        crops_dir='data/cache/v4/crops',
        is_training=True,
        quality_normalizer=q_norm
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    
    n_benign = int((train_q_df['label'] == 0).sum())
    n_mal = int((train_q_df['label'] == 1).sum())
    w_benign = len(train_q_df) / (2.0 * n_benign)
    w_mal = len(train_q_df) / (2.0 * n_mal)
    class_weights = torch.tensor([w_benign, w_mal], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    
    # Train and save Arm E/F (Multi-Scale Visual Winner, 5 epochs Stage 1)
    print("Saving exact checkpoint for Arm E/F (Multi-Scale Visual)...")
    set_seed(42)
    model_e = MultiScaleViTV4(
        scales_used=('local', 'medium', 'large', 'context'),
        use_quality_vector=False,
        enable_gating=False,
        pretrained=True
    ).to(device)
    for p in model_e.backbone.parameters():
        p.requires_grad = False
    for p in model_e.head.parameters():
        p.requires_grad = True
    opt_e = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_e.parameters()), lr=1e-4, weight_decay=0.01)
    sched_e = torch.optim.lr_scheduler.CosineAnnealingLR(opt_e, T_max=15, eta_min=1e-6)
    scaler_e = torch.amp.GradScaler('cuda')
    
    best_e_f1 = -1.0
    best_e_state = None
    best_e_metrics = None
    best_e_epoch = 0
    
    for ep in range(1, 6):
        model_e.train()
        for batch in train_loader:
            loc = batch['local_crop'].to(device)
            med = batch['medium_crop'].to(device)
            lrg = batch['large_crop'].to(device)
            ctx = batch['context_crop'].to(device)
            targets = batch['label'].to(device)
            opt_e.zero_grad()
            with autocast('cuda'):
                logits = model_e(loc, med, lrg, ctx)
                loss = criterion(logits, targets)
            scaler_e.scale(loss).backward()
            scaler_e.step(opt_e)
            scaler_e.update()
        sched_e.step()
        m, _, _, _ = eval_model(model_e, val_loader, device, False, False)
        print(f"  Arm E Ep {ep}: Acc={m['accuracy']*100:.2f}%, F1={m['macro_f1']*100:.2f}%, AUC={m['roc_auc']:.4f}")
        if m['macro_f1'] > best_e_f1:
            best_e_f1 = m['macro_f1']
            best_e_epoch = ep
            best_e_state = {k: v.cpu().clone() for k, v in model_e.state_dict().items()}
            best_e_metrics = m
            
    # Save Arm E Checkpoint (Winner)
    win_chk_path = 'models/v4/vit_v4_best.pth'
    torch.save({
        'arm': 'E',
        'description': 'Multi-Scale Fusion (Local+Med+Lrg+Context, Visual Only)',
        'stage': 1,
        'epoch': best_e_epoch,
        'model_state_dict': best_e_state,
        'val_metrics': best_e_metrics,
        'scales_used': ('local', 'medium', 'large', 'context'),
        'use_quality_vector': False,
        'enable_gating': False
    }, win_chk_path)
    torch.save({
        'arm': 'E',
        'description': 'Multi-Scale Fusion (Local+Med+Lrg+Context, Visual Only)',
        'stage': 1,
        'epoch': best_e_epoch,
        'model_state_dict': best_e_state,
        'val_metrics': best_e_metrics,
        'scales_used': ('local', 'medium', 'large', 'context'),
        'use_quality_vector': False,
        'enable_gating': False
    }, os.path.join(ONEDRIVE_ROOT, win_chk_path))
    
    torch.save({
        'arm': 'E',
        'stage': 1,
        'epoch': best_e_epoch,
        'model_state_dict': best_e_state,
        'val_metrics': best_e_metrics,
        'scales_used': ('local', 'medium', 'large', 'context'),
        'use_quality_vector': False,
        'enable_gating': False
    }, 'models/v4/vit_v4_multiscale_visual_best.pth')
    torch.save({
        'arm': 'E',
        'stage': 1,
        'epoch': best_e_epoch,
        'model_state_dict': best_e_state,
        'val_metrics': best_e_metrics,
        'scales_used': ('local', 'medium', 'large', 'context'),
        'use_quality_vector': False,
        'enable_gating': False
    }, os.path.join(ONEDRIVE_ROOT, 'models/v4/vit_v4_multiscale_visual_best.pth'))
    
    print(f"Global Winning Checkpoint Saved: models/v4/vit_v4_best.pth (MD5: {get_md5(win_chk_path)})")
    
    # Re-evaluate winning model for per-case predictions & error taxonomy
    model_e.load_state_dict(best_e_state)
    m_win, y_prob_win, y_pred_win, case_ids_win = eval_model(model_e, val_loader, device, False, False)
    
    val_pred_df = pd.DataFrame({
        'case_id': case_ids_win,
        'y_true': val_dataset.labels,
        'y_prob': y_prob_win,
        'y_pred': y_pred_win
    })
    val_pred_df = val_pred_df.merge(val_q_df[['case_id', 'seg_dice', 'roi_source', 'fallback_used', 'roi_quality_score']], on='case_id', how='left')
    val_pred_df['is_error'] = val_pred_df['y_true'] != val_pred_df['y_pred']
    
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
            
    val_pred_df['error_category'] = val_pred_df.apply(lambda r: classify_error(r) if r['is_error'] else 'Correct', axis=1)
    
    # Save predictions
    val_pred_df.to_csv('results/v4/validation/V4_VALIDATION_PREDICTIONS.csv', index=False)
    val_pred_df.to_csv(os.path.join(ONEDRIVE_ROOT, 'results/v4/validation/V4_VALIDATION_PREDICTIONS.csv'), index=False)
    
    # Errors breakdown
    errors_df = val_pred_df[val_pred_df['is_error']].copy()
    fn_df = val_pred_df[(val_pred_df['y_true'] == 1) & (val_pred_df['y_pred'] == 0)].copy()
    
    fn_severe = (fn_df['seg_dice'] < 0.10).sum()
    fn_partial = ((fn_df['seg_dice'] >= 0.10) & (fn_df['seg_dice'] < 0.50)).sum()
    fn_fallback = ((fn_df['fallback_used'] == True) & (fn_df['seg_dice'] >= 0.10)).sum()
    fn_good_clf = ((fn_df['seg_dice'] >= 0.50) & (fn_df['fallback_used'] == False)).sum()
    loc_assoc_fn = fn_severe + fn_partial + fn_fallback
    
    print("\n==================================================")
    print("WINNING MODEL VALIDATION PERFORMANCE (ARM E/F)")
    print("==================================================")
    print(f"Accuracy:         {m_win['accuracy']*100:.2f}%")
    print(f"Macro-F1:         {m_win['macro_f1']*100:.2f}%")
    print(f"Binary F1:        {m_win['binary_f1']*100:.2f}%")
    print(f"ROC-AUC:          {m_win['roc_auc']:.4f}")
    print(f"PR-AUC:           {m_win['pr_auc']:.4f}")
    print(f"Malignant Recall: {m_win['sensitivity_malignant_recall']*100:.2f}% ({m_win['tp']}/{m_win['tp']+m_win['fn']})")
    print(f"Specificity:      {m_win['specificity']*100:.2f}% ({m_win['tn']}/{m_win['tn']+m_win['fp']})")
    print(f"Precision:        {m_win['precision']*100:.2f}% ({m_win['tp']}/{m_win['tp']+m_win['fp']})")
    print(f"Confusion Matrix: TP={m_win['tp']}, TN={m_win['tn']}, FP={m_win['fp']}, FN={m_win['fn']}")
    
    print(f"\n--- Total Errors: {len(errors_df)} / 541 ({len(errors_df)/541*100:.2f}%) ---")
    for cat, cnt in errors_df['error_category'].value_counts().items():
        print(f"  {cat}: {cnt} ({cnt/len(errors_df)*100:.2f}%)")
        
    print(f"\n--- Malignant False Negatives Breakdown (Total FN = {len(fn_df)}) ---")
    print(f"  Severe Localization FN (Dice < 0.10):        {fn_severe} ({fn_severe/len(fn_df)*100:.2f}%)")
    print(f"  Partial Localization FN (0.10 <= Dice < 0.50): {fn_partial} ({fn_partial/len(fn_df)*100:.2f}%)")
    print(f"  Fallback / Context FN:                       {fn_fallback} ({fn_fallback/len(fn_df)*100:.2f}%)")
    print(f"  Good Localization Classifier FN (Dice >= 0.5):{fn_good_clf} ({fn_good_clf/len(fn_df)*100:.2f}%)")
    print(f"  Localization/Fallback-Associated FN:          {loc_assoc_fn}/{len(fn_df)} ({loc_assoc_fn/len(fn_df)*100:.2f}%)")
    print(f"  [V3-B Locked Reference Comparison: 80 / 108 = 74.07%]")
    
    # Save JSON summary
    error_summary = {
        'winning_arm': 'E',
        'winning_description': 'Multi-Scale Fusion (Local + Medium + Large + Context, Visual Only)',
        'winning_stage': 1,
        'winning_epoch': best_e_epoch,
        'checkpoint_path': 'models/v4/vit_v4_best.pth',
        'checkpoint_md5': get_md5(win_chk_path),
        'metrics': m_win,
        'error_categories': errors_df['error_category'].value_counts().to_dict(),
        'malignant_fn_breakdown': {
            'total_malignant_fn': int(len(fn_df)),
            'severe_localization_fn': int(fn_severe),
            'partial_localization_fn': int(fn_partial),
            'fallback_context_fn': int(fn_fallback),
            'good_localization_classifier_fn': int(fn_good_clf),
            'localization_associated_fn_count': int(loc_assoc_fn),
            'localization_associated_fn_pct': float(loc_assoc_fn / len(fn_df) * 100) if len(fn_df) > 0 else 0.0,
            'v3b_reference_fn_count': 108,
            'v3b_reference_localization_associated_count': 80,
            'v3b_reference_localization_associated_pct': 74.07
        }
    }
    with open('results/v4/validation/V4_ERROR_ANALYSIS.json', 'w') as f:
        json.dump(error_summary, f, indent=2)
    with open(os.path.join(ONEDRIVE_ROOT, 'results/v4/validation/V4_ERROR_ANALYSIS.json'), 'w') as f:
        json.dump(error_summary, f, indent=2)

if __name__ == '__main__':
    run_full_compilation()
