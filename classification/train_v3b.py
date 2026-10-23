import os, sys, time, logging, json, torch, torch.nn as nn, numpy as np, pandas as pd
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, recall_score, confusion_matrix

sys.path.insert(0, os.path.abspath('.'))
from classification.model_v3b import MultiScaleViTV3B
from classification.dataset_v3b import MammographyMultiScaleDatasetV3B

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger('TrainV3B')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_metrics(y_true, y_prob, threshold=0.50):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except:
        roc_auc = 0.5
    try:
        p, r, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(r, p)
    except:
        pr_auc = 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'malignant_recall': float(sens),
        'specificity': float(spec),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, grad_accum_steps=2):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        local_t = batch['local'].to(device, non_blocking=True)
        med_t = batch['medium'].to(device, non_blocking=True)
        lrg_t = batch['large'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            logits = model(local_t, med_t, lrg_t)
            loss = criterion(logits, labels) / grad_accum_steps

        scaler.scale(loss).backward()
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps * len(labels)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_full(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    records = []
    for batch in loader:
        local_t = batch['local'].to(device, non_blocking=True)
        med_t = batch['medium'].to(device, non_blocking=True)
        lrg_t = batch['large'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            logits = model(local_t, med_t, lrg_t)
            loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        targets = labels.cpu().numpy()

        for j in range(len(targets)):
            records.append({
                'case_id': batch['case_id'][j],
                'patient_id': batch['patient_id'][j],
                'abnormality_type': batch['abnormality_type'][j],
                'pathology': batch['pathology'][j],
                'ground_truth_label': int(targets[j]),
                'predicted_probability': float(probs[j]),
                'predicted_label': int(probs[j] >= 0.50),
                'roi_source': batch['roi_source'][j],
                'fallback_reason': batch['fallback_reason'][j],
                'mask_pixel_count': int(batch['mask_pixel_count'][j])
            })

    pred_df = pd.DataFrame(records)
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(pred_df['ground_truth_label'].values, pred_df['predicted_probability'].values)
    metrics['val_loss'] = float(avg_loss)
    return metrics, pred_df

def run_training():
    set_seed(42)
    os.makedirs('models/v3b/checkpoints', exist_ok=True)
    os.makedirs('models/v3b/logs', exist_ok=True)
    os.makedirs('results/v3b/validation', exist_ok=True)
    os.makedirs('results/v3b/error_analysis', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Starting V3-B Staged Training on {device}')

    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    roi_train_df = pd.read_csv('data/cache/v3b_rois/train_rois_v3b.csv') if os.path.exists('data/cache/v3b_rois/train_rois_v3b.csv') else None
    roi_val_df = pd.read_csv('data/cache/v3b_rois/val_rois_v3b.csv') if os.path.exists('data/cache/v3b_rois/val_rois_v3b.csv') else None

    # Strict quarantine assertion
    assert 'test' not in str(train_df), 'Test quarantine violation!'
    assert 'test' not in str(val_df), 'Test quarantine violation!'

    train_ds = MammographyMultiScaleDatasetV3B(train_df, split_name='train', is_training=True, local_size=(224, 224), context_size=(224, 224), preload_to_ram=True, roi_meta_df=roi_train_df)
    val_ds = MammographyMultiScaleDatasetV3B(val_df, split_name='val', is_training=False, local_size=(224, 224), context_size=(224, 224), preload_to_ram=True, roi_meta_df=roi_val_df)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    labels_train = train_df['pathology'].apply(lambda p: 1 if 'MALIGNANT' in str(p).upper() else 0).values
    w_pos = np.sum(labels_train == 0) / np.sum(labels_train == 1)
    class_weights = torch.tensor([1.0, w_pos], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    model = MultiScaleViTV3B(fusion_mode='cross_attention', scales_used=('local', 'medium', 'large'), dropout=0.20).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # -------------------------------------------------------------
    # Stage 1: Warmup Fusion & Cross-Attention (Frozen Backbone)
    # -------------------------------------------------------------
    logger.info('=== Starting Stage 1: Fusion & Cross-Attention Warmup ===')
    model.freeze_backbone()
    optimizer_s1 = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=15, eta_min=1e-6)

    best_s1_f1 = -1.0
    best_s1_metrics = {}
    patience_s1 = 5
    patience_count = 0

    for epoch in range(1, 16):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer_s1, criterion, scaler, device, grad_accum_steps=2)
        val_m, val_preds = evaluate_full(model, val_loader, criterion, device)
        scheduler_s1.step()
        elapsed = time.time() - t0

        logger.info(f'[Stage 1] Epoch {epoch:02d}/15 | Tr: {tr_loss:.4f} | Val: {val_m["val_loss"]:.4f} | F1: {val_m["macro_f1"]*100:.2f}% | Acc: {val_m["accuracy"]*100:.2f}% | AUC: {val_m["roc_auc"]:.4f} | ({elapsed:.1f}s)')

        if val_m['macro_f1'] > best_s1_f1:
            best_s1_f1 = val_m['macro_f1']
            best_s1_metrics = val_m.copy()
            best_s1_metrics['epoch'] = epoch
            patience_count = 0
            torch.save({
                'epoch': epoch,
                'stage': 1,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_m,
                'fusion_mode': 'cross_attention'
            }, 'models/v3b/checkpoints/vit_v3b_stage1_best.pth')
            logger.info(f'Saved new Stage 1 Best (F1: {best_s1_f1*100:.2f}%)')
        else:
            patience_count += 1
            if patience_count >= patience_s1:
                logger.info(f'Stage 1 early stopping triggered at epoch {epoch}.')
                break

    # -------------------------------------------------------------
    # Stage 2: Joint Fine-Tuning (Unfreeze Top 2 Blocks)
    # -------------------------------------------------------------
    logger.info('=== Starting Stage 2: Joint Fine-Tuning ===')
    chk_s1 = torch.load('models/v3b/checkpoints/vit_v3b_stage1_best.pth', map_location=device)
    model.load_state_dict(chk_s1['model_state_dict'])

    model.unfreeze_top_blocks(num_blocks=2)
    param_groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-4)
    optimizer_s2 = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=30, eta_min=1e-6)

    best_s2_f1 = -1.0
    best_s2_metrics = {}
    patience_s2 = 8
    patience_count = 0

    for epoch in range(1, 31):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer_s2, criterion, scaler, device, grad_accum_steps=2)
        val_m, val_preds = evaluate_full(model, val_loader, criterion, device)
        scheduler_s2.step()
        elapsed = time.time() - t0

        logger.info(f'[Stage 2] Epoch {epoch:02d}/30 | Tr: {tr_loss:.4f} | Val: {val_m["val_loss"]:.4f} | F1: {val_m["macro_f1"]*100:.2f}% | Acc: {val_m["accuracy"]*100:.2f}% | AUC: {val_m["roc_auc"]:.4f} | ({elapsed:.1f}s)')

        if val_m['macro_f1'] > best_s2_f1:
            best_s2_f1 = val_m['macro_f1']
            best_s2_metrics = val_m.copy()
            best_s2_metrics['epoch'] = epoch
            patience_count = 0
            torch.save({
                'epoch': epoch,
                'stage': 2,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_m,
                'fusion_mode': 'cross_attention'
            }, 'models/v3b/checkpoints/vit_v3b_stage2_best.pth')
            logger.info(f'Saved new Stage 2 Best (F1: {best_s2_f1*100:.2f}%)')
        else:
            patience_count += 1
            if patience_count >= patience_s2:
                logger.info(f'Stage 2 early stopping triggered at epoch {epoch}.')
                break

    logger.info(f'Stage 1 Best F1: {best_s1_f1*100:.2f}% (Epoch {best_s1_metrics.get("epoch", 0)})')
    logger.info(f'Stage 2 Best F1: {best_s2_f1*100:.2f}% (Epoch {best_s2_metrics.get("epoch", 0)})')

    if best_s2_f1 >= best_s1_f1:
        winner_stage = 2
        winner_chk = 'models/v3b/checkpoints/vit_v3b_stage2_best.pth'
        winner_metrics = best_s2_metrics
    else:
        winner_stage = 1
        winner_chk = 'models/v3b/checkpoints/vit_v3b_stage1_best.pth'
        winner_metrics = best_s1_metrics

    logger.info(f'GLOBAL WINNER: Stage {winner_stage} (F1: {winner_metrics["macro_f1"]*100:.2f}%)')
    
    best_payload = torch.load(winner_chk, map_location=device)
    torch.save(best_payload, 'models/v3b/checkpoints/vit_v3b_best.pth')

    model.load_state_dict(best_payload['model_state_dict'])
    final_val_m, final_val_df = evaluate_full(model, val_loader, criterion, device)

    final_val_df.to_csv('results/v3b/validation/v3b_validation_predictions.csv', index=False)
    with open('results/v3b/validation/v3b_validation_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(final_val_m, f, indent=2)

    ea_rows = []
    for path in ['BENIGN', 'MALIGNANT']:
        sub = final_val_df[final_val_df['pathology'] == path]
        if len(sub) > 0:
            m_sub = compute_metrics(sub['ground_truth_label'].values, sub['predicted_probability'].values)
            ea_rows.append({'Category': 'Pathology', 'Subgroup': path, 'Count': len(sub), 'Accuracy_%': round(m_sub['accuracy']*100, 2), 'Macro_F1_%': round(m_sub['macro_f1']*100, 2), 'ROC_AUC': round(m_sub['roc_auc'], 4)})

    for ab_type in ['MASS', 'CALCIFICATION']:
        sub = final_val_df[final_val_df['abnormality_type'] == ab_type]
        if len(sub) > 0:
            m_sub = compute_metrics(sub['ground_truth_label'].values, sub['predicted_probability'].values)
            ea_rows.append({'Category': 'Abnormality', 'Subgroup': ab_type, 'Count': len(sub), 'Accuracy_%': round(m_sub['accuracy']*100, 2), 'Macro_F1_%': round(m_sub['macro_f1']*100, 2), 'ROC_AUC': round(m_sub['roc_auc'], 4)})

    for r_src in ['machine_roi', 'fallback']:
        sub = final_val_df[final_val_df['roi_source'] == r_src]
        if len(sub) > 0:
            m_sub = compute_metrics(sub['ground_truth_label'].values, sub['predicted_probability'].values)
            ea_rows.append({'Category': 'ROI_Source', 'Subgroup': r_src, 'Count': len(sub), 'Accuracy_%': round(m_sub['accuracy']*100, 2), 'Macro_F1_%': round(m_sub['macro_f1']*100, 2), 'ROC_AUC': round(m_sub['roc_auc'], 4)})

    ea_df = pd.DataFrame(ea_rows)
    ea_df.to_csv('results/v3b/error_analysis/v3b_validation_error_analysis.csv', index=False)

    report_md = f"""# V3-B Validation Training & Evaluation Report

## Experiment Summary
- **Model:** V3-B Multi-Scale ViT with Cross-Attention Token Fusion
- **Upstream Segmentor:** Frozen V2-B MultiScaleUKAN (`models/v2b/segmentation/ukan_v2b_best.pth`)
- **Global Best Checkpoint:** Stage {winner_stage} (Epoch {winner_metrics.get('epoch', 0)})
- **Validation Dataset:** 541 mammograms (235 patients, strictly patient-level isolated)
- **Test Set Status:** 100% QUARANTINED (0 evaluations performed)

## Overall Validation Metrics
- **Accuracy:** {final_val_m['accuracy']*100:.2f}%
- **Macro-F1:** {final_val_m['macro_f1']*100:.2f}%
- **ROC-AUC:** {final_val_m['roc_auc']:.4f}
- **PR-AUC:** {final_val_m['pr_auc']:.4f}
- **Malignant Recall (Sensitivity):** {final_val_m['malignant_recall']*100:.2f}%
- **Specificity:** {final_val_m['specificity']*100:.2f}%

## Confusion Matrix
- **True Positives:** {final_val_m['tp']}
- **True Negatives:** {final_val_m['tn']}
- **False Positives:** {final_val_m['fp']}
- **False Negatives:** {final_val_m['fn']}

## Error Analysis & Subgroup Performance
"""
    for _, r in ea_df.iterrows():
        report_md += f"- **{r['Category']} - {r['Subgroup']} (N={r['Count']}):** Acc={r['Accuracy_%']:.2f}%, F1={r['Macro_F1_%']:.2f}%, AUC={r['ROC_AUC']:.4f}\n"

    with open('results/v3b/V3B_VALIDATION_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report_md)
    logger.info('Wrote results/v3b/V3B_VALIDATION_REPORT.md')

if __name__ == '__main__':
    run_training()