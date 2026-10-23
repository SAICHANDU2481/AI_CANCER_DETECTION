import os, sys, time, logging, json, torch, torch.nn as nn, numpy as np, pandas as pd
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, recall_score, confusion_matrix

sys.path.insert(0, os.path.abspath('.'))
from classification.model_v3b import MultiScaleViTV3B
from classification.dataset_v3b import MammographyMultiScaleDatasetV3B

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger('AblationV3B')

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
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets, all_probs = [], []
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
        all_probs.extend(probs)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_probs))
    metrics['val_loss'] = float(avg_loss)
    return metrics, np.array(all_probs), np.array(all_targets)

def run_ablation(config: Dict[str, Any], train_df, val_df, roi_train_df, roi_val_df, device):
    set_seed(42)
    ablation_id = config['id']
    logger.info(f'=== Starting Ablation {ablation_id}: {config["name"]} ===')

    local_size = config.get('local_size', (224, 224))
    context_size = (224, 224)
    batch_size = config.get('batch_size', 8)
    grad_accum_steps = config.get('grad_accum', 2)

    train_ds = MammographyMultiScaleDatasetV3B(
        train_df, split_name='train', is_training=True,
        local_size=local_size, context_size=context_size,
        preload_to_ram=True, roi_meta_df=roi_train_df
    )
    val_ds = MammographyMultiScaleDatasetV3B(
        val_df, split_name='val', is_training=False,
        local_size=local_size, context_size=context_size,
        preload_to_ram=True, roi_meta_df=roi_val_df
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    labels_train = train_df['pathology'].apply(lambda p: 1 if 'MALIGNANT' in str(p).upper() else 0).values
    n_pos = np.sum(labels_train == 1)
    n_neg = np.sum(labels_train == 0)
    w_pos = n_neg / n_pos if n_pos > 0 else 1.0
    class_weights = torch.tensor([1.0, w_pos], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    model = MultiScaleViTV3B(
        fusion_mode=config['fusion_mode'],
        scales_used=config.get('scales_used', ('local', 'medium', 'large')),
        dropout=0.20
    ).to(device)

    model.freeze_backbone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    epochs = config.get('epochs', 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_macro_f1 = -1.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, grad_accum_steps)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(f'Epoch {epoch:02d}/{epochs} | Tr Loss: {train_loss:.4f} | Val Loss: {val_metrics["val_loss"]:.4f} | Val F1: {val_metrics["macro_f1"]*100:.2f}% | Val Acc: {val_metrics["accuracy"]*100:.2f}% | Val AUC: {val_metrics["roc_auc"]:.4f} | ({elapsed:.1f}s)')

        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            best_metrics = val_metrics.copy()
            best_metrics['best_epoch'] = epoch

    logger.info(f'Ablation {ablation_id} Complete | Best Val Macro-F1: {best_metrics["macro_f1"]*100:.2f}% | Best Epoch: {best_metrics["best_epoch"]}')
    return best_metrics

def main():
    os.makedirs('results/v3b/ablation', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Running V3-B Validation Ablation Suite on {device}')

    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    roi_train_df = pd.read_csv('data/cache/v3b_rois/train_rois_v3b.csv') if os.path.exists('data/cache/v3b_rois/train_rois_v3b.csv') else None
    roi_val_df = pd.read_csv('data/cache/v3b_rois/val_rois_v3b.csv') if os.path.exists('data/cache/v3b_rois/val_rois_v3b.csv') else None

    ablations = [
        {
            'id': 'A', 'name': 'Baseline Single ROI (Local Only 224x224)',
            'fusion_mode': 'single_roi', 'scales_used': ('local',), 'local_size': (224, 224),
            'epochs': 10, 'batch_size': 8, 'grad_accum': 2
        },
        {
            'id': 'B', 'name': 'Machine-ROI Local Only (224x224)',
            'fusion_mode': 'single_roi', 'scales_used': ('local',), 'local_size': (224, 224),
            'epochs': 10, 'batch_size': 8, 'grad_accum': 2
        },
        {
            'id': 'C', 'name': 'Local + Medium Context (MLP Fusion 224x224)',
            'fusion_mode': 'mlp', 'scales_used': ('local', 'medium'), 'local_size': (224, 224),
            'epochs': 10, 'batch_size': 8, 'grad_accum': 2
        },
        {
            'id': 'D', 'name': 'Local + Medium + Large Context (MLP Fusion 224x224)',
            'fusion_mode': 'mlp', 'scales_used': ('local', 'medium', 'large'), 'local_size': (224, 224),
            'epochs': 10, 'batch_size': 8, 'grad_accum': 2
        },
        {
            'id': 'E', 'name': 'Cross-Attention Fusion (Local attends to Medium+Large 224x224)',
            'fusion_mode': 'cross_attention', 'scales_used': ('local', 'medium', 'large'), 'local_size': (224, 224),
            'epochs': 10, 'batch_size': 8, 'grad_accum': 2
        },
        {
            'id': 'F', 'name': 'Cross-Attention + High-Res Local (384x384 Local + 224x224 Context)',
            'fusion_mode': 'cross_attention', 'scales_used': ('local', 'medium', 'large'), 'local_size': (384, 384),
            'epochs': 10, 'batch_size': 4, 'grad_accum': 4
        }
    ]

    results = []
    for ab_cfg in ablations:
        m = run_ablation(ab_cfg, train_df, val_df, roi_train_df, roi_val_df, device)
        results.append({
            'Ablation_ID': ab_cfg['id'],
            'Configuration': ab_cfg['name'],
            'Fusion_Mode': ab_cfg['fusion_mode'],
            'Local_Resolution': f"{ab_cfg['local_size'][0]}x{ab_cfg['local_size'][1]}",
            'Context_Resolution': '224x224',
            'Best_Epoch': m['best_epoch'],
            'Val_Accuracy_%': round(m['accuracy'] * 100, 2),
            'Val_Macro_F1_%': round(m['macro_f1'] * 100, 2),
            'Val_ROC_AUC': round(m['roc_auc'], 4),
            'Val_PR_AUC': round(m['pr_auc'], 4),
            'Val_Malignant_Recall_%': round(m['malignant_recall'] * 100, 2),
            'Val_Specificity_%': round(m['specificity'] * 100, 2)
        })

    res_df = pd.DataFrame(results)
    csv_out = 'results/v3b/ablation/V3B_ABLATION_RESULTS.csv'
    res_df.to_csv(csv_out, index=False)
    logger.info(f'Saved ablation results to {csv_out}')

    rows_md = []
    for _, r in res_df.iterrows():
        aid = r['Ablation_ID']
        cfg = r['Configuration']
        fm = r['Fusion_Mode']
        lr = r['Local_Resolution']
        acc = f"{r['Val_Accuracy_%']:.2f}%"
        f1 = f"**{r['Val_Macro_F1_%']:.2f}%**"
        roc = f"{r['Val_ROC_AUC']:.4f}"
        pr = f"{r['Val_PR_AUC']:.4f}"
        rec = f"{r['Val_Malignant_Recall_%']:.2f}%"
        spc = f"{r['Val_Specificity_%']:.2f}%"
        rows_md.append(f"| **{aid}** | {cfg} | `{fm}` | {lr} | {acc} | {f1} | {roc} | {pr} | {rec} | {spc} |")

    table_body = chr(10).join(rows_md)
    hdr = "| Ablation ID | Configuration | Fusion Mode | Local Res | Val Acc (%) | Val Macro-F1 (%) | Val ROC-AUC | Val PR-AUC | Malignant Recall (%) | Specificity (%) |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    md_report = f"# V3-B Multi-Scale ROI Classification Ablation Suite Report\n\n## Controlled Validation Ablations (Quarantined Test Cohort)\n\n{hdr}\n{sep}\n{table_body}\n"

    with open('results/v3b/ablation/V3B_ABLATION_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(md_report)
    logger.info('Wrote results/v3b/ablation/V3B_ABLATION_REPORT.md')

if __name__ == '__main__':
    main()