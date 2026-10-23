import os
import sys
import json
import time
import copy
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix

PROJECT_ROOT = r'C:\Users\sys\Desktop\AI_CANCER_DETECTION'
ONEDRIVE_ROOT = r'C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION'
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from classification.v4.model_v4 import MultiScaleViTV4
from classification.v4.dataset_v4 import MultiScaleCancerDatasetV4
from classification.v4.roi_quality import QualityFeatureNormalizer, QUALITY_FEATURE_NAMES

logger = logging.getLogger('TrainV4')
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.50).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    binary_f1 = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5
        
    try:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = 0.0
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'binary_f1': binary_f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'sensitivity_malignant_recall': sens,
        'specificity': spec,
        'precision': prec,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }

def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scales_used: Tuple[str, ...],
    use_quality_vector: bool,
    enable_gating: bool
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    all_targets = []
    all_probs = []
    all_preds = []
    records = []
    
    with torch.no_grad():
        for batch in loader:
            loc = batch['local_crop'].to(device)
            med = batch['medium_crop'].to(device)
            lrg = batch['large_crop'].to(device)
            ctx = batch['context_crop'].to(device)
            qvec = batch['quality_vector'].to(device) if use_quality_vector else None
            qscore = batch['roi_quality_score'].to(device) if enable_gating else None
            targets = batch['label'].to(device)
            
            with autocast('cuda'):
                logits = model(loc, med, lrg, ctx, qvec, qscore)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                
            p_np = probs.cpu().numpy()
            t_np = targets.cpu().numpy()
            all_targets.extend(t_np)
            all_probs.extend(p_np)
            all_preds.extend((p_np >= 0.50).astype(int))
            
            for i in range(len(t_np)):
                records.append({
                    'case_id': batch['case_id'][i],
                    'patient_id': batch['patient_id'][i],
                    'abnormality_type': batch['abnormality_type'][i],
                    'pathology': batch['pathology'][i],
                    'roi_source': batch['roi_source'][i],
                    'ground_truth': int(t_np[i]),
                    'predicted_probability': float(p_np[i]),
                    'predicted_label': int(p_np[i] >= 0.50),
                    'correct': bool((p_np[i] >= 0.50) == t_np[i]),
                    'seg_dice': float(batch['seg_dice'][i]),
                    'seg_iou': float(batch['seg_iou'][i]),
                    'roi_quality_score': float(batch['roi_quality_score'][i])
                })
                
    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    metrics = compute_metrics(y_true, y_prob)
    pred_df = pd.DataFrame(records)
    return metrics, pred_df

def run_staged_training(
    train_quality_csv: str = 'data/cache/v4/roi_quality_train.csv',
    val_quality_csv: str = 'data/cache/v4/roi_quality_val.csv',
    norm_stats_json: str = 'data/cache/v4/quality_norm_stats.json',
    scales_used: Tuple[str, ...] = ('local', 'medium', 'large', 'context'),
    use_quality_vector: bool = True,
    enable_gating: bool = False,
    gating_threshold: float = 0.40,
    batch_size: int = 16,
    seed: int = 42
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Starting V4-A Staged Training on {device} (scales={scales_used}, quality_vec={use_quality_vector}, gating={enable_gating})...")
    
    # 1. Load Data
    train_df = pd.read_csv(train_quality_csv)
    val_df = pd.read_csv(val_quality_csv)
    
    with open(norm_stats_json, 'r', encoding='utf-8') as f:
        norm_dict = json.load(f)
    normalizer = QualityFeatureNormalizer(
        mean=np.array(norm_dict['mean'], dtype=np.float32),
        std=np.array(norm_dict['std'], dtype=np.float32)
    )
    
    train_ds = MultiScaleCancerDatasetV4(train_df, is_training=True, quality_normalizer=normalizer)
    val_ds = MultiScaleCancerDatasetV4(val_df, is_training=False, quality_normalizer=normalizer)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 2. Compute Class Weights strictly from Training Set
    n_benign = int((train_df['label'] == 0).sum())
    n_mal = int((train_df['label'] == 1).sum())
    total_train = len(train_df)
    w_benign = total_train / (2.0 * n_benign)
    w_mal = total_train / (2.0 * n_mal)
    class_weights = torch.tensor([w_benign, w_mal], dtype=torch.float32, device=device)
    logger.info(f"Training Class Counts: Benign={n_benign}, Malignant={n_mal}. Class Weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    
    # 3. Initialize Model
    model = MultiScaleViTV4(
        model_name='vit_base_patch16_224.augreg2_in21k_ft_in1k',
        num_classes=2,
        pretrained=True,
        dropout=0.20,
        scales_used=scales_used,
        use_quality_vector=use_quality_vector,
        enable_gating=enable_gating,
        gating_threshold=gating_threshold
    ).to(device)
    
    # =============================================================
    # STAGE 1: ViT Backbone Frozen, Train Fusion Head Only
    # =============================================================
    logger.info("==================================================")
    logger.info("STAGE 1: WARMUP (ViT Backbone Frozen, LR=1e-4, Max 15 Epochs)")
    logger.info("==================================================")
    
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    if enable_gating:
        for p in model.head_path_b.parameters():
            p.requires_grad = True
            
    optimizer_s1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.01)
    scheduler_s1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=15, eta_min=1e-6)
    scaler_s1 = GradScaler('cuda')
    
    best_val_f1_s1 = -1.0
    best_s1_state = None
    best_s1_epoch = 0
    patience_s1 = 5
    patience_counter_s1 = 0
    stage1_history = []
    
    for epoch in range(1, 16):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        
        for batch in train_loader:
            loc = batch['local_crop'].to(device)
            med = batch['medium_crop'].to(device)
            lrg = batch['large_crop'].to(device)
            ctx = batch['context_crop'].to(device)
            qvec = batch['quality_vector'].to(device) if use_quality_vector else None
            qscore = batch['roi_quality_score'].to(device) if enable_gating else None
            targets = batch['label'].to(device)
            
            optimizer_s1.zero_grad()
            with autocast('cuda'):
                logits = model(loc, med, lrg, ctx, qvec, qscore)
                loss = criterion(logits, targets)
                
            scaler_s1.scale(loss).backward()
            scaler_s1.step(optimizer_s1)
            scaler_s1.update()
            
            train_loss += loss.item() * len(targets)
            
        scheduler_s1.step()
        train_loss /= len(train_ds)
        
        val_metrics, _ = evaluate_loader(model, val_loader, device, scales_used, use_quality_vector, enable_gating)
        val_f1 = val_metrics['macro_f1']
        val_acc = val_metrics['accuracy']
        val_auc = val_metrics['roc_auc']
        
        stage1_history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_acc': val_acc,
            'val_macro_f1': val_f1, 'val_roc_auc': val_auc
        })
        
        logger.info(f"Stage 1 [Epoch {epoch:02d}/15] Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}% | Val AUC: {val_auc:.4f} ({time.time()-t0:.1f}s)")
        
        if val_f1 > best_val_f1_s1:
            best_val_f1_s1 = val_f1
            best_s1_epoch = epoch
            best_s1_state = copy.deepcopy(model.state_dict())
            patience_counter_s1 = 0
            # Save stage 1 checkpoint
            s1_chk_path = 'models/v4/vit_v4_stage1_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_s1_state,
                'val_metrics': val_metrics,
                'scales_used': scales_used,
                'use_quality_vector': use_quality_vector,
                'enable_gating': enable_gating
            }, s1_chk_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_s1_state,
                'val_metrics': val_metrics
            }, os.path.join(ONEDRIVE_ROOT, s1_chk_path))
        else:
            patience_counter_s1 += 1
            if patience_counter_s1 >= patience_s1:
                logger.info(f"Stage 1 Early stopping triggered at epoch {epoch} (Best: Epoch {best_s1_epoch}, Macro-F1: {best_val_f1_s1*100:.2f}%)")
                break
                
    # =============================================================
    # STAGE 2: Unfreeze Top 2 ViT Blocks, Joint Fine-Tuning
    # =============================================================
    logger.info("==================================================")
    logger.info("STAGE 2: JOINT FINE-TUNING (Top 2 ViT Blocks Unfrozen, Backbone LR=1e-5, Head LR=1e-4)")
    logger.info("==================================================")
    
    # Load best Stage 1 weights
    model.load_state_dict(best_s1_state)
    
    # Unfreeze top 2 ViT blocks
    for p in model.backbone.parameters():
        p.requires_grad = False
    for block in model.backbone.blocks[-2:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.backbone.norm.parameters():
        p.requires_grad = True
        
    for p in model.head.parameters():
        p.requires_grad = True
    if enable_gating:
        for p in model.head_path_b.parameters():
            p.requires_grad = True
            
    # Differential learning rates
    backbone_params = []
    for block in model.backbone.blocks[-2:]:
        backbone_params.extend([p for p in block.parameters() if p.requires_grad])
    backbone_params.extend([p for p in model.backbone.norm.parameters() if p.requires_grad])
    
    head_params = list(model.head.parameters())
    if enable_gating:
        head_params.extend(list(model.head_path_b.parameters()))
        
    optimizer_s2 = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-4}
    ], weight_decay=0.01)
    
    scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=30, eta_min=1e-6)
    scaler_s2 = GradScaler('cuda')
    
    best_val_f1_s2 = -1.0
    best_s2_state = None
    best_s2_epoch = 0
    best_s2_metrics = None
    best_s2_pred_df = None
    patience_s2 = 8
    patience_counter_s2 = 0
    stage2_history = []
    
    for epoch in range(1, 31):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        
        for batch in train_loader:
            loc = batch['local_crop'].to(device)
            med = batch['medium_crop'].to(device)
            lrg = batch['large_crop'].to(device)
            ctx = batch['context_crop'].to(device)
            qvec = batch['quality_vector'].to(device) if use_quality_vector else None
            qscore = batch['roi_quality_score'].to(device) if enable_gating else None
            targets = batch['label'].to(device)
            
            optimizer_s2.zero_grad()
            with autocast('cuda'):
                logits = model(loc, med, lrg, ctx, qvec, qscore)
                loss = criterion(logits, targets)
                
            scaler_s2.scale(loss).backward()
            scaler_s2.step(optimizer_s2)
            scaler_s2.update()
            
            train_loss += loss.item() * len(targets)
            
        scheduler_s2.step()
        train_loss /= len(train_ds)
        
        val_metrics, val_pred_df = evaluate_loader(model, val_loader, device, scales_used, use_quality_vector, enable_gating)
        val_f1 = val_metrics['macro_f1']
        val_acc = val_metrics['accuracy']
        val_auc = val_metrics['roc_auc']
        
        stage2_history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_acc': val_acc,
            'val_macro_f1': val_f1, 'val_roc_auc': val_auc
        })
        
        logger.info(f"Stage 2 [Epoch {epoch:02d}/30] Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}% | Val AUC: {val_auc:.4f} ({time.time()-t0:.1f}s)")
        
        if val_f1 > best_val_f1_s2:
            best_val_f1_s2 = val_f1
            best_s2_epoch = epoch
            best_s2_state = copy.deepcopy(model.state_dict())
            best_s2_metrics = val_metrics
            best_s2_pred_df = val_pred_df
            patience_counter_s2 = 0
            
            # Save stage 2 checkpoint
            s2_chk_path = 'models/v4/vit_v4_stage2_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_s2_state,
                'val_metrics': val_metrics,
                'scales_used': scales_used,
                'use_quality_vector': use_quality_vector,
                'enable_gating': enable_gating
            }, s2_chk_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_s2_state,
                'val_metrics': val_metrics
            }, os.path.join(ONEDRIVE_ROOT, s2_chk_path))
        else:
            patience_counter_s2 += 1
            if patience_counter_s2 >= patience_s2:
                logger.info(f"Stage 2 Early stopping triggered at epoch {epoch} (Best: Epoch {best_s2_epoch}, Macro-F1: {best_val_f1_s2*100:.2f}%)")
                break
                
    # Save Global Winner
    global_chk_path = 'models/v4/vit_v4_best.pth'
    torch.save({
        'stage': 2 if best_val_f1_s2 >= best_val_f1_s1 else 1,
        'epoch': best_s2_epoch if best_val_f1_s2 >= best_val_f1_s1 else best_s1_epoch,
        'model_state_dict': best_s2_state if best_val_f1_s2 >= best_val_f1_s1 else best_s1_state,
        'val_metrics': best_s2_metrics if best_val_f1_s2 >= best_val_f1_s1 else val_metrics,
        'scales_used': scales_used,
        'use_quality_vector': use_quality_vector,
        'enable_gating': enable_gating
    }, global_chk_path)
    torch.save({
        'stage': 2 if best_val_f1_s2 >= best_val_f1_s1 else 1,
        'epoch': best_s2_epoch if best_val_f1_s2 >= best_val_f1_s1 else best_s1_epoch,
        'model_state_dict': best_s2_state if best_val_f1_s2 >= best_val_f1_s1 else best_s1_state,
        'val_metrics': best_s2_metrics if best_val_f1_s2 >= best_val_f1_s1 else val_metrics
    }, os.path.join(ONEDRIVE_ROOT, global_chk_path))
    
    return {
        'stage1_best_epoch': best_s1_epoch,
        'stage1_best_f1': best_val_f1_s1,
        'stage2_best_epoch': best_s2_epoch,
        'stage2_best_f1': best_val_f1_s2,
        'best_val_metrics': best_s2_metrics,
        'best_pred_df': best_s2_pred_df,
        'stage1_history': stage1_history,
        'stage2_history': stage2_history
    }
