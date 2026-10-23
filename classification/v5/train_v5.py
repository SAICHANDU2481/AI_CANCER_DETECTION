"""
================================================================================
V5 Training Engine: Multi-Scale Top-K + Full-Mammogram Vision Transformer
================================================================================
Strict Governance:
- 100% Isolated Train (2501) and Val (541) cohorts
- ZERO access to test.csv
- Two-stage training:
    Stage 1: Fusion Head & Candidate Aggregator Warmup (Backbone Frozen)
    Stage 2: End-to-End Fine-Tuning of Top ViT Transformer Blocks
================================================================================
"""

import os
import sys
import copy
import time
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
ROOTS = [PROJECT_ROOT, ONEDRIVE_ROOT]

if os.path.exists(PROJECT_ROOT):
    os.chdir(PROJECT_ROOT)

for r in ROOTS:
    if r not in sys.path and os.path.exists(r):
        sys.path.insert(0, r)

from classification.v5.model_v5 import MultiScaleViTV5
from classification.v5.dataset_v5 import get_v5_dataloaders

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainV5")

# Quarantine guard
FORBIDDEN = ['test.csv', 'verified_test.csv', 'test_predictions', 'test_masks']
for f in FORBIDDEN:
    assert f not in __file__, f"Test string found in execution script: {f}"


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    y_pred = (y_prob >= 0.50).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    binary_f1 = float(f1_score(y_true, y_pred, average='binary', zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.50

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


def evaluate_loader_v5(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scales_used: Tuple[str, ...],
    use_full_image: bool,
    use_calc_patch: bool
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            targets = batch['label'].to(device)
            full_img = batch['full_image'].to(device) if use_full_image else None
            calc_img = batch['calc_enhanced'].to(device) if use_calc_patch and 'calc_enhanced' in batch else None

            cand_crops = {s: batch[s].to(device) for s in scales_used}

            with autocast('cuda'):
                logits = model(cand_crops, full_image_tensor=full_img, calc_patch_tensor=calc_img)
                probs = torch.softmax(logits, dim=-1)[:, 1]

            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().float().numpy().tolist())

    y_true = np.array(all_targets, dtype=int)
    y_prob = np.array(all_probs, dtype=np.float32)
    metrics = compute_metrics(y_true, y_prob)

    return metrics, y_true, y_prob


def run_staged_training_v5(
    arm_name: str = 'Arm_H_Full_Mammogram_Top3_MultiScale',
    scales_used: Tuple[str, ...] = ('local', 'medium', 'large', 'context'),
    use_full_image: bool = True,
    use_calc_patch: bool = False,
    k: int = 3,
    aggregation_mode: str = 'attention_weighted',
    batch_size: int = 16,
    stage1_epochs: int = 8,
    stage2_epochs: int = 3,
    stage1_lr: float = 3e-4,
    stage2_lr_backbone: float = 1e-5,
    stage2_lr_head: float = 1e-4,
    weight_decay: float = 1e-3,
    patience: int = 3,
    seed: int = 42,
    save_checkpoint: bool = True,
    checkpoint_name: str = 'vit_v5_best.pth'
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n==================================================")
    logger.info(f"STARTING V5 TRAINING: {arm_name}")
    logger.info(f"Scales: {scales_used} | Full Image: {use_full_image} | Calc: {use_calc_patch} | K: {k}")
    logger.info(f"==================================================")

    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')

    train_loader, val_loader = get_v5_dataloaders(
        train_df=train_df,
        val_df=val_df,
        batch_size=batch_size,
        k=k,
        use_full_image=use_full_image,
        use_calc_enhanced=use_calc_patch,
        scales=scales_used
    )

    # Model instantiation
    model = MultiScaleViTV5(
        num_classes=2,
        use_full_image=use_full_image,
        use_calc_patch=use_calc_patch,
        scales_used=scales_used,
        aggregation_mode=aggregation_mode,
        k=k,
        pretrained=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())

    # Class-balanced cross entropy (Benign: 1.0, Malignant: 1.56)
    class_weights = torch.tensor([1.0, 1.56], device=device).float()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler('cuda')

    best_val_macro_f1 = -1.0
    best_val_metrics = {}
    best_stage = "None"
    best_epoch = 0
    best_state_dict = None
    stage1_history = []
    stage2_history = []

    # =========================================================================
    # STAGE 1: Train Fusion Head & Aggregators (ViT Backbone Frozen)
    # =========================================================================
    logger.info("\n>>> STAGE 1: Fusion Head & Candidate Aggregator Training (Backbone Frozen) <<<")
    for param in model.backbone.parameters():
        param.requires_grad = False

    head_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params_stage1 = sum(p.numel() for p in head_params)
    logger.info(f"Total Parameters: {total_params:,} | Stage 1 Trainable Parameters: {trainable_params_stage1:,}")

    optimizer = optim.AdamW(head_params, lr=stage1_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs, eta_min=1e-6)

    epochs_without_improvement = 0

    for ep in range(1, stage1_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            targets = batch['label'].to(device)
            full_img = batch['full_image'].to(device) if use_full_image else None
            calc_img = batch['calc_enhanced'].to(device) if use_calc_patch and 'calc_enhanced' in batch else None
            cand_crops = {s: batch[s].to(device) for s in scales_used}

            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(cand_crops, full_image_tensor=full_img, calc_patch_tensor=calc_img)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(1, n_batches)

        val_metrics, _, _ = evaluate_loader_v5(
            model, val_loader, device, scales_used, use_full_image, use_calc_patch
        )
        val_f1 = val_metrics['macro_f1']
        val_acc = val_metrics['accuracy']
        val_auc = val_metrics['roc_auc']
        val_sens = val_metrics['sensitivity_malignant_recall']

        logger.info(f"  Stage 1 Ep {ep:02d}/{stage1_epochs:02d} -> Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}% | ROC-AUC: {val_auc:.4f} | Sens: {val_sens*100:.2f}%")

        stage1_history.append({'epoch': ep, 'train_loss': avg_train_loss, 'metrics': val_metrics})

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_val_metrics = val_metrics
            best_stage = "Stage 1"
            best_epoch = ep
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"  Early stopping Stage 1 at epoch {ep} (patience={patience}).")
                break

    # Load best weights before Stage 2
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # =========================================================================
    # STAGE 2: Fine-Tuning Top ViT Blocks
    # =========================================================================
    if stage2_epochs > 0:
        logger.info("\n>>> STAGE 2: Fine-Tuning Top 2 ViT Transformer Blocks <<<")
        # Unfreeze top 2 Transformer blocks and norm
        for block in model.backbone.blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.backbone.norm.parameters():
            param.requires_grad = True

        backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if not n.startswith('backbone') and p.requires_grad]
        trainable_params_stage2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Stage 2 Trainable Parameters: {trainable_params_stage2:,}")

        optimizer_s2 = optim.AdamW([
            {'params': backbone_params, 'lr': stage2_lr_backbone, 'weight_decay': 1e-4},
            {'params': head_params, 'lr': stage2_lr_head, 'weight_decay': weight_decay}
        ])
        scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=stage2_epochs, eta_min=1e-7)

        epochs_without_improvement = 0

        for ep in range(1, stage2_epochs + 1):
            model.train()
            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                targets = batch['label'].to(device)
                full_img = batch['full_image'].to(device) if use_full_image else None
                calc_img = batch['calc_enhanced'].to(device) if use_calc_patch and 'calc_enhanced' in batch else None
                cand_crops = {s: batch[s].to(device) for s in scales_used}

                optimizer_s2.zero_grad()
                with autocast('cuda'):
                    logits = model(cand_crops, full_image_tensor=full_img, calc_patch_tensor=calc_img)
                    loss = criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer_s2)
                scaler.update()

                train_loss += loss.item()
                n_batches += 1

            scheduler_s2.step()
            avg_train_loss = train_loss / max(1, n_batches)

            val_metrics, _, _ = evaluate_loader_v5(
                model, val_loader, device, scales_used, use_full_image, use_calc_patch
            )
            val_f1 = val_metrics['macro_f1']
            val_acc = val_metrics['accuracy']
            val_auc = val_metrics['roc_auc']
            val_sens = val_metrics['sensitivity_malignant_recall']

            logger.info(f"  Stage 2 Ep {ep:02d}/{stage2_epochs:02d} -> Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}% | ROC-AUC: {val_auc:.4f} | Sens: {val_sens*100:.2f}%")

            stage2_history.append({'epoch': ep, 'train_loss': avg_train_loss, 'metrics': val_metrics})

            if val_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_f1
                best_val_metrics = val_metrics
                best_stage = "Stage 2"
                best_epoch = ep
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"  Early stopping Stage 2 at epoch {ep} (patience={patience}).")
                    break

    # Save Checkpoint if requested
    saved_path = ""
    if save_checkpoint and best_state_dict is not None:
        chk_dir = 'models/v5/classification'
        os.makedirs(chk_dir, exist_ok=True)
        onedrive_chk_dir = os.path.join(ONEDRIVE_ROOT, chk_dir)
        os.makedirs(onedrive_chk_dir, exist_ok=True)

        save_p = os.path.join(chk_dir, checkpoint_name)
        saved_path = save_p
        payload = {
            'arm_name': arm_name,
            'scales_used': scales_used,
            'use_full_image': use_full_image,
            'use_calc_patch': use_calc_patch,
            'k': k,
            'aggregation_mode': aggregation_mode,
            'total_params': total_params,
            'best_stage': best_stage,
            'best_epoch': best_epoch,
            'best_val_metrics': best_val_metrics,
            'model_state_dict': best_state_dict
        }
        torch.save(payload, save_p)
        try:
            torch.save(payload, os.path.join(onedrive_chk_dir, checkpoint_name))
        except Exception:
            pass

        logger.info(f"[SAVED] Checkpoint: {save_p} (Val Macro-F1: {best_val_macro_f1*100:.2f}%, Val Acc: {best_val_metrics['accuracy']*100:.2f}%)")

    return {
        'arm_name': arm_name,
        'total_params': total_params,
        'trainable_params_stage1': trainable_params_stage1,
        'trainable_params_stage2': trainable_params_stage2 if stage2_epochs > 0 else 0,
        'best_stage': best_stage,
        'best_epoch': best_epoch,
        'best_val_metrics': best_val_metrics,
        'checkpoint_path': saved_path,
        'stage1_history': stage1_history,
        'stage2_history': stage2_history
    }
