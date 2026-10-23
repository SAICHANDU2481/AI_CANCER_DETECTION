"""
================================================================================
V3-A Staged Training Pipeline: Multi-Scale Machine-ROI Vision Transformer
================================================================================
Executes:
- Stage 1: Head Warmup (Frozen ViT Backbone, LR=1e-4, Max 15 epochs, Patience 5)
- Stage 2: Joint Fine-Tuning (Unfreeze Top 2 Blocks, Backbone LR=1e-5, Head LR=1e-4, Max 30 epochs, Patience 8)
- Optimization: AdamW, Weighted CrossEntropy (Benign 0.8507, Malignant 1.2129), label smoothing 0.05
- Checkpoint Selection & Early Stopping: Validation Macro-F1
- Saves checkpoints to models/v3/checkpoints/
- Saves history and figures to results/v3/
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from utils.seed import set_seed
from classification.dataset_v3 import MammographyMultiScaleDatasetV3
from classification.model_v3 import MultiScaleMachineRoiViT

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("models/v3/logs/v3a_training.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("TrainV3A")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """Computes comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    bin_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5

    try:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(r_curve, p_curve)
    except Exception:
        pr_auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "binary_f1": float(bin_f1),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    """Evaluates model on validation cohort."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    records = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            targets = labels.cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets)

            for i in range(len(labels)):
                records.append({
                    "case_id": batch["case_id"][i],
                    "patient_id": batch["patient_id"][i],
                    "abnormality_type": batch["abnormality_type"][i],
                    "pathology": batch["pathology"][i],
                    "roi_source": batch["roi_source"][i],
                    "target": int(targets[i]),
                    "prediction": int(preds[i]),
                    "prob_malignant": float(probs[i])
                })

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds), np.array(all_probs))
    pred_df = pd.DataFrame(records)
    return avg_loss, metrics, pred_df


def train_v3a():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting V3-A Staged Training on device: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Load Data
    train_df = pd.read_csv("data/processed/verified_train.csv")
    val_df = pd.read_csv("data/processed/verified_val.csv")

    tr = train_df[train_df["is_valid_for_classification"] == True].reset_index(drop=True)
    vl = val_df[val_df["is_valid_for_classification"] == True].reset_index(drop=True)
    logger.info(f"Loaded {len(tr)} training scans ({tr['patient_id'].nunique()} pts) and {len(vl)} validation scans ({vl['patient_id'].nunique()} pts).")

    # Datasets with RAM Preloading
    logger.info("Preloading Training and Validation datasets into RAM...")
    t0 = time.time()
    train_ds = MammographyMultiScaleDatasetV3(tr, is_training=True, preload_to_ram=True)
    val_ds = MammographyMultiScaleDatasetV3(vl, is_training=False, preload_to_ram=True)
    logger.info(f"Datasets preloaded into RAM in {time.time() - t0:.1f}s.")

    micro_batch = 8
    grad_accum = 2  # Effective batch size = 16

    train_loader = DataLoader(train_ds, batch_size=micro_batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=micro_batch, shuffle=False)

    # Class Weights: Benign (0): 0.8507, Malignant (1): 1.2129
    class_weights = torch.tensor([0.8507, 1.2129], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    # Model Initialization
    model = MultiScaleMachineRoiViT(pretrained=True).to(device)

    # Directory Setup
    ckpt_dir = "models/v3/checkpoints"
    res_dir = "results/v3/classification"
    fig_dir = "results/v3/figures"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    history = []
    total_training_start = time.time()

    # =========================================================================
    # STAGE 1: HEAD WARMUP (Frozen ViT Backbone)
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STAGE 1: HEAD WARMUP (ViT Backbone FROZEN, LR = 1e-4, Max 15 Epochs, Patience 5)")
    logger.info("="*70)
    
    model.freeze_backbone()
    optimizer_s1 = torch.optim.AdamW(model.fusion_head.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s1, T_max=15, eta_min=1e-6)

    s1_best_f1 = -1.0
    s1_best_epoch = 0
    s1_patience_cnt = 0
    s1_best_ckpt_path = os.path.join(ckpt_dir, "vit_v3a_stage1_best.pth")

    for epoch in range(1, 16):
        ep_start = time.time()
        model.train()
        running_loss = 0.0
        optimizer_s1.zero_grad()

        for step_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels) / grad_accum

            scaler.scale(loss).backward()

            if (step_idx + 1) % grad_accum == 0 or (step_idx + 1) == len(train_loader):
                scaler.step(optimizer_s1)
                scaler.update()
                optimizer_s1.zero_grad()

            running_loss += loss.item() * grad_accum * len(labels)

        scheduler_s1.step()
        train_loss = running_loss / len(train_ds)
        val_loss, val_m, _ = evaluate(model, val_loader, criterion, device)
        ep_dur = time.time() - ep_start
        peak_vram = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        is_best = val_m["macro_f1"] > s1_best_f1
        if is_best:
            s1_best_f1 = val_m["macro_f1"]
            s1_best_epoch = epoch
            s1_patience_cnt = 0
            torch.save({
                "stage": 1,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_m,
                "val_loss": val_loss
            }, s1_best_ckpt_path)
            best_tag = "[BEST SAVED]"
        else:
            s1_patience_cnt += 1
            best_tag = f"Patience: {s1_patience_cnt}/5"

        logger.info(
            f"[Stage 1] Epoch {epoch:02d}/15 [{ep_dur:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_m['accuracy']:.4f} | Val Macro-F1: {val_m['macro_f1']:.4f} | "
            f"Val AUC: {val_m['roc_auc']:.4f} | Best S1 F1: {s1_best_f1:.4f} (Ep {s1_best_epoch}) | {best_tag}"
        )

        history.append({
            "stage": 1,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_m["accuracy"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_specificity": val_m["specificity"],
            "val_binary_f1": val_m["binary_f1"],
            "val_macro_f1": val_m["macro_f1"],
            "val_roc_auc": val_m["roc_auc"],
            "val_pr_auc": val_m["pr_auc"],
            "learning_rate": scheduler_s1.get_last_lr()[0],
            "epoch_duration_sec": ep_dur,
            "peak_vram_gb": peak_vram
        })

        if s1_patience_cnt >= 5:
            logger.info(f"Stage 1 early stopping triggered at Epoch {epoch} (Patience 5 reached).")
            break

    logger.info(f"STAGE 1 COMPLETE. Best Checkpoint: Epoch {s1_best_epoch} with Validation Macro-F1 = {s1_best_f1:.4f}")

    # =========================================================================
    # STAGE 2: JOINT FINE-TUNING (Unfreeze Top 2 Blocks)
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STAGE 2: JOINT FINE-TUNING (Top 2 Blocks Unfrozen, Backbone LR=1e-5, Head LR=1e-4, Max 30 Epochs)")
    logger.info("="*70)

    # Load best Stage 1 weights
    logger.info(f"Loading best Stage 1 checkpoint from '{s1_best_ckpt_path}'...")
    s1_ckpt = torch.load(s1_best_ckpt_path, map_location=device)
    model.load_state_dict(s1_ckpt["model_state_dict"])

    # Unfreeze top 2 transformer blocks + head
    model.unfreeze_top_blocks(num_blocks=2)

    param_groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-4, weight_decay=0.01)
    optimizer_s2 = torch.optim.AdamW(param_groups)
    scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=30, eta_min=1e-6)

    s2_best_f1 = -1.0
    s2_best_epoch = 0
    s2_patience_cnt = 0
    s2_best_ckpt_path = os.path.join(ckpt_dir, "vit_v3a_stage2_best.pth")
    overall_best_ckpt_path = os.path.join(ckpt_dir, "vit_v3a_best.pth")
    best_val_df = None
    best_val_metrics = None

    for epoch in range(1, 31):
        ep_start = time.time()
        model.train()
        running_loss = 0.0
        optimizer_s2.zero_grad()

        for step_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels) / grad_accum

            scaler.scale(loss).backward()

            if (step_idx + 1) % grad_accum == 0 or (step_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer_s2)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer_s2)
                scaler.update()
                optimizer_s2.zero_grad()

            running_loss += loss.item() * grad_accum * len(labels)

        scheduler_s2.step()
        train_loss = running_loss / len(train_ds)
        val_loss, val_m, pred_df = evaluate(model, val_loader, criterion, device)
        ep_dur = time.time() - ep_start
        peak_vram = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        is_best = val_m["macro_f1"] > s2_best_f1
        if is_best:
            s2_best_f1 = val_m["macro_f1"]
            s2_best_epoch = epoch
            s2_patience_cnt = 0
            best_val_df = pred_df
            best_val_metrics = val_m
            
            # Save Stage 2 Best & Overall Best
            torch.save({
                "stage": 2,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_m,
                "val_loss": val_loss
            }, s2_best_ckpt_path)
            torch.save({
                "stage": 2,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_m,
                "val_loss": val_loss
            }, overall_best_ckpt_path)
            best_tag = "[BEST SAVED]"
        else:
            s2_patience_cnt += 1
            best_tag = f"Patience: {s2_patience_cnt}/8"

        logger.info(
            f"[Stage 2] Epoch {epoch:02d}/30 [{ep_dur:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_m['accuracy']:.4f} | Val Macro-F1: {val_m['macro_f1']:.4f} | "
            f"Val AUC: {val_m['roc_auc']:.4f} | Best S2 F1: {s2_best_f1:.4f} (Ep {s2_best_epoch}) | {best_tag}"
        )

        history.append({
            "stage": 2,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_m["accuracy"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_specificity": val_m["specificity"],
            "val_binary_f1": val_m["binary_f1"],
            "val_macro_f1": val_m["macro_f1"],
            "val_roc_auc": val_m["roc_auc"],
            "val_pr_auc": val_m["pr_auc"],
            "learning_rate": scheduler_s2.get_last_lr()[0],
            "epoch_duration_sec": ep_dur,
            "peak_vram_gb": peak_vram
        })

        if s2_patience_cnt >= 8:
            logger.info(f"Stage 2 early stopping triggered at Epoch {epoch} (Patience 8 reached).")
            break

    total_training_time_min = (time.time() - total_training_start) / 60.0
    logger.info(f"V3-A STAGED TRAINING COMPLETED in {total_training_time_min:.2f} minutes.")

    # 4. Save History and Best Validation Predictions
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(res_dir, "v3a_training_history.csv"), index=False)
    
    if best_val_df is not None:
        best_val_df.to_csv(os.path.join(res_dir, "v3a_val_predictions.csv"), index=False)
    
    val_metrics_out = {
        "experiment": "V3-A Multi-Scale Machine-ROI ViT",
        "stage1_best_epoch": s1_best_epoch,
        "stage1_best_macro_f1": s1_best_f1,
        "stage2_best_epoch": s2_best_epoch,
        "stage2_best_macro_f1": s2_best_f1,
        "total_training_time_min": total_training_time_min,
        "best_validation_metrics": best_val_metrics
    }
    with open(os.path.join(res_dir, "v3a_val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics_out, f, indent=2)

    # 5. Generate Training Figures
    # Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df.index + 1, hist_df["train_loss"], label="Train Loss", color="#1d3557", linewidth=2)
    plt.plot(hist_df.index + 1, hist_df["val_loss"], label="Val Loss", color="#e63946", linewidth=2)
    plt.axvline(s1_best_epoch, color="#457b9d", linestyle="--", label=f"Stage 1 Best (Ep {s1_best_epoch})")
    plt.title("V3-A Staged Training & Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Total Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "v3a_loss_curve.png"), dpi=300)
    plt.close()

    # Macro-F1 Curve
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df.index + 1, hist_df["val_macro_f1"], label="Val Macro-F1", color="#2a9d8f", linewidth=2)
    plt.plot(hist_df.index + 1, hist_df["val_accuracy"], label="Val Accuracy", color="#f4a261", linestyle="--", linewidth=1.5)
    plt.plot(hist_df.index + 1, hist_df["val_roc_auc"], label="Val ROC-AUC", color="#e76f51", linestyle=":", linewidth=1.5)
    plt.title("V3-A Validation Metrics Across Staged Training", fontsize=12, fontweight="bold")
    plt.xlabel("Total Epoch", fontsize=11)
    plt.ylabel("Metric Score", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "v3a_macro_f1_curve.png"), dpi=300)
    plt.close()

    # Confusion Matrix Plot
    if best_val_metrics is not None:
        cm = np.array([
            [best_val_metrics["tn"], best_val_metrics["fp"]],
            [best_val_metrics["fn"], best_val_metrics["tp"]]
        ])
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.title(f"V3-A Validation Confusion Matrix\n(Macro-F1: {s2_best_f1:.4f}, Acc: {best_val_metrics['accuracy']:.4f})", fontsize=11, fontweight="bold")
        plt.colorbar()
        classes = ["Benign (0)", "Malignant (1)"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=10)
        plt.yticks(tick_marks, classes, fontsize=10)
        
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=color, fontsize=12, fontweight="bold")
        plt.ylabel("True Diagnosis", fontsize=11)
        plt.xlabel("Predicted Diagnosis", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "v3a_confusion_matrix.png"), dpi=300)
        plt.close()

    logger.info("Training history, metrics, and figures successfully saved.")
    return val_metrics_out


if __name__ == "__main__":
    train_v3a()
