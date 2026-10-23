import os
import sys
import time
import argparse
import logging
import json
from typing import Dict, Any, Optional
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from tqdm import tqdm

# Ensure workspace in sys.path
sys.path.insert(0, os.path.abspath("."))

from utils.seed import set_seed, get_device
from classification.vit_model import build_vit_from_config
from classification.dataset import get_classification_loaders

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ViTTrain")


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes all standard classification metrics including Macro-F1, Specificity, ROC-AUC, and PR-AUC.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    binary_f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = float(tn / (tn + fp + 1e-7))
    else:
        spec = 0.0

    roc_auc = 0.5
    pr_auc = 0.5
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = float(np.mean(y_true))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "binary_f1": binary_f1,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


def plot_vit_training_curves(history_df: pd.DataFrame, save_path: str = "results/classification/vit_training_curves.png"):
    """
    Plots training loss, validation loss, validation Macro-F1, Accuracy, ROC-AUC, and PR-AUC.
    """
    if len(history_df) == 0:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#0f172a")

    epochs = history_df["epoch"]

    # 1. Loss Curve
    ax = axes[0, 0]
    ax.set_facecolor("#1e293b")
    ax.plot(epochs, history_df["train_loss"], label="Train Loss", color="#38bdf8", lw=2, marker="o", markersize=4)
    ax.plot(epochs, history_df["val_loss"], label="Val Loss", color="#f43f5e", lw=2, marker="s", markersize=4)
    ax.set_title("Loss Progression", color="#f8fafc", fontweight="bold", fontsize=12)
    ax.set_xlabel("Epoch", color="#94a3b8")
    ax.set_ylabel("Weighted Cross-Entropy Loss", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, alpha=0.15, color="#64748b")
    ax.legend(facecolor="#334155", edgecolor="none", labelcolor="#f8fafc")

    # 2. Macro-F1 & Binary F1
    ax = axes[0, 1]
    ax.set_facecolor("#1e293b")
    ax.plot(epochs, history_df["val_macro_f1"], label="Val Macro-F1", color="#10b981", lw=2.5, marker="o", markersize=5)
    ax.plot(epochs, history_df["val_binary_f1"], label="Val Binary-F1 (Malignant)", color="#a855f7", lw=1.5, ls="--", marker="^", markersize=4)
    ax.set_title("Validation F1 Score (Model Selection Metric)", color="#f8fafc", fontweight="bold", fontsize=12)
    ax.set_xlabel("Epoch", color="#94a3b8")
    ax.set_ylabel("F1 Score", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, alpha=0.15, color="#64748b")
    ax.legend(facecolor="#334155", edgecolor="none", labelcolor="#f8fafc")

    # 3. Accuracy, Sensitivity, Specificity
    ax = axes[1, 0]
    ax.set_facecolor("#1e293b")
    ax.plot(epochs, history_df["val_accuracy"], label="Accuracy", color="#38bdf8", lw=2, marker="o")
    ax.plot(epochs, history_df["val_recall"], label="Sensitivity / Recall", color="#f59e0b", lw=2, marker="^")
    ax.plot(epochs, history_df["val_specificity"], label="Specificity", color="#06b6d4", lw=2, marker="v")
    ax.set_title("Validation Accuracy, Sensitivity & Specificity", color="#f8fafc", fontweight="bold", fontsize=12)
    ax.set_xlabel("Epoch", color="#94a3b8")
    ax.set_ylabel("Score", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, alpha=0.15, color="#64748b")
    ax.legend(facecolor="#334155", edgecolor="none", labelcolor="#f8fafc")

    # 4. ROC-AUC & PR-AUC
    ax = axes[1, 1]
    ax.set_facecolor("#1e293b")
    ax.plot(epochs, history_df["val_roc_auc"], label="Val ROC-AUC", color="#e11d48", lw=2.5, marker="o")
    ax.plot(epochs, history_df["val_pr_auc"], label="Val PR-AUC", color="#8b5cf6", lw=2, marker="s")
    ax.set_title("Validation Discriminative Capacity (AUC)", color="#f8fafc", fontweight="bold", fontsize=12)
    ax.set_xlabel("Epoch", color="#94a3b8")
    ax.set_ylabel("Area Under Curve", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, alpha=0.15, color="#64748b")
    ax.legend(facecolor="#334155", edgecolor="none", labelcolor="#f8fafc")

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)


def train_vit(
    config_path: str = "config.yaml",
    resume: bool = False
) -> Dict[str, Any]:
    """
    Executes Vision Transformer fine-tuning for Benign vs Malignant breast cancer classification.
    Features intra-epoch recovery checkpointing (every 25 batches) and strict state preservation.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Seed & Device
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)
    device = get_device(cfg.get("project", {}).get("device", "auto"))

    # 2. DataLoaders & Class Weights (Computed strictly from TRAIN split)
    train_loader, val_loader, _, class_weights = get_classification_loaders(config_path)
    if train_loader is None or len(train_loader) == 0:
        raise RuntimeError("Train dataset empty or unconfigured.")

    # 3. Model Setup
    model = build_vit_from_config(cfg).to(device)

    cls_train_cfg = cfg.get("classification", {}).get("training", {})
    epochs = cls_train_cfg.get("epochs", 40)
    lr = float(cls_train_cfg.get("lr", 1e-4))
    weight_decay = float(cls_train_cfg.get("weight_decay", 0.01))
    patience = cls_train_cfg.get("patience", 8)
    label_smoothing = float(cls_train_cfg.get("label_smoothing", 0.05))
    use_amp = cls_train_cfg.get("mixed_precision", True) and (device.type == "cuda")

    # 4. Criterion, Optimizer & Scheduler
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp)

    best_checkpoint_path = cfg.get("classification", {}).get("checkpoint_path", "models/vit_best.pth")
    latest_checkpoint_path = "models/vit_latest.pth"
    os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
    os.makedirs("results/classification", exist_ok=True)

    # Attach file handler to logger if not already present
    file_log_path = "results/classification/vit_train.log"
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(file_log_path, mode="a")
        fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    csv_history_path = "results/classification/vit_training_history.csv"
    json_history_path = "results/classification/vit_training_history.json"
    curves_plot_path = "results/classification/vit_training_curves.png"

    start_epoch = 1
    start_batch = 0
    global_step = 0
    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history_records = []

    # 5. Resume from Checkpoint if requested
    if resume and os.path.exists(latest_checkpoint_path):
        logger.info(f"Resuming ViT training from recovery checkpoint: {latest_checkpoint_path}")
        ckpt = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        
        saved_epoch = ckpt.get("epoch", 1)
        saved_batch = ckpt.get("batch", 0)
        global_step = ckpt.get("global_step", 0)
        best_val_macro_f1 = ckpt.get("best_val_macro_f1", -1.0)
        best_epoch = ckpt.get("best_epoch", 0)
        patience_counter = ckpt.get("patience_counter", 0)

        if "rng_state" in ckpt:
            torch.set_rng_state(ckpt["rng_state"].cpu())

        if os.path.exists(csv_history_path):
            df_prev = pd.read_csv(csv_history_path)
            history_records = df_prev.to_dict("records")

        completed_epochs = [r.get("epoch") for r in history_records if isinstance(r, dict)]

        # If saved_batch is at end of epoch, check whether validation pass was recorded
        if saved_batch >= len(train_loader) - 1:
            if saved_epoch not in completed_epochs:
                start_epoch = saved_epoch
                start_batch = len(train_loader) # Skips training, runs validation pass for saved_epoch
                logger.info(f"Epoch {saved_epoch} training batches (157/157) already complete. Resuming directly at Epoch {saved_epoch} validation pass.")
            else:
                start_epoch = saved_epoch + 1
                start_batch = 0
        else:
            start_epoch = saved_epoch
            start_batch = saved_batch + 1

        logger.info(f"Resumed at Epoch {start_epoch}, Batch {start_batch}/{len(train_loader)}, Global Step {global_step} | Best Val Macro-F1: {best_val_macro_f1:.4f} (Epoch {best_epoch})")

    logger.info("=" * 70)
    logger.info("STARTING VISION TRANSFORMER (ViT) TRAINING")
    logger.info(f"Model: {cfg.get('classification', {}).get('model_name', 'vit_base_patch16_224')} | Target Device: {device}")
    logger.info(f"Training Samples: {len(train_loader.dataset)} ({len(train_loader)} batches/epoch, Batch Size: 16)")
    logger.info(f"Validation Samples: {len(val_loader.dataset)} ({len(val_loader)} batches/epoch)")
    logger.info(f"Max Epochs: {epochs} | Early Stopping Patience: {patience} epochs on Val Macro-F1")
    logger.info(f"Class Weights: Benign (0) = {class_weights[0]:.4f}, Malignant (1) = {class_weights[1]:.4f}")
    logger.info("=" * 70)

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        current_lr = optimizer.param_groups[0]["lr"]
        for batch_idx, batch in enumerate(train_loader):
            # Skip batches if resuming within epoch
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                raise FloatingPointError(f"NaN/Inf loss encountered at epoch {epoch}, batch {batch_idx}!")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.cpu().numpy())

            # Save recovery checkpoint and log every 25 batches
            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(train_loader):
                logger.info(f"Epoch [{epoch:02d}/{epochs:02d}] Batch [{batch_idx+1:03d}/{len(train_loader):03d}] - Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
                torch.save({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_epoch": best_epoch,
                    "patience_counter": patience_counter,
                    "rng_state": torch.get_rng_state(),
                    "config": cfg,
                    "seed": seed
                }, latest_checkpoint_path)

        # Reset start_batch for subsequent epochs
        start_batch = 0
        scheduler.step()
        epoch_duration = time.time() - epoch_start_time
        epoch_train_loss = train_loss / len(train_loader.dataset) if len(train_targets) > 0 else 0.0
        train_acc = float(accuracy_score(train_targets, train_preds)) if len(train_targets) > 0 else 0.0

        # 6. Validation Loop (Evaluated strictly on Validation partition)
        model.eval()
        val_loss = 0.0
        val_preds, val_targets, val_probs = [], [], []

        with torch.no_grad():
            for val_idx, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                with autocast(enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                val_probs.extend(probs)
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_metrics = compute_comprehensive_metrics(val_targets, val_preds, val_probs)

        record = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "val_loss": round(epoch_val_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_precision": round(val_metrics["precision"], 4),
            "val_recall": round(val_metrics["recall"], 4),
            "val_specificity": round(val_metrics["specificity"], 4),
            "val_binary_f1": round(val_metrics["binary_f1"], 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "val_roc_auc": round(val_metrics["roc_auc"], 4),
            "val_pr_auc": round(val_metrics["pr_auc"], 4),
            "learning_rate": round(current_lr, 6),
            "duration_sec": round(epoch_duration, 1)
        }
        history_records.append(record)

        # Save continuous CSV & JSON history
        df_history = pd.DataFrame(history_records)
        df_history.to_csv(csv_history_path, index=False)
        with open(json_history_path, "w") as f:
            json.dump(history_records, f, indent=2)

        plot_vit_training_curves(df_history, curves_plot_path)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, Macro-F1: {val_metrics['macro_f1']:.4f}, "
            f"ROC-AUC: {val_metrics['roc_auc']:.4f}, PR-AUC: {val_metrics['pr_auc']:.4f} ({epoch_duration:.1f}s)"
        )

        # 7. Checkpoint Savings (Best & Latest Recovery)
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_macro_f1": best_val_macro_f1,
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_metrics": val_metrics,
                "config": cfg,
                "seed": seed
            }, best_checkpoint_path)
            logger.info(f"==> NEW BEST CHECKPOINT SAVED (Epoch {epoch}, Val Macro-F1: {best_val_macro_f1:.4f}) to {best_checkpoint_path}")
        else:
            patience_counter += 1

        # Always update recovery checkpoint at epoch end
        torch.save({
            "epoch": epoch,
            "batch": len(train_loader) - 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_macro_f1": best_val_macro_f1,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "rng_state": torch.get_rng_state(),
            "config": cfg,
            "seed": seed
        }, latest_checkpoint_path)

        # 8. Early Stopping Check
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch} (No improvement in {patience} epochs since Epoch {best_epoch}).")
            break

    logger.info("=" * 70)
    logger.info("VISION TRANSFORMER TRAINING COMPLETE")
    logger.info(f"Total Epochs Processed: {len(history_records)}")
    logger.info(f"Best Validation Epoch:  Epoch {best_epoch}")
    logger.info(f"Best Validation Macro-F1: {best_val_macro_f1:.4f}")
    logger.info(f"Best Model Checkpoint:  {best_checkpoint_path}")
    logger.info(f"Training History CSV:   {csv_history_path}")
    logger.info("=" * 70)

    return {
        "status": "completed",
        "total_epochs": len(history_records),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_checkpoint": best_checkpoint_path,
        "history_csv": csv_history_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision Transformer classifier on CBIS-DDSM lesion ROIs.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from models/vit_latest.pth")
    args = parser.parse_args()
    train_vit(config_path=args.config, resume=args.resume)
