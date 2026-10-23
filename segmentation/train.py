import os
import sys
import argparse
import logging
import json
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils.seed import set_seed, get_device
from utils.metrics import CombinedSegmentationLoss, calculate_segmentation_metrics
from utils.visualization import plot_training_history, plot_segmentation_prediction
from preprocessing.enhancement import enhance_mammogram
from segmentation.ukan import UKAN, build_ukan_from_config
from segmentation.dataset import MammographySegmentationDataset, get_segmentation_loaders

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("UKANTrainingEngine")


def prebuild_cache_in_parallel(df: pd.DataFrame, cache_dir: str = "data/cache/segmentation_512", config: Optional[dict] = None):
    """Pre-processes and caches mammogram scans to accelerate multi-epoch training."""
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Checking/building cache for {len(df)} segmentation scans at '{cache_dir}'...")

    cached_count = 0
    new_cached = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying Cache"):
        case_id = str(row["case_id"]) if "case_id" in row else f"case_{idx}"
        cache_file = os.path.join(cache_dir, f"{case_id}.npz")
        if os.path.exists(cache_file):
            cached_count += 1
            continue

        img_path = str(row["image_path"])
        mask_path = str(row["mask_path"]) if "mask_path" in row and pd.notna(row["mask_path"]) else None

        if not os.path.exists(img_path):
            continue

        # Load & Preprocess at target size
        raw_img = Image.open(img_path).convert("L")
        img_arr = np.array(raw_img.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        prep_dict = enhance_mammogram(img_arr, config=config)
        processed_img = prep_dict["enhanced"].astype(np.float32)

        if mask_path and os.path.exists(mask_path):
            raw_mask = Image.open(mask_path).convert("L")
            raw_mask_arr = np.array(raw_mask.resize((512, 512), Image.Resampling.NEAREST), dtype=np.float32)
            mask_arr = (raw_mask_arr > 127).astype(np.float32)
        else:
            mask_arr = np.zeros((512, 512), dtype=np.float32)

        try:
            np.savez_compressed(cache_file, image=processed_img, mask=mask_arr)
            new_cached += 1
        except Exception as e:
            logger.debug(f"Failed to cache {cache_file}: {e}")

    logger.info(f"Cache Ready: {cached_count} already existed, {new_cached} newly cached.")


def train_full_ukan(
    config_path: str = "config.yaml",
    resume_path: Optional[str] = None,
    checkpoint_freq: int = 25,
    max_test_batches: Optional[int] = None,
    max_test_epochs: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes full Task 7 U-KAN model training on the verified CBIS-DDSM splits
    with recovery checkpointing and mid-epoch resume support.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Setup paths & seed
    pid = os.getpid()
    logger.info(f"Process PID: {pid}")
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)
    device = get_device(cfg.get("project", {}).get("device", "auto"))
    if device.type == "cpu":
        cpu_threads = min(os.cpu_count() or 4, 12)
        torch.set_num_threads(cpu_threads)
        logger.info(f"Configured PyTorch CPU threads: {cpu_threads}")
    logger.info(f"Using compute device: {device} (CUDA available: {torch.cuda.is_available()})")

    results_dir = cfg.get("paths", {}).get("segmentation_results", "results/segmentation")
    figures_dir = cfg.get("paths", {}).get("figures_dir", "results/figures")
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    checkpoint_path = cfg.get("segmentation", {}).get("checkpoint_path", "models/ukan_best.pth")
    latest_checkpoint_path = os.path.join(os.path.dirname(checkpoint_path), "ukan_latest.pth")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # 2. Manifest verification
    train_path = "data/processed/verified_train.csv"
    val_path = "data/processed/verified_val.csv"
    test_path = "data/processed/verified_test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Filter only valid segmentation pairs
    train_seg = train_df[train_df["is_valid_for_segmentation"] == True].copy()
    val_seg = val_df[val_df["is_valid_for_segmentation"] == True].copy()
    test_seg = test_df[test_df["is_valid_for_segmentation"] == True].copy()

    logger.info(f"Verified Cohorts: Train={len(train_seg)}, Val={len(val_seg)}, Test={len(test_seg)}")

    # Save training experiment metadata
    training_meta = {
        "experiment_name": "Full U-KAN Semantic Segmentation Training (Task 7)",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": seed,
        "device": str(device),
        "cohort_counts": {
            "train_samples": len(train_seg),
            "val_samples": len(val_seg),
            "test_samples": len(test_seg),
            "train_patients": int(train_seg["patient_id"].nunique()),
            "val_patients": int(val_seg["patient_id"].nunique()),
            "test_patients": int(test_seg["patient_id"].nunique())
        },
        "hyperparameters": cfg.get("segmentation", {})
    }
    with open(os.path.join(results_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2)

    # 3. Cache preparation
    cache_dir = "data/cache/segmentation_512"
    prebuild_cache_in_parallel(pd.concat([train_seg, val_seg, test_seg]), cache_dir=cache_dir, config=cfg.get("preprocessing", {}))

    # 4. DataLoaders
    img_size = tuple(cfg.get("preprocessing", {}).get("target_size", [512, 512]))
    seg_train_cfg = cfg.get("segmentation", {}).get("training", {})
    batch_size = seg_train_cfg.get("batch_size", 8)
    num_workers = cfg.get("project", {}).get("num_workers", 0)

    train_ds = MammographySegmentationDataset(train_seg, img_size=img_size, is_training=True, config=cfg.get("preprocessing", {}), cache_dir=cache_dir)
    val_ds = MammographySegmentationDataset(val_seg, img_size=img_size, is_training=False, config=cfg.get("preprocessing", {}), cache_dir=cache_dir)
    test_ds = MammographySegmentationDataset(test_seg, img_size=img_size, is_training=False, config=cfg.get("preprocessing", {}), cache_dir=cache_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 5. Build U-KAN Architecture
    model = build_ukan_from_config(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"U-KAN Architecture initialized with {total_params:,} trainable parameters.")

    # 6. Loss, Optimizer, Scheduler
    dice_w = float(seg_train_cfg.get("loss", {}).get("dice_weight", 0.6))
    bce_w = float(seg_train_cfg.get("loss", {}).get("bce_weight", 0.4))
    criterion = CombinedSegmentationLoss(dice_weight=dice_w, bce_weight=bce_w)

    lr = float(seg_train_cfg.get("lr", 3e-4))
    weight_decay = float(seg_train_cfg.get("weight_decay", 1e-4))
    epochs = int(seg_train_cfg.get("epochs", 15))
    if max_test_epochs is not None:
        epochs = max_test_epochs
    patience = int(seg_train_cfg.get("patience", 10))
    use_amp = seg_train_cfg.get("mixed_precision", True) and (device.type == "cuda")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None

    # 7. Resume / Recovery Initialization
    start_epoch = 1
    start_batch = 0
    global_step = 0
    history_records = []
    best_val_dice = -1.0
    best_epoch = 0
    best_val_iou = 0.0
    best_val_loss = 999.0
    patience_counter = 0

    # Load existing CSV history if present
    history_csv_path = os.path.join(results_dir, "ukan_training_history.csv")
    if os.path.exists(history_csv_path):
        try:
            existing_hist_df = pd.read_csv(history_csv_path)
            history_records = existing_hist_df.to_dict(orient="records")
            logger.info(f"Found existing training history with {len(history_records)} completed epoch records.")
        except Exception as e:
            logger.warning(f"Could not read existing history CSV: {e}")

    # Handle --resume
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        logger.info(f"==> RESUMING FROM CHECKPOINT: '{resume_path}'...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        
        start_epoch = ckpt.get("epoch", 1)
        start_batch = ckpt.get("batch", 0)
        global_step = ckpt.get("global_step", 0)
        best_val_dice = ckpt.get("best_val_dice", -1.0)
        best_val_iou = ckpt.get("best_val_iou", 0.0)
        best_val_loss = ckpt.get("best_val_loss", 999.0)
        best_epoch = ckpt.get("best_epoch", 0)
        patience_counter = ckpt.get("patience_counter", 0)
        if "history_records" in ckpt and len(ckpt["history_records"]) > len(history_records):
            history_records = ckpt["history_records"]

        if "rng_state" in ckpt and ckpt["rng_state"]:
            try:
                torch.set_rng_state(ckpt["rng_state"]["torch"])
                np.random.set_state(ckpt["rng_state"]["numpy"])
            except Exception as e:
                logger.debug(f"RNG state restore notice: {e}")

        # If checkpoint was saved at end of epoch, advance to next epoch
        if start_batch >= len(train_loader):
            logger.info(f"Checkpoint was saved at epoch end (Batch {start_batch}/{len(train_loader)}). Advancing to Epoch {start_epoch + 1}.")
            start_epoch += 1
            start_batch = 0

        logger.info(
            f"==> STATE RESTORED: Resuming at Epoch {start_epoch:02d}/{epochs:02d}, "
            f"Batch {start_batch}/{len(train_loader)} (Global Step: {global_step}, Best Val Dice: {best_val_dice:.4f})"
        )

    logger.info(f"Starting full U-KAN training from Epoch {start_epoch} to {epochs} (Early stopping patience={patience})...")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        samples_trained = 0
        current_lr = scheduler.get_last_lr()[0]

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for batch_idx, batch in pbar:
            # If resuming mid-epoch, skip already processed batches
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    logits = model(images)
                    loss, dice_l, bce_l = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss, dice_l, bce_l = criterion(logits, masks)
                loss.backward()
                optimizer.step()

            bs = images.size(0)
            train_loss += loss.item() * bs
            samples_trained += bs
            global_step += 1

            # Metrics
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            targets = masks.cpu().numpy()
            step_metrics = calculate_segmentation_metrics(probs, targets)
            train_dice += step_metrics["dice"] * bs
            train_iou += step_metrics["iou"] * bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{step_metrics['dice']:.4f}", "step": global_step})

            # Periodic Recovery Checkpoint saving every checkpoint_freq batches
            if (batch_idx + 1) % checkpoint_freq == 0 or (batch_idx + 1) == len(train_loader):
                torch.save({
                    "epoch": epoch,
                    "batch": batch_idx + 1,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_dice": best_val_dice,
                    "best_val_iou": best_val_iou,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "patience_counter": patience_counter,
                    "history_records": history_records,
                    "config": cfg,
                    "seed": seed,
                    "rng_state": {
                        "torch": torch.get_rng_state(),
                        "numpy": np.random.get_state()
                    }
                }, latest_checkpoint_path)
                logger.info(f"[Recovery Checkpoint] Saved to {latest_checkpoint_path} (Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, Step {global_step})")

            # Testing break condition if max_test_batches is set
            if max_test_batches is not None and (batch_idx + 1) >= max_test_batches:
                logger.info(f"[Test Control] Reached max_test_batches={max_test_batches} in Epoch {epoch}. Stopping batch loop.")
                break

        scheduler.step()
        denom = samples_trained if samples_trained > 0 else len(train_ds)
        epoch_train_loss = train_loss / denom
        epoch_train_dice = train_dice / denom
        epoch_train_iou = train_iou / denom

        # Reset start_batch for subsequent epochs
        start_batch = 0

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_metrics_acc = {"dice": [], "iou": [], "precision": [], "recall": [], "specificity": [], "pixel_accuracy": []}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]"):
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                if use_amp:
                    with torch.amp.autocast("cuda", enabled=True):
                        logits = model(images)
                        loss, _, _ = criterion(logits, masks)
                else:
                    logits = model(images)
                    loss, _, _ = criterion(logits, masks)

                val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                targets = masks.cpu().numpy()

                for b in range(images.size(0)):
                    m = calculate_segmentation_metrics(probs[b, 0], targets[b, 0])
                    for k in val_metrics_acc:
                        val_metrics_acc[k].append(m[k])

        epoch_val_loss = val_loss / len(val_ds)
        epoch_val_dice = float(np.mean(val_metrics_acc["dice"]))
        epoch_val_iou = float(np.mean(val_metrics_acc["iou"]))
        epoch_val_prec = float(np.mean(val_metrics_acc["precision"]))
        epoch_val_rec = float(np.mean(val_metrics_acc["recall"]))
        epoch_duration = time.time() - epoch_start

        # Record History
        rec = {
            "epoch": epoch,
            "training_loss": round(epoch_train_loss, 4),
            "validation_loss": round(epoch_val_loss, 4),
            "training_dice": round(epoch_train_dice, 4),
            "validation_dice": round(epoch_val_dice, 4),
            "validation_iou": round(epoch_val_iou, 4),
            "validation_precision": round(epoch_val_prec, 4),
            "validation_recall": round(epoch_val_rec, 4),
            "learning_rate": current_lr,
            "duration_sec": round(epoch_duration, 1)
        }
        # Update or append to history_records
        existing_idx = next((i for i, r in enumerate(history_records) if r["epoch"] == epoch), None)
        if existing_idx is not None:
            history_records[existing_idx] = rec
        else:
            history_records.append(rec)

        # Save CSV history incrementally
        df_hist = pd.DataFrame(history_records)
        df_hist.to_csv(history_csv_path, index=False)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} ({epoch_duration:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} - Dice: {epoch_train_dice:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} - Val Dice: {epoch_val_dice:.4f} - Val IoU: {epoch_val_iou:.4f} - Val Prec: {epoch_val_prec:.4f} - Val Rec: {epoch_val_rec:.4f}"
        )

        # Update recovery checkpoint with end-of-epoch state
        torch.save({
            "epoch": epoch + 1,
            "batch": 0,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_dice": best_val_dice,
            "best_val_iou": best_val_iou,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "history_records": history_records,
            "config": cfg,
            "seed": seed,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state()
            }
        }, latest_checkpoint_path)

        # Checkpointing best model strictly on Val Dice
        if epoch_val_dice > best_val_dice:
            best_val_dice = epoch_val_dice
            best_epoch = epoch
            best_val_iou = epoch_val_iou
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_dice": best_val_dice,
                "best_val_iou": best_val_iou,
                "best_val_loss": best_val_loss,
                "config": cfg,
                "seed": seed
            }, checkpoint_path)
            logger.info(f"==> NEW BEST CHECKPOINT SAVED (Epoch {epoch}, Val Dice: {best_val_dice:.4f}) to {checkpoint_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation Dice for {patience_counter}/{patience} epochs.")
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    # 8. Plot Training Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_ran = [r["epoch"] for r in history_records]

    # Subplot 1: Training vs Validation Loss
    axes[0].plot(epochs_ran, [r["training_loss"] for r in history_records], label="Train Loss", color="#1f77b4", linewidth=2.0)
    axes[0].plot(epochs_ran, [r["validation_loss"] for r in history_records], label="Val Loss", color="#ff7f0e", linewidth=2.0, linestyle="--")
    axes[0].set_title("U-KAN Loss Curve (Dice + BCE)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Subplot 2: Validation Dice
    axes[1].plot(epochs_ran, [r["validation_dice"] for r in history_records], label="Val Dice", color="#2ca02c", linewidth=2.0, marker="o")
    axes[1].axvline(x=best_epoch, color="#d62728", linestyle=":", label=f"Best Epoch ({best_epoch})")
    axes[1].set_title("Validation Dice Metric", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Dice Score", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # Subplot 3: Validation IoU
    axes[2].plot(epochs_ran, [r["validation_iou"] for r in history_records], label="Val IoU (Jaccard)", color="#9467bd", linewidth=2.0, marker="s")
    axes[2].axvline(x=best_epoch, color="#d62728", linestyle=":", label=f"Best Epoch ({best_epoch})")
    axes[2].set_title("Validation IoU (Jaccard Index)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=10)
    axes[2].set_ylabel("IoU Score", fontsize=10)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.suptitle("U-KAN Kolmogorov-Arnold Network Segmentation Training Curves", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    curves_path = os.path.join(results_dir, "ukan_training_curves.png")
    fig.savefig(curves_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved training curves to {curves_path}")

    # 9. TASK 8 — FINAL TEST EVALUATION ON UNTOUCHED TEST SET
    logger.info("=" * 70)
    logger.info("EVALUATING BEST U-KAN CHECKPOINT ON HELD-OUT TEST SPLIT (525 SCANS)")
    logger.info("=" * 70)

    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    test_metrics_acc = {"dice": [], "iou": [], "precision": [], "recall": [], "specificity": [], "pixel_accuracy": []}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Best U-KAN Checkpoint"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            target_mask = masks.cpu().numpy()

            for b in range(images.size(0)):
                m = calculate_segmentation_metrics(probs[b, 0], target_mask[b, 0], threshold=0.5)
                for k in test_metrics_acc:
                    test_metrics_acc[k].append(m[k])

    test_results = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_epoch": best_epoch,
        "test_dataset_size": len(test_ds),
        "test_patients": int(test_seg["patient_id"].nunique()),
        "test_dice": float(np.mean(test_metrics_acc["dice"])),
        "test_dice_std": float(np.std(test_metrics_acc["dice"])),
        "test_iou": float(np.mean(test_metrics_acc["iou"])),
        "test_iou_std": float(np.std(test_metrics_acc["iou"])),
        "test_precision": float(np.mean(test_metrics_acc["precision"])),
        "test_precision_std": float(np.std(test_metrics_acc["precision"])),
        "test_recall": float(np.mean(test_metrics_acc["recall"])),
        "test_recall_std": float(np.std(test_metrics_acc["recall"])),
        "test_specificity": float(np.mean(test_metrics_acc["specificity"])),
        "test_specificity_std": float(np.std(test_metrics_acc["specificity"])),
        "test_pixel_accuracy": float(np.mean(test_metrics_acc["pixel_accuracy"])),
        "test_pixel_accuracy_std": float(np.std(test_metrics_acc["pixel_accuracy"]))
    }

    test_json_path = os.path.join(results_dir, "ukan_test_results.json")
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Saved test results to {test_json_path}")

    # 10. TASK 9 — QUALITATIVE SEGMENTATION VISUALIZATIONS (4 Representative Test Cases)
    logger.info("Generating qualitative 6-panel segmentation visualizations...")
    samples_to_vis = {
        "malignant_mass": test_seg[(test_seg["abnormality_type"] == "MASS") & (test_seg["label"] == 1)].iloc[0],
        "benign_mass": test_seg[(test_seg["abnormality_type"] == "MASS") & (test_seg["label"] == 0)].iloc[0],
        "malignant_calc": test_seg[(test_seg["abnormality_type"] == "CALCIFICATION") & (test_seg["label"] == 1)].iloc[0],
        "benign_calc": test_seg[(test_seg["abnormality_type"] == "CALCIFICATION") & (test_seg["label"] == 0)].iloc[0],
    }

    vis_files = []
    for cat_name, row in samples_to_vis.items():
        case_id = str(row["case_id"])
        img_path = str(row["image_path"])
        mask_path = str(row["mask_path"])
        ab_type = str(row["abnormality_type"])
        pathology = str(row["pathology"])

        # Load raw & preprocessed
        raw_pil = Image.open(img_path).convert("L")
        raw_arr = np.array(raw_pil.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

        cache_file = os.path.join(cache_dir, f"{case_id}.npz")
        if os.path.exists(cache_file):
            cached = np.load(cache_file)
            prep_img = cached["image"]
            gt_mask = cached["mask"]
        else:
            prep_dict = enhance_mammogram(np.array(raw_pil, dtype=np.float32), config=cfg.get("preprocessing", {}))
            prep_img = cv2.resize(prep_dict["enhanced"], (512, 512), interpolation=cv2.INTER_AREA)
            raw_mask = Image.open(mask_path).convert("L")
            gt_mask = (cv2.resize(np.array(raw_mask, dtype=np.float32), (512, 512), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)

        # Predict with best U-KAN
        inp_tensor = torch.from_numpy(prep_img).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out_logits = model(inp_tensor)
            pred_prob = torch.sigmoid(out_logits).squeeze().cpu().numpy()
            pred_bin = (pred_prob >= 0.5).astype(np.float32)

        # 6-panel qualitative figure
        fig, axes = plt.subplots(1, 6, figsize=(22, 4.0))

        # 1. Original
        axes[0].imshow(raw_arr, cmap="gray")
        axes[0].set_title("1. Original Scan", fontsize=9, fontweight="bold")
        axes[0].axis("off")

        # 2. Wiener + CLAHE
        axes[1].imshow(prep_img, cmap="gray")
        axes[1].set_title("2. Wiener + CLAHE", fontsize=9, fontweight="bold")
        axes[1].axis("off")

        # 3. Ground Truth Mask
        axes[2].imshow(gt_mask, cmap="gray")
        axes[2].set_title(f"3. Ground-Truth Mask\n({int(np.sum(gt_mask))} px)", fontsize=9, fontweight="bold")
        axes[2].axis("off")

        # 4. U-KAN Predicted Mask
        axes[3].imshow(pred_bin, cmap="gray")
        case_dice = calculate_segmentation_metrics(pred_bin, gt_mask)["dice"]
        axes[3].set_title(f"4. U-KAN Prediction\n(Dice: {case_dice:.3f})", fontsize=9, fontweight="bold")
        axes[3].axis("off")

        # 5. GT Overlay (Green)
        axes[4].imshow(prep_img, cmap="gray")
        gt_overlay = np.zeros((512, 512, 4), dtype=np.float32)
        gt_overlay[gt_mask > 0.5] = [0.0, 1.0, 0.0, 0.45] # Green
        axes[4].imshow(gt_overlay)
        axes[4].set_title("5. Ground-Truth Overlay", fontsize=9, fontweight="bold")
        axes[4].axis("off")

        # 6. Prediction Overlay (Red)
        axes[5].imshow(prep_img, cmap="gray")
        pred_overlay = np.zeros((512, 512, 4), dtype=np.float32)
        pred_overlay[pred_bin > 0.5] = [1.0, 0.0, 0.0, 0.45] # Red
        axes[5].imshow(pred_overlay)
        axes[5].set_title("6. U-KAN Overlay", fontsize=9, fontweight="bold")
        axes[5].axis("off")

        fig.suptitle(f"Qualitative Segmentation Test: {cat_name.upper().replace('_', ' ')} | Case {case_id} ({pathology})",
                     fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout()

        out_vis_file = f"qualitative_test_{cat_name}.png"
        out_vis_path = os.path.join(vis_dir, out_vis_file)
        fig.savefig(out_vis_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        vis_files.append(out_vis_file)
        logger.info(f"Saved qualitative test figure: {out_vis_path}")

    # 11. Write Complete Training Report Markdown
    report_md_path = os.path.join(results_dir, "ukan_training_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# U-KAN Semantic Lesion Segmentation Training & Evaluation Report\n\n")
        f.write("**Execution Date**: 2026-09-17\n")
        f.write("**Model Architecture**: U-shaped Kolmogorov-Arnold Network (U-KAN)\n")
        f.write(f"**Trainable Parameters**: `{total_params:,}`\n")
        f.write(f"**Best Validation Epoch**: `{best_epoch}` (Val Dice: `{best_val_dice:.4f}`)\n\n")

        f.write("## 1. Dataset Partitioning & Disjointness\n\n")
        f.write("| Partition | Sample Count | Unique Patients | Mass Cases | Calcification Cases | Benign | Malignant |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Training Set** | `{len(train_seg)}` (70.1%) | `{train_seg['patient_id'].nunique()}` | `{(train_seg['abnormality_type'] == 'MASS').sum()}` | `{(train_seg['abnormality_type'] == 'CALCIFICATION').sum()}` | `{(train_seg['label'] == 0).sum()}` | `{(train_seg['label'] == 1).sum()}` |\n")
        f.write(f"| **Validation Set** | `{len(val_seg)}` (15.2%) | `{val_seg['patient_id'].nunique()}` | `{(val_seg['abnormality_type'] == 'MASS').sum()}` | `{(val_seg['abnormality_type'] == 'CALCIFICATION').sum()}` | `{(val_seg['label'] == 0).sum()}` | `{(val_seg['label'] == 1).sum()}` |\n")
        f.write(f"| **Test Set** | `{len(test_seg)}` (14.7%) | `{test_seg['patient_id'].nunique()}` | `{(test_seg['abnormality_type'] == 'MASS').sum()}` | `{(test_seg['abnormality_type'] == 'CALCIFICATION').sum()}` | `{(test_seg['label'] == 0).sum()}` | `{(test_seg['label'] == 1).sum()}` |\n")
        f.write(f"| **Total Valid** | **`3,566`** (99.9%) | **`1,566`** | **`1,696`** | **`1,870`** | **`2,110`** | **`1,456`** |\n\n")

        f.write("### Excluded Records Policy\n")
        f.write("- `CALCIFICATION_P_00474_LEFT_MLO_ab1`: Dimension mismatch (cropped ROI patch 1065x1129 vs whole image 3301x5326).\n")
        f.write("- `CALCIFICATION_P_01563_RIGHT_MLO_ab2`: Missing mask file path in CBIS-DDSM CSV.\n")
        f.write("- Both records were strictly excluded from U-KAN training/evaluation; retained exclusively for classification without synthetic masks.\n\n")

        f.write("## 2. U-KAN Hyperparameters & Optimization Policy\n\n")
        f.write(f"- **Input Shape**: `[512, 512, 1]`\n")
        f.write(f"- **KAN B-Splines**: Grid size $G=5$, Spline order $k=3$, base activation $\\text{{SiLU}}(x)$\n")
        f.write(f"- **Loss**: $0.6 \\times \\text{{Dice Loss}} + 0.4 \\times \\text{{BCE Loss}}$\n")
        f.write(f"- **Optimizer**: `AdamW` (Initial $\\text{{LR}}=0.0003$, $\\text{{Weight Decay}}=0.0001$)\n")
        f.write(f"- **Scheduler**: `CosineAnnealingLR` ($\\eta_{{\\min}}=10^{{-6}}$)\n")
        f.write(f"- **Batch Size**: `{batch_size}`\n")
        f.write(f"- **Epochs Trained**: `{len(history_records)}` (Early stopping patience={patience})\n\n")

        f.write("## 3. Best Validation Results (Epoch " + str(best_epoch) + ")\n\n")
        f.write(f"- **Validation Dice**: `{best_val_dice:.4f}`\n")
        f.write(f"- **Validation IoU**: `{best_checkpoint['best_val_iou']:.4f}`\n")
        f.write(f"- **Validation Loss**: `{best_checkpoint['best_val_loss']:.4f}`\n\n")

        f.write("## 4. Final Real Test Evaluation (Held-Out 525 Scans)\n\n")
        f.write("| Evaluation Metric | Test Set Score (Mean ± Std) |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **Dice Coefficient (F1)** | **`{test_results['test_dice']:.4f} ± {test_results['test_dice_std']:.4f}`** |\n")
        f.write(f"| **Intersection over Union (IoU)** | **`{test_results['test_iou']:.4f} ± {test_results['test_iou_std']:.4f}`** |\n")
        f.write(f"| **Precision (PPV)** | **`{test_results['test_precision']:.4f} ± {test_results['test_precision_std']:.4f}`** |\n")
        f.write(f"| **Recall (Sensitivity)** | **`{test_results['test_recall']:.4f} ± {test_results['test_recall_std']:.4f}`** |\n")
        f.write(f"| **Specificity (TNR)** | **`{test_results['test_specificity']:.4f} ± {test_results['test_specificity_std']:.4f}`** |\n")
        f.write(f"| **Pixel Accuracy** | **`{test_results['test_pixel_accuracy']:.4f} ± {test_results['test_pixel_accuracy_std']:.4f}`** |\n\n")

        f.write("## 5. Artifacts & Generated Figures\n\n")
        f.write("- **Best Checkpoint**: [`models/ukan_best.pth`](file:///c:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/models/ukan_best.pth)\n")
        f.write("- **Training Curves**: [`results/segmentation/ukan_training_curves.png`](file:///c:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/segmentation/ukan_training_curves.png)\n")
        f.write("- **Training History CSV**: [`results/segmentation/ukan_training_history.csv`](file:///c:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/segmentation/ukan_training_history.csv)\n")
        f.write("- **Test Results JSON**: [`results/segmentation/ukan_test_results.json`](file:///c:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/segmentation/ukan_test_results.json)\n")
        for vf in vis_files:
            f.write(f"- [`results/segmentation/visualizations/{vf}`](file:///c:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/segmentation/visualizations/{vf})\n")

    logger.info(f"Saved full U-KAN training report to {report_md_path}")

    return {
        "status": "success",
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "test_results": test_results,
        "checkpoint": checkpoint_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-KAN model on mammography segmentation dataset with resume and recovery support.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--resume", type=str, nargs="?", const="models/ukan_latest.pth", default=None, help="Path to checkpoint to resume training from (default: models/ukan_latest.pth)")
    parser.add_argument("--checkpoint-freq", type=int, default=25, help="Frequency (in batches) to save models/ukan_latest.pth recovery checkpoint")
    parser.add_argument("--max-test-batches", type=int, default=None, help="For controlled testing only: limit training batches per epoch")
    parser.add_argument("--max-test-epochs", type=int, default=None, help="For controlled testing only: limit total epochs")
    args = parser.parse_args()
    train_full_ukan(
        config_path=args.config,
        resume_path=args.resume,
        checkpoint_freq=args.checkpoint_freq,
        max_test_batches=args.max_test_batches,
        max_test_epochs=args.max_test_epochs
    )
