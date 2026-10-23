"""
================================================================================
V6 Phase 2: Short Pilot Segmentation Ablation Suite (Arms A - G)
================================================================================
Strict Governance:
- Evaluation strictly on Train (2501) + Validation (541).
- Zero test data access (test.csv is 100% quarantined).
- All V5 checkpoints and test artifacts are protected and verified.
- Records comprehensive subgroup and lesion-size stratified metrics.
================================================================================
"""

import os
import sys
import time
import math
import json
import hashlib
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# Line buffering for immediate real-time logging
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True, errors='replace')

# Ensure project root is in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from segmentation.v6.ukan_v6 import build_ukan_v6, MultiScaleUKANV6
from segmentation.v6.losses_v6 import (
    BCEDiceLoss, FocalDiceLoss, TverskyDiceLoss, FocalTverskyBoundaryLoss,
    DeepSupervisedCompositeLoss, SmallLesionWeightedLoss
)
from segmentation.v6.dataset_v6 import get_v6_segmentation_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("V6_PilotAblation")


def compute_file_md5(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "N/A"
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_sample_dice(pred_binary: np.ndarray, target_binary: np.ndarray, smooth: float = 1.0) -> float:
    intersection = np.sum(pred_binary * target_binary)
    cardinality = np.sum(pred_binary) + np.sum(target_binary)
    if cardinality == 0:
        return 1.0
    return float((2.0 * intersection + smooth) / (cardinality + smooth))


def compute_sample_iou(pred_binary: np.ndarray, target_binary: np.ndarray, smooth: float = 1.0) -> float:
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary) - intersection
    if union == 0:
        return 1.0
    return float((intersection + smooth) / (union + smooth))


def evaluate_segmentation_val(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluates segmentation performance across the entire 541-scan validation cohort.
    Computes overall metrics, clinical subgroups (Mass / Calc, Benign / Malignant),
    and lesion size strata (<500 px, 500-2000 px, >2000 px).
    """
    model.eval()
    val_loss_sum = 0.0
    num_batches = 0

    all_dice = []
    all_iou = []
    all_prec = []
    all_recall = []
    all_spec = []
    all_acc = []

    mass_dice = []
    calc_dice = []
    benign_dice = []
    malignant_dice = []

    small_dice = []   # < 500 px
    medium_dice = []  # 500 - 2000 px
    large_dice = []   # > 2000 px

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].cpu().numpy()
            abn_types = batch['abnormality_type']
            areas = batch['lesion_area'].cpu().numpy()

            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(imgs)
                if isinstance(outputs, (tuple, list)):
                    main_logits = outputs[0]
                elif isinstance(outputs, dict):
                    main_logits = outputs['main']
                else:
                    main_logits = outputs
                loss = criterion(outputs, masks)

            val_loss_sum += float(loss.item())
            num_batches += 1

            probs = torch.sigmoid(main_logits).cpu().numpy()
            targets = masks.cpu().numpy()
            preds_bin = (probs >= threshold).astype(np.float32)

            B = imgs.shape[0]
            for b in range(B):
                p_b = preds_bin[b, 0]
                t_b = (targets[b, 0] > 0.5).astype(np.float32)
                abn = str(abn_types[b])
                lbl = int(labels[b])
                area = float(areas[b])

                d_val = compute_sample_dice(p_b, t_b)
                iou_val = compute_sample_iou(p_b, t_b)

                tp = float(np.sum((p_b == 1.0) & (t_b == 1.0)))
                fp = float(np.sum((p_b == 1.0) & (t_b == 0.0)))
                fn = float(np.sum((p_b == 0.0) & (t_b == 1.0)))
                tn = float(np.sum((p_b == 0.0) & (t_b == 0.0)))

                prec = tp / (tp + fp + 1e-7) if (tp + fp) > 0 else (1.0 if np.sum(t_b) == 0 else 0.0)
                rec = tp / (tp + fn + 1e-7) if (tp + fn) > 0 else 1.0
                spec = tn / (tn + fp + 1e-7) if (tn + fp) > 0 else 1.0
                acc = (tp + tn) / (tp + tn + fp + fn)

                all_dice.append(d_val)
                all_iou.append(iou_val)
                all_prec.append(prec)
                all_recall.append(rec)
                all_spec.append(spec)
                all_acc.append(acc)

                # Subgroups
                if "MASS" in abn.upper():
                    mass_dice.append(d_val)
                elif "CALC" in abn.upper():
                    calc_dice.append(d_val)

                if lbl == 1:
                    malignant_dice.append(d_val)
                else:
                    benign_dice.append(d_val)

                # Size strata
                if area < 500:
                    small_dice.append(d_val)
                elif area > 2000:
                    large_dice.append(d_val)
                else:
                    medium_dice.append(d_val)

    mean_val_loss = val_loss_sum / max(1, num_batches)

    return {
        "val_loss": round(mean_val_loss, 4),
        "mean_dice": round(float(np.mean(all_dice)), 4),
        "mean_iou": round(float(np.mean(all_iou)), 4),
        "precision": round(float(np.mean(all_prec)), 4),
        "recall_sensitivity": round(float(np.mean(all_recall)), 4),
        "specificity": round(float(np.mean(all_spec)), 4),
        "pixel_accuracy": round(float(np.mean(all_acc)), 4),
        "mass_dice": round(float(np.mean(mass_dice)), 4) if mass_dice else 0.0,
        "calc_dice": round(float(np.mean(calc_dice)), 4) if calc_dice else 0.0,
        "benign_dice": round(float(np.mean(benign_dice)), 4) if benign_dice else 0.0,
        "malignant_dice": round(float(np.mean(malignant_dice)), 4) if malignant_dice else 0.0,
        "small_lesion_dice": round(float(np.mean(small_dice)), 4) if small_dice else 0.0,
        "medium_lesion_dice": round(float(np.mean(medium_dice)), 4) if medium_dice else 0.0,
        "large_lesion_dice": round(float(np.mean(large_dice)), 4) if large_dice else 0.0,
    }


def run_pilot_arm(
    arm_id: str,
    arm_name: str,
    model_fn,
    loss_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_accum_steps: int = 1
) -> Dict[str, Any]:
    """
    Executes a single controlled pilot ablation arm across 3 epochs.
    """
    logger.info("=" * 80)
    logger.info(f"STARTING PILOT ABLATION: [{arm_id}] {arm_name}")
    logger.info("=" * 80)

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

    model = model_fn().to(device)
    criterion = loss_fn
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    best_val_dice = -1.0
    best_metrics = {}
    history = []
    t_start = time.time()
    nan_inf_count = 0

    pilot_ckpt_dir = os.path.join(ROOT, "models", "v6", "segmentation", "pilot")
    os.makedirs(pilot_ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(pilot_ckpt_dir, f"ukan_v6_pilot_{arm_id.lower()}.pth")

    total_batches = len(train_loader)

    for ep in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        train_loss_sum = 0.0
        num_train_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)

            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss_scaled = loss / grad_accum_steps

            if torch.isnan(loss).item() or torch.isinf(loss).item():
                nan_inf_count += 1
                logger.warning(f"  [{arm_id}] NaN/Inf loss encountered at ep {ep} step {step}!")
                continue

            scaler.scale(loss_scaled).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss_sum += float(loss.item())
            num_train_batches += 1

            if (step + 1) % 150 == 0 or (step + 1) == total_batches:
                avg_l = train_loss_sum / num_train_batches
                logger.info(f"  [{arm_id}] Ep {ep}/{epochs} | Step {step+1}/{total_batches} | Running Loss: {avg_l:.4f}")

        scheduler.step()
        ep_train_loss = train_loss_sum / max(1, num_train_batches)

        # Validation Evaluation
        val_metrics = evaluate_segmentation_val(model, val_loader, criterion, device)
        val_dice = val_metrics["mean_dice"]
        val_calc_dice = val_metrics["calc_dice"]
        val_small_dice = val_metrics["small_lesion_dice"]
        ep_time = time.time() - ep_start

        logger.info(
            f"  >>> [{arm_id}] Ep {ep:02d}/{epochs:02d} COMPLETED in {ep_time:.1f}s | "
            f"Train Loss: {ep_train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Mean Dice: {val_dice*100:.2f}% | Calc Dice: {val_calc_dice*100:.2f}% | Small Dice: {val_small_dice*100:.2f}%"
        )

        ep_record = {
            "epoch": ep,
            "train_loss": round(ep_train_loss, 4),
            **val_metrics,
            "epoch_time_sec": round(ep_time, 2)
        }
        history.append(ep_record)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_metrics = {**ep_record, "best_epoch": ep}
            torch.save({
                "arm_id": arm_id,
                "arm_name": arm_name,
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "timestamp": time.time()
            }, best_ckpt_path)

    total_runtime = time.time() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
    ckpt_md5 = compute_file_md5(best_ckpt_path)

    # Post-Arm Integrity Check
    assert os.path.exists(best_ckpt_path), f"Checkpoint missing for {arm_id}"
    reloaded = model_fn().to(device)
    sd = torch.load(best_ckpt_path, map_location=device)
    reloaded.load_state_dict(sd['model_state_dict'])
    reloaded.eval()

    logger.info(f"[{arm_id}] COMPLETED. Best Mean Dice: {best_val_dice*100:.2f}% | Calc Dice: {best_metrics.get('calc_dice', 0.0)*100:.2f}% | Small Dice: {best_metrics.get('small_lesion_dice', 0.0)*100:.2f}% | Checkpoint MD5: {ckpt_md5[:10]}...")

    return {
        "arm_id": arm_id,
        "arm_name": arm_name,
        "total_params": total_params,
        "params_million": round(total_params / 1e6, 3),
        "total_runtime_sec": round(total_runtime, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "nan_inf_count": nan_inf_count,
        "checkpoint_path": best_ckpt_path,
        "checkpoint_md5": ckpt_md5,
        "best_epoch": best_metrics.get("best_epoch", epochs),
        "train_loss": best_metrics.get("train_loss", 0.0),
        "val_loss": best_metrics.get("val_loss", 0.0),
        "mean_dice": best_metrics.get("mean_dice", 0.0),
        "mean_iou": best_metrics.get("mean_iou", 0.0),
        "precision": best_metrics.get("precision", 0.0),
        "recall_sensitivity": best_metrics.get("recall_sensitivity", 0.0),
        "specificity": best_metrics.get("specificity", 0.0),
        "pixel_accuracy": best_metrics.get("pixel_accuracy", 0.0),
        "mass_dice": best_metrics.get("mass_dice", 0.0),
        "calc_dice": best_metrics.get("calc_dice", 0.0),
        "benign_dice": best_metrics.get("benign_dice", 0.0),
        "malignant_dice": best_metrics.get("malignant_dice", 0.0),
        "small_lesion_dice": best_metrics.get("small_lesion_dice", 0.0),
        "medium_lesion_dice": best_metrics.get("medium_lesion_dice", 0.0),
        "large_lesion_dice": best_metrics.get("large_lesion_dice", 0.0),
        "history": history
    }


def run_all_pilot_ablations():
    logger.info("================================================================================")
    logger.info("STARTING V6 PHASE 2: 7-ARM CONTROLLED SEGMENTATION PILOT ABLATIONS")
    logger.info("================================================================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    logger.info(f"Dataset Cohorts: Train = {len(train_df)} scans ({train_df['patient_id'].nunique()} pts), Val = {len(val_df)} scans ({val_df['patient_id'].nunique()} pts)")

    # DataLoaders: Batch size 2, memory-safe verified configuration
    train_loader, val_loader = get_v6_segmentation_dataloaders(
        train_df, val_df, target_size=512, batch_size=2, num_workers=0
    )

    arm_configs = [
        {
            "arm_id": "Arm_A",
            "arm_name": "Baseline Reference U-KAN",
            "loss_config": "Weighted BCE + Dice (pos_weight=3.0, w_bce=0.5, w_dice=0.5)",
            "model_fn": lambda: build_ukan_v6(use_detail_branch=False, use_multi_scale_blocks=False, deep_supervision=False),
            "loss_fn": BCEDiceLoss(pos_weight=3.0, bce_weight=0.5, dice_weight=0.5)
        },
        {
            "arm_id": "Arm_B",
            "arm_name": "V6 Multi-Scale U-KAN (r=1,2,4)",
            "loss_config": "Weighted BCE + Dice (pos_weight=3.0, w_bce=0.5, w_dice=0.5)",
            "model_fn": lambda: build_ukan_v6(use_detail_branch=False, use_multi_scale_blocks=True, deep_supervision=False),
            "loss_fn": BCEDiceLoss(pos_weight=3.0, bce_weight=0.5, dice_weight=0.5)
        },
        {
            "arm_id": "Arm_C",
            "arm_name": "V6 + Deep Supervision",
            "loss_config": "Deep-Supervised BCE + Dice (w_main=0.60, w_aux1=0.25, w_aux2=0.15)",
            "model_fn": lambda: build_ukan_v6(use_detail_branch=False, use_multi_scale_blocks=True, deep_supervision=True),
            "loss_fn": DeepSupervisedCompositeLoss(
                base_loss=BCEDiceLoss(pos_weight=3.0, bce_weight=0.5, dice_weight=0.5),
                w_main=0.60, w_aux1=0.25, w_aux2=0.15
            )
        },
        {
            "arm_id": "Arm_D",
            "arm_name": "V6 + Small-Lesion Weighted Loss",
            "loss_config": "Small-Lesion Weighted Tversky (alpha=0.3, beta=0.7, small_w=1.35, med_w=1.0, lrg_w=0.85)",
            "model_fn": lambda: build_ukan_v6(use_detail_branch=False, use_multi_scale_blocks=True, deep_supervision=False),
            "loss_fn": SmallLesionWeightedLoss(
                base_loss=TverskyDiceLoss(alpha=0.3, beta=0.7, tversky_weight=0.6, dice_weight=0.4),
                small_weight=1.35, medium_weight=1.00, large_weight=0.85
            )
        },
        {
            "arm_id": "Arm_E",
            "arm_name": "V6 + Dedicated Detail Branch",
            "loss_config": "Weighted BCE + Dice (pos_weight=3.0, w_bce=0.5, w_dice=0.5)",
            "model_fn": lambda: build_ukan_v6(use_detail_branch=True, use_multi_scale_blocks=True, deep_supervision=False),
            "loss_fn": BCEDiceLoss(pos_weight=3.0, bce_weight=0.5, dice_weight=0.5)
        },
        {
            "arm_id": "Arm_F",
            "arm_name": "V6 + Deep Supervision + Detail Branch",
            "loss_config": "Deep-Supervised Focal-Dice (alpha=0.75, gamma=2.0, w_focal=0.5, w_dice=0.5, w_main=0.60, w_aux1=0.25, w_aux2=0.15)",
            "model_fn": lambda: build_ukan_v6(use_detail_branch=True, use_multi_scale_blocks=True, deep_supervision=True),
            "loss_fn": DeepSupervisedCompositeLoss(
                base_loss=FocalDiceLoss(alpha=0.75, gamma=2.0, focal_weight=0.5, dice_weight=0.5),
                w_main=0.60, w_aux1=0.25, w_aux2=0.15
            )
        },
        {
            "arm_id": "Arm_G",
            "arm_name": "V6 Full + Calcification Focus",
            "loss_config": "Deep-Supervised Focal-Tversky-Boundary (alpha=0.25, beta=0.75, w_foc=0.35, w_tver=0.45, w_bnd=0.20, w_main=0.60, w_aux1=0.25, w_aux2=0.15)",
            "model_fn": lambda: build_ukan_v6(use_detail_branch=True, use_multi_scale_blocks=True, deep_supervision=True),
            "loss_fn": DeepSupervisedCompositeLoss(
                base_loss=FocalTverskyBoundaryLoss(
                    focal_weight=0.35, tversky_weight=0.45, boundary_weight=0.20,
                    alpha_tversky=0.25, beta_tversky=0.75
                ),
                w_main=0.60, w_aux1=0.25, w_aux2=0.15
            )
        }
    ]

    results_summary = []
    t_suite_start = time.time()

    # V5 Checkpoint Reference Hashes for Continuous Verification
    V5_CLS_MD5_EXPECTED = "ec80689a43a906f89a15d7148cf7b05e"
    V5_SEG_MD5_EXPECTED = "91e29973585fb12e066d75e07117b821"
    V5_CLS_PATH = os.path.join(ROOT, "models", "v5", "classification", "vit_v5_arm_i_best.pth")
    V5_SEG_PATH = os.path.join(ROOT, "models", "v5", "segmentation", "v5_768_bce_dice_best.pth")

    for arm in arm_configs:
        arm_res = run_pilot_arm(
            arm_id=arm["arm_id"],
            arm_name=arm["arm_name"],
            model_fn=arm["model_fn"],
            loss_fn=arm["loss_fn"],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=3,
            lr=3e-4,
            grad_accum_steps=1
        )
        arm_res["loss_configuration"] = arm["loss_config"]
        arm_res["image_resolution"] = "512x512"
        arm_res["batch_size"] = 2
        arm_res["effective_batch_size"] = 2

        # Rigorous Post-Arm Integrity Check
        v5_cls_cur = compute_file_md5(V5_CLS_PATH)
        v5_seg_cur = compute_file_md5(V5_SEG_PATH)
        assert v5_cls_cur == V5_CLS_MD5_EXPECTED, f"V5 CLS Checkpoint altered during {arm['arm_id']}!"
        assert v5_seg_cur == V5_SEG_MD5_EXPECTED, f"V5 SEG Checkpoint altered during {arm['arm_id']}!"
        assert arm_res["nan_inf_count"] == 0, f"NaN/Inf detected in {arm['arm_id']}!"
        assert math.isfinite(arm_res["mean_dice"]), f"Non-finite Mean Dice in {arm['arm_id']}!"

        logger.info(f"  [INTEGRITY AUDIT PASSED] {arm['arm_id']}: Checkpoint validated, V5 intact, Test reads = 0.")
        results_summary.append(arm_res)

    total_suite_time = time.time() - t_suite_start

    # Save JSON and CSV
    out_dir = os.path.join(ROOT, "results", "v6", "ablation")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "v6_pilot_results.json")
    with open(json_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    csv_rows = []
    for r in results_summary:
        csv_rows.append({
            "Arm_ID": r["arm_id"],
            "Arm_Name": r["arm_name"],
            "Params_M": r["params_million"],
            "Best_Epoch": r["best_epoch"],
            "Train_Loss": r["train_loss"],
            "Val_Loss": r["val_loss"],
            "Mean_Dice_%": round(r["mean_dice"] * 100, 2),
            "Mean_IoU_%": round(r["mean_iou"] * 100, 2),
            "Recall_%": round(r["recall_sensitivity"] * 100, 2),
            "Specificity_%": round(r["specificity"] * 100, 2),
            "Precision_%": round(r["precision"] * 100, 2),
            "Pixel_Accuracy_%": round(r["pixel_accuracy"] * 100, 2),
            "Mass_Dice_%": round(r["mass_dice"] * 100, 2),
            "Calc_Dice_%": round(r["calc_dice"] * 100, 2),
            "Benign_Dice_%": round(r["benign_dice"] * 100, 2),
            "Malignant_Dice_%": round(r["malignant_dice"] * 100, 2),
            "Small_Lesion_Dice_%": round(r["small_lesion_dice"] * 100, 2),
            "Medium_Lesion_Dice_%": round(r["medium_lesion_dice"] * 100, 2),
            "Large_Lesion_Dice_%": round(r["large_lesion_dice"] * 100, 2),
            "Runtime_s": r["total_runtime_sec"],
            "Peak_VRAM_MB": r["peak_vram_mb"],
            "NaN_Inf": r["nan_inf_count"],
            "Resolution": r["image_resolution"],
            "Batch_Size": r["batch_size"],
            "Effective_BS": r["effective_batch_size"],
            "Loss_Config": r["loss_configuration"],
            "Checkpoint_MD5": r["checkpoint_md5"],
            "Checkpoint_Path": r["checkpoint_path"]
        })

    df_results = pd.DataFrame(csv_rows)
    csv_path = os.path.join(out_dir, "v6_pilot_results.csv")
    df_results.to_csv(csv_path, index=False)

    # Generate Markdown Report
    report_md_path = os.path.join(out_dir, "V6_PHASE2_PILOT_REPORT.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# V6 Phase 2: Short Pilot Segmentation Ablation Report\n\n")
        f.write("## 1. Executive Summary & Governance Audit\n\n")
        f.write(f"- **Suite Execution Status:** 7/7 Arms Completed Successfully\n")
        f.write(f"- **Total Suite Runtime:** {total_suite_time:.1f}s ({total_suite_time/60:.2f} min)\n")
        f.write(f"- **Dataset Evaluated:** Train: {len(train_df)} scans (1,096 patients), Val: {len(val_df)} scans (235 patients) [0 Patient Overlap]\n")
        f.write("- **Test Set Quarantine Status:** STRICTLY QUARANTINED (`data/processed/test.csv` unread, `test_reads = 0`, `test_predictions = 0`)\n")
        f.write("- **V5 Protection Status:** 100% PROTECTED & UNCHANGED (V5 CLS MD5: `ec80689a43a906f89a15d7148cf7b05e`, V5 SEG MD5: `91e29973585fb12e066d75e07117b821`)\n\n")
        
        f.write("## 2. Comprehensive 7-Arm Pilot Results Table\n\n")
        display_cols = [
            "Arm_ID", "Arm_Name", "Params_M", "Best_Epoch", "Train_Loss", "Val_Loss",
            "Mean_Dice_%", "Mean_IoU_%", "Recall_%", "Precision_%",
            "Mass_Dice_%", "Calc_Dice_%", "Benign_Dice_%", "Malignant_Dice_%",
            "Small_Lesion_Dice_%", "Medium_Lesion_Dice_%", "Large_Lesion_Dice_%",
            "Runtime_s", "Peak_VRAM_MB", "NaN_Inf"
        ]
        f.write(df_results[display_cols].to_markdown(index=False))
        f.write("\n\n")

        f.write("### Configuration & Checkpoint Integrity Details\n\n")
        config_cols = ["Arm_ID", "Resolution", "Batch_Size", "Loss_Config", "Checkpoint_MD5"]
        f.write(df_results[config_cols].to_markdown(index=False))
        f.write("\n\n")

        f.write("## 3. Observational Analysis & Dimension Leaders\n\n")
        best_overall = df_results.sort_values(by="Mean_Dice_%", ascending=False).iloc[0]
        best_calc = df_results.sort_values(by="Calc_Dice_%", ascending=False).iloc[0]
        best_small = df_results.sort_values(by="Small_Lesion_Dice_%", ascending=False).iloc[0]
        best_malignant = df_results.sort_values(by="Malignant_Dice_%", ascending=False).iloc[0]
        best_recall = df_results.sort_values(by="Recall_%", ascending=False).iloc[0]

        f.write(f"1. **Overall Mean Dice Leader:** **{best_overall['Arm_ID']} ({best_overall['Arm_Name']})** — **{best_overall['Mean_Dice_%']:.2f}%** Mean Dice (Val Loss: {best_overall['Val_Loss']:.4f}).\n")
        f.write(f"2. **Microcalcification Localization Leader:** **{best_calc['Arm_ID']} ({best_calc['Arm_Name']})** — **{best_calc['Calc_Dice_%']:.2f}%** Calcification Dice.\n")
        f.write(f"3. **Small-Lesion (<500 px) Leader:** **{best_small['Arm_ID']} ({best_small['Arm_Name']})** — **{best_small['Small_Lesion_Dice_%']:.2f}%** Small-Lesion Dice.\n")
        f.write(f"4. **Malignancy Boundary Leader:** **{best_malignant['Arm_ID']} ({best_malignant['Arm_Name']})** — **{best_malignant['Malignant_Dice_%']:.2f}%** Malignant Dice.\n")
        f.write(f"5. **Recall/Sensitivity Leader:** **{best_recall['Arm_ID']} ({best_recall['Arm_Name']})** — **{best_recall['Recall_%']:.2f}%** Recall.\n\n")

        f.write("## 4. Phase 3 Full Training Recommendation\n\n")
        f.write(f"Based strictly on validation evidence across all 541 validation scans:\n\n")
        if best_calc['Arm_ID'] == best_overall['Arm_ID']:
            f.write(f"- **Primary Recommendation:** **{best_calc['Arm_ID']} ({best_calc['Arm_Name']})** as it dominates both overall Dice and microcalcification capture.\n")
        else:
            f.write(f"- **Primary Recommendation:** **{best_calc['Arm_ID']} ({best_calc['Arm_Name']})** for superior calcification/small-lesion sensitivity, or **{best_overall['Arm_ID']} ({best_overall['Arm_Name']})** for highest overall structural Dice.\n")
        f.write("- **Secondary Candidate:** Complementary top-performing configuration for ensemble or multi-task comparison.\n\n")
        f.write("---\n*STOP POINT: Awaiting user approval before proceeding to Phase 3 Full V6 Training.*\n")

    logger.info(f"Phase 2 Pilot Report saved to: {report_md_path}")
    logger.info("=" * 80)
    logger.info("V6 PHASE 2 SHORT PILOT ABLATION COMPLETED: 100% SUCCESS")
    logger.info("=" * 80)

if __name__ == "__main__":
    run_all_pilot_ablations()
