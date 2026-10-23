import os
import argparse
import logging
import json
from typing import Dict, Any, Optional
import numpy as np
import yaml
import torch
from tqdm import tqdm

from utils.seed import get_device
from utils.metrics import calculate_segmentation_metrics
from utils.visualization import plot_segmentation_prediction
from segmentation.ukan import build_ukan_from_config
from segmentation.dataset import get_segmentation_loaders

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("UKANEvaluate")


def evaluate_ukan(
    config_path: str = "config.yaml",
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates trained U-KAN model strictly on the held-out test dataset split.
    Calculates actual Dice, IoU, Precision, Recall, Sensitivity, and Specificity.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if checkpoint_path is None:
        checkpoint_path = cfg.get("segmentation", {}).get("checkpoint_path", "models/ukan_best.pth")

    if not os.path.exists(checkpoint_path):
        msg = f"Model checkpoint not available at '{checkpoint_path}'. Training is required before evaluation."
        logger.warning(msg)
        return {"status": "checkpoint_missing", "message": msg}

    # 1. Device and Loader
    device = get_device(cfg.get("project", {}).get("device", "auto"))
    _, _, test_loader = get_segmentation_loaders(config_path)

    if test_loader is None or len(test_loader) == 0:
        msg = "Test dataset split is not available or metadata is unconfigured."
        logger.warning(msg)
        return {"status": "test_data_missing", "message": msg}

    # 2. Load Model Checkpoint
    model = build_ukan_from_config(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Loaded U-KAN checkpoint from {checkpoint_path}. Evaluating {len(test_loader)} test cases...")

    metrics_accumulator = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "sensitivity": [],
        "specificity": []
    }

    results_dir = cfg.get("paths", {}).get("segmentation_results", "results/segmentation")
    figures_dir = cfg.get("paths", {}).get("figures_dir", "results/figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating U-KAN Test Set")):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            patient_id = batch["patient_id"][0]

            logits = model(images)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            target_mask = masks.squeeze().cpu().numpy()
            raw_img = images.squeeze().cpu().numpy()

            m = calculate_segmentation_metrics(probs, target_mask, threshold=0.5)
            for k in metrics_accumulator:
                metrics_accumulator[k].append(m[k])

            # Save the first 5 test case visualizations
            if i < 5:
                pred_binary = (probs >= 0.5).astype(np.uint8)
                fig_path = os.path.join(figures_dir, f"test_segmentation_patient_{patient_id}_{i+1}.png")
                plot_segmentation_prediction(raw_img, pred_binary, true_mask=target_mask, save_path=fig_path)

    # Compute aggregate real metrics
    summary_metrics = {
        "status": "success",
        "num_test_samples": len(test_loader),
        "mean_dice": float(np.mean(metrics_accumulator["dice"])),
        "std_dice": float(np.std(metrics_accumulator["dice"])),
        "mean_iou": float(np.mean(metrics_accumulator["iou"])),
        "std_iou": float(np.std(metrics_accumulator["iou"])),
        "mean_precision": float(np.mean(metrics_accumulator["precision"])),
        "mean_recall": float(np.mean(metrics_accumulator["recall"])),
        "mean_sensitivity": float(np.mean(metrics_accumulator["sensitivity"])),
        "mean_specificity": float(np.mean(metrics_accumulator["specificity"]))
    }

    # Save to JSON
    json_path = os.path.join(results_dir, "test_evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info("="*50)
    logger.info("REAL U-KAN TEST EVALUATION RESULTS:")
    for k, v in summary_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:20s}: {v:.4f}")
        else:
            logger.info(f"  {k:20s}: {v}")
    logger.info("="*50)

    return summary_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained U-KAN segmentation model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    args = parser.parse_args()
    evaluate_ukan(config_path=args.config, checkpoint_path=args.checkpoint)
