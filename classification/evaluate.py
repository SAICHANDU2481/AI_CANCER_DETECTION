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
from utils.metrics import calculate_classification_metrics
from utils.visualization import plot_confusion_matrix, plot_roc_curve
from classification.vit_model import build_vit_from_config
from classification.dataset import get_classification_loaders

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ViTEvaluate")


def evaluate_vit(
    config_path: str = "config.yaml",
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates trained ViT classifier strictly on the held-out test dataset split.
    Calculates actual Accuracy, Precision, Recall, Sensitivity, Specificity, F1, and ROC-AUC.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if checkpoint_path is None:
        checkpoint_path = cfg.get("classification", {}).get("checkpoint_path", "models/vit_best.pth")

    if not os.path.exists(checkpoint_path):
        msg = f"Model checkpoint not available at '{checkpoint_path}'. Training is required before evaluation."
        logger.warning(msg)
        return {"status": "checkpoint_missing", "message": msg}

    # 1. Device and Loader
    device = get_device(cfg.get("project", {}).get("device", "auto"))
    _, _, test_loader, _ = get_classification_loaders(config_path)

    if test_loader is None or len(test_loader) == 0:
        msg = "Test dataset split is not available or metadata is unconfigured."
        logger.warning(msg)
        return {"status": "test_data_missing", "message": msg}

    # 2. Load Checkpoint
    model = build_vit_from_config(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Loaded ViT checkpoint from {checkpoint_path}. Evaluating {len(test_loader)} test cases...")

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating ViT Test Set"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    # 3. Calculate real metrics
    metrics = calculate_classification_metrics(all_targets, all_preds, all_probs)
    metrics["status"] = "success"
    metrics["num_test_samples"] = len(test_loader)

    # 4. Save metrics and plots
    results_dir = cfg.get("paths", {}).get("classification_results", "results/classification")
    figures_dir = cfg.get("paths", {}).get("figures_dir", "results/figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "test_evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    cm_arr = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm_arr, class_names=["Benign", "Malignant"], save_path=os.path.join(figures_dir, "vit_confusion_matrix.png"))

    if metrics["roc_auc"] is not None:
        plot_roc_curve(np.array(all_targets), np.array(all_probs), roc_auc=metrics["roc_auc"], save_path=os.path.join(figures_dir, "vit_roc_curve.png"))

    logger.info("="*50)
    logger.info("REAL ViT TEST EVALUATION RESULTS:")
    logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")
    logger.info(f"  Precision:   {metrics['precision']:.4f}")
    logger.info(f"  Recall/Sens: {metrics['recall']:.4f}")
    logger.info(f"  Specificity: {metrics['specificity']:.4f}")
    logger.info(f"  F1-Score:    {metrics['f1_score']:.4f}")
    if metrics['roc_auc'] is not None:
        logger.info(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    logger.info("="*50)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Vision Transformer classifier on test set.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    args = parser.parse_args()
    evaluate_vit(config_path=args.config, checkpoint_path=args.checkpoint)
