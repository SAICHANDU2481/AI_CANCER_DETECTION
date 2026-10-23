"""
================================================================================
V7 Stage 0: Base Module & Fast Patient-Clustered Bootstrap Statistics
================================================================================
STRICT GOVERNANCE:
- Operates EXCLUSIVELY on TRAIN (2,501) and VALIDATION (541) splits.
- ZERO access to data/processed/test.csv or test images/masks.
- Evaluates frozen V6 U-KAN (Arm G) and frozen V6 ViT (Arm C).
- Computes patient-clustered bootstrap confidence intervals (1,000 resamples).
================================================================================
"""

import os
import sys
import time
import json
import logging
import random
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from classification.v6.model_v6 import MultiScaleViTV6
from classification.v6.dataset_v6 import V6CachedMultiScaleDataset
from classification.v6.roi_generator_v6 import MultiScaleROIGeneratorV6
from segmentation.v6.ukan_v6 import build_ukan_v6

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("V7_Stage0_Base")

# Strict quarantine assertions
FORBIDDEN_KEYWORDS = ['test.csv', 'verified_test.csv', 'results/v7/test', 'data/cache/v7_proposals/test']
for kw in FORBIDDEN_KEYWORDS:
    assert kw not in __file__, f"Quarantine violation in file: {kw}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(ROOT, "results", "v7")
DATA_INTEGRITY_DIR = os.path.join(RESULTS_DIR, "data_integrity")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_INTEGRITY_DIR, exist_ok=True)

FROZEN_UKAN_PATH = os.path.join(ROOT, "models", "v6", "segmentation", "final", "ukan_v6_arm_g_best.pth")
FROZEN_VIT_PATH = os.path.join(ROOT, "models", "v6", "classification", "vit_v6_arm_c_best.pth")
TRAIN_CACHE_DIR = os.path.join(ROOT, "data", "cache", "v6_machine_rois", "train")
VAL_CACHE_DIR = os.path.join(ROOT, "data", "cache", "v6_machine_rois", "val")
TRAIN_CSV = os.path.join(ROOT, "data", "processed", "train.csv")
VAL_CSV = os.path.join(ROOT, "data", "processed", "val.csv")


def resolve_image_path(p: str) -> str:
    if not isinstance(p, str) or not p:
        return ""
    if os.path.exists(p):
        return p
    p_fixed = p.replace("OneDrive\\Desktop\\Capstone", "Desktop\\Capstone").replace("OneDrive/Desktop/Capstone", "Desktop/Capstone")
    if os.path.exists(p_fixed):
        return p_fixed
    p_fixed2 = p.replace("OneDrive\\Desktop\\AI_CANCER_DETECTION", "Desktop\\AI_CANCER_DETECTION").replace("OneDrive/Desktop/AI_CANCER_DETECTION", "Desktop/AI_CANCER_DETECTION")
    if os.path.exists(p_fixed2):
        return p_fixed2
    return p


def patient_clustered_bootstrap_metric(
    df: pd.DataFrame,
    pred_col: str,
    prob_col: str,
    target_col: str,
    patient_col: str = "patient_id",
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Computes patient-clustered bootstrap 95% confidence intervals for all classification metrics.
    Fast vectorized numpy implementation.
    """
    df_clean = df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    
    unique_patients = df_clean[patient_col].unique()
    n_patients = len(unique_patients)
    if n_patients == 0:
        return {}
        
    patient_to_indices = {p: np.where(df_clean[patient_col] == p)[0] for p in unique_patients}
    
    y_true_all = df_clean[target_col].values.astype(int)
    y_pred_all = df_clean[pred_col].values.astype(int)
    y_prob_all = df_clean[prob_col].values.astype(float)
    
    metrics = {'acc': [], 'prec': [], 'rec': [], 'spec': [], 'bin_f1': [], 'macro_f1': [], 'roc_auc': [], 'pr_auc': []}
    
    for _ in range(n_bootstraps):
        sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        sampled_indices = np.concatenate([patient_to_indices[p] for p in sampled_patients])
        
        y_true = y_true_all[sampled_indices]
        y_pred = y_pred_all[sampled_indices]
        y_prob = y_prob_all[sampled_indices]
        
        # Check if both classes are present
        if len(np.unique(y_true)) < 2:
            continue
            
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        total = tp + tn + fp + fn
        
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bin_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_0 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) > 0 else 0.0
        macro_f1 = (bin_f1 + f1_0) / 2.0
        
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        except Exception:
            roc_auc, pr_auc = 0.5, 0.5
            
        metrics['acc'].append(acc)
        metrics['prec'].append(prec)
        metrics['rec'].append(rec)
        metrics['spec'].append(spec)
        metrics['bin_f1'].append(bin_f1)
        metrics['macro_f1'].append(macro_f1)
        metrics['roc_auc'].append(roc_auc)
        metrics['pr_auc'].append(pr_auc)
        
    summary = {}
    for k, v in metrics.items():
        arr = np.array(v)
        if len(arr) == 0:
            summary[k] = {'mean': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0, 'std': 0.0}
        else:
            summary[k] = {
                'mean': float(np.mean(arr)),
                'ci_lower': float(np.percentile(arr, 2.5)),
                'ci_upper': float(np.percentile(arr, 97.5)),
                'std': float(np.std(arr))
            }
    return summary


def compute_metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bin_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    f1_0 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) > 0 else 0.0
    macro_f1 = (bin_f1 + f1_0) / 2.0
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        roc_auc, pr_auc = 0.5, 0.5
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'specificity': float(spec),
        'binary_f1': float(bin_f1),
        'macro_f1': float(macro_f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


def load_frozen_v6_classifier() -> MultiScaleViTV6:
    """Loads frozen V6 Arm-C classifier checkpoint."""
    logger.info(f"Loading frozen V6 Arm-C Classifier from {FROZEN_VIT_PATH}")
    model = MultiScaleViTV6(
        backbone_name='vit_base_patch16_224.augreg2_in21k_ft_in1k',
        num_classes=2,
        use_full_image=True,
        use_calc_patch=True,
        scales_used=('local', 'medium', 'large', 'context'),
        pretrained=False,
        k=3
    )
    ckpt = torch.load(FROZEN_VIT_PATH, map_location="cpu")
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
