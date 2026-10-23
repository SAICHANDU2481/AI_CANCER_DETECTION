"""
================================================================================
V9 Phase 5: Final Locked Test Evaluation
================================================================================
Scientific Objective:
Perform ONE final, strictly locked, patient-level independent test evaluation of
the complete V9 system:
- Frozen V9 Phase 2 Arm G Multi-Candidate Classifier
- Frozen V6 U-KAN Multi-Scale Segmentation
- Frozen Temperature Scaling (T = 1.2128)
- Fixed Decision Threshold = 0.50
- Primary Test Cohort (N = 525 native-ROI scans / 235 patients)
- Secondary Comprehensive Cohort (N = 526 scans)

ABSOLUTE GOVERNANCE:
- ONE-TIME LOCKED TEST EXECUTION.
- NO post-test tuning, NO TTA, NO threshold optimization.
================================================================================
"""

import os
import sys
import time
import math
import json
import hashlib
import random
import gc
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from sklearn import metrics as sk_metrics
from sklearn.calibration import calibration_curve

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', write_through=True)

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.config import UKAN_V6_PATH, EXPECTED_MD5, verify_file_md5
from app.preprocessing import preprocess_mammogram_full
from segmentation.v6.ukan_v6 import MultiScaleUKANV6
from classification.v9.topk_proposals_v9 import (
    V9MultiSourceProposalGenerator,
    detect_breast_mask,
    compute_box_iou,
    apply_safety_margin
)
from classification.v9.model_v9 import V9MultiCandidateClassifier
from classification.v9.dataset_v9 import (
    V9MultiCandidateDataset,
    verify_or_build_v9_phase2_cache,
    resolve_file_path,
    compute_train_quality_statistics
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

FROZEN_TEMPERATURE = 1.2128
FROZEN_THRESHOLD = 0.50


def compute_file_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Tuple[float, List[Dict[str, Any]]]:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    bin_details = []
    if n == 0:
        return 0.0, bin_details

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        count_in_bin = int(np.sum(in_bin))
        prop_in_bin = float(count_in_bin / n)
        
        if count_in_bin > 0:
            accuracy_in_bin = float(np.mean(y_true[in_bin]))
            avg_confidence_in_bin = float(np.mean(y_prob[in_bin]))
            bin_err = float(np.abs(avg_confidence_in_bin - accuracy_in_bin))
            ece += bin_err * prop_in_bin
        else:
            accuracy_in_bin = 0.0
            avg_confidence_in_bin = float((bin_lower + bin_upper) / 2.0)
            bin_err = 0.0

        bin_details.append({
            'bin_idx': i + 1,
            'bin_lower': round(float(bin_lower), 4),
            'bin_upper': round(float(bin_upper), 4),
            'bin_center': round(float((bin_lower + bin_upper) / 2.0), 4),
            'count': count_in_bin,
            'prop': round(prop_in_bin, 4),
            'avg_confidence': round(avg_confidence_in_bin, 4),
            'empirical_accuracy': round(accuracy_in_bin, 4),
            'absolute_error': round(bin_err, 4)
        })

    return float(ece), bin_details


def compute_nll(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    p_clipped = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped)))


def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50, n_bins: int = 15) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = float(sk_metrics.accuracy_score(y_true, y_pred))
    prec = float(sk_metrics.precision_score(y_true, y_pred, zero_division=0))
    rec = float(sk_metrics.recall_score(y_true, y_pred, zero_division=0))
    spec_tn_fp = sk_metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    if spec_tn_fp.shape == (2, 2):
        tn, fp, fn, tp = int(spec_tn_fp[0, 0]), int(spec_tn_fp[0, 1]), int(spec_tn_fp[1, 0]), int(spec_tn_fp[1, 1])
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    f1 = float(sk_metrics.f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(sk_metrics.f1_score(y_true, y_pred, average='macro', zero_division=0))

    try:
        roc_auc = float(sk_metrics.roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.50
    try:
        pr_auc = float(sk_metrics.average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))

    brier = float(np.mean((y_prob - y_true) ** 2))
    ece, bin_details = compute_ece(y_true, y_prob, n_bins=n_bins)
    nll = compute_nll(y_true, y_prob)

    return {
        "accuracy": round(acc * 100.0, 2),
        "precision": round(prec * 100.0, 2),
        "recall_sensitivity": round(rec * 100.0, 2),
        "specificity": round(spec * 100.0, 2),
        "f1_binary": round(f1 * 100.0, 2),
        "macro_f1": round(macro_f1 * 100.0, 2),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece * 100.0, 2),
        "nll": round(nll, 4),
        "mean_confidence": round(float(np.mean(y_prob)), 4),
        "empirical_positive_rate": round(float(np.mean(y_true)), 4),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "bin_details": bin_details
    }


def run_patient_clustered_bootstrap(
    df_preds: pd.DataFrame,
    prob_col: str,
    target_col: str = 'target',
    patient_col: str = 'patient_id',
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    patients = df_preds[patient_col].unique()
    n_pts = len(patients)

    boot_metrics = {
        'accuracy': [],
        'precision': [],
        'recall_sensitivity': [],
        'specificity': [],
        'f1_binary': [],
        'macro_f1': [],
        'roc_auc': [],
        'pr_auc': [],
        'brier_score': [],
        'ece': []
    }

    patient_to_indices = {p: df_preds[df_preds[patient_col] == p].index.values for p in patients}

    for _ in range(n_bootstraps):
        sampled_pts = rng.choice(patients, size=n_pts, replace=True)
        sampled_idx = np.concatenate([patient_to_indices[p] for p in sampled_pts])
        
        y_t = df_preds.loc[sampled_idx, target_col].values
        y_p = df_preds.loc[sampled_idx, prob_col].values
        
        if len(np.unique(y_t)) < 2:
            continue
            
        m = compute_all_metrics(y_t, y_p)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])

    results = {}
    for k, vals in boot_metrics.items():
        arr = np.array(vals)
        low = float(np.percentile(arr, 2.5))
        high = float(np.percentile(arr, 97.5))
        mean_v = float(np.mean(arr))
        results[k] = {
            'mean': round(mean_v, 4),
            'ci_lower': round(low, 4),
            'ci_upper': round(high, 4),
            'ci_str': f"[{low:.2f}% - {high:.2f}%]" if 'auc' not in k and 'brier' not in k and 'nll' not in k else f"[{low:.4f} - {high:.4f}]"
        }

    return results


def run_phase5_final_test():
    t_start = time.time()
    print("=" * 85)
    print("V9 PHASE 5 — FINAL LOCKED TEST EVALUATION")
    print("=" * 85)

    # 1. PRE-TEST GOVERNANCE AUDIT
    print("\n--- 1. Pre-Test Checkpoint & Split Integrity Audit ---")
    arm_g_path = os.path.join("models", "v9", "phase2", "v9_phase2_arm_g_best.pth")
    assert os.path.exists(arm_g_path), f"Missing Arm G checkpoint: {arm_g_path}"
    arm_g_md5 = compute_file_md5(arm_g_path)
    expected_arm_g_md5 = "c60b1aab0a1bc809c9ae27a57ddc65ec"
    assert arm_g_md5 == expected_arm_g_md5, f"Arm G MD5 mismatch: {arm_g_md5} != {expected_arm_g_md5}"
    print(f"[AUDIT] Locked Arm G Checkpoint MD5: {arm_g_md5} -> PASS (EXACT MATCH)")

    is_u_ok, u_md5 = verify_file_md5(UKAN_V6_PATH, EXPECTED_MD5['ukan_v6'])
    assert is_u_ok, f"U-KAN verification failed! MD5: {u_md5}"
    print(f"[AUDIT] Frozen V6 U-KAN Checkpoint MD5: {u_md5} -> PASS (EXACT MATCH)")

    # Data split verification
    train_df = pd.read_csv('data/processed/verified_train.csv')
    val_df = pd.read_csv('data/processed/verified_val.csv')
    full_test_df = pd.read_csv('data/processed/verified_test.csv')

    train_pts = set(train_df['patient_id'].astype(str).unique())
    val_pts = set(val_df['patient_id'].astype(str).unique())
    test_pts = set(full_test_df['patient_id'].astype(str).unique())

    print(f"[AUDIT] Train Patients: {len(train_pts)}, Val Patients: {len(val_pts)}, Test Patients: {len(test_pts)}")
    assert len(train_pts.intersection(val_pts)) == 0, "Train-Val patient leakage detected!"
    assert len(train_pts.intersection(test_pts)) == 0, "Train-Test patient leakage detected!"
    assert len(val_pts.intersection(test_pts)) == 0, "Val-Test patient leakage detected!"
    print("[AUDIT] Patient Isolation: Strict 0 overlap across all splits -> PASS")

    # Primary Cohort (525 native-ROI scans) and Comprehensive Cohort (526 scans)
    primary_test_df = full_test_df[full_test_df['case_id'] != 'CALCIFICATION_P_01563_RIGHT_MLO_ab2'].copy().reset_index(drop=True)
    assert len(primary_test_df) == 525, f"Primary cohort mismatch: {len(primary_test_df)} != 525"
    assert len(primary_test_df['patient_id'].unique()) == 235, f"Primary patient count mismatch: {len(primary_test_df['patient_id'].unique())} != 235"
    print(f"[AUDIT] Primary Test Cohort: {len(primary_test_df)} scans ({len(primary_test_df[primary_test_df['label']==0])} Benign, {len(primary_test_df[primary_test_df['label']==1])} Malignant, 235 Patients) -> PASS")
    print(f"[AUDIT] Comprehensive Test Cohort: {len(full_test_df)} scans ({len(full_test_df[full_test_df['label']==0])} Benign, {len(full_test_df[full_test_df['label']==1])} Malignant, 235 Patients) -> PASS")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[HARDWARE] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    results_dir = os.path.join("results", "v9", "phase5_final_test")
    os.makedirs(results_dir, exist_ok=True)

    # 2. GENERATE TEST PROPOSALS USING FROZEN U-KAN (Arm 4 / Arm 9 Multi-Source)
    print("\n--- 2. Generating Test Split Proposals using Frozen U-KAN ---")
    test_proposals_cache_path = os.path.join("data", "cache", "v9", "phase1", "test_proposals_k5.pt")
    
    if os.path.exists(test_proposals_cache_path) and os.path.getsize(test_proposals_cache_path) > 10000:
        print(f"[CACHE] Loading existing Test Proposal Cache from {test_proposals_cache_path}")
        test_cache_proposals = torch.load(test_proposals_cache_path, weights_only=False)
    else:
        ukan = MultiScaleUKANV6(in_channels=1, out_channels=1)
        ckpt = torch.load(UKAN_V6_PATH, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        ukan.load_state_dict(sd)
        ukan.to(device).eval()

        prop_gen = V9MultiSourceProposalGenerator(
            min_area=12,
            thresholds=[0.10, 0.20, 0.35, 0.50],
            nms_iou_thresh=0.50,
            margin_ratio=0.20
        )

        test_cache_proposals = {}
        t0_prop = time.time()
        n_test = len(full_test_df)
        
        for idx, row in full_test_df.iterrows():
            if (idx + 1) % 100 == 0 or (idx + 1) == n_test:
                el = time.time() - t0_prop
                print(f"  Processed {idx + 1}/{n_test} test proposal scans ({el:.1f}s, {(idx+1)/el:.1f} scans/s)...")
                
            case_id = str(row.get('case_id', f"test_{idx}"))
            patient_id = str(row['patient_id'])
            img_p = resolve_file_path(row['image_path'])
            
            raw_img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            prep = preprocess_mammogram_full(raw_img)
            img_512 = prep["input_512"].astype(np.float32)
            img_768 = prep["input_768"].astype(np.float32)
            
            t_512 = torch.from_numpy(img_512).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.inference_mode():
                with autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    p_512_map = torch.sigmoid(ukan(t_512)).squeeze().float().cpu().numpy().astype(np.float32)
                    
            t_768 = torch.from_numpy(img_768).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.inference_mode():
                with autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    p_768_map = torch.sigmoid(ukan(t_768)).squeeze().float().cpu().numpy().astype(np.float32)
                    
            all_arms_test = prop_gen.generate_all_arms(img_512, img_768, p_512_map, p_768_map, k=5)
            test_cache_proposals[case_id] = {
                'patient_id': patient_id,
                'proposals': all_arms_test['Arm 9 (V9 Full Multi-Source + Ranking)']
            }
            
        torch.save(test_cache_proposals, test_proposals_cache_path)
        print(f"[CACHE] Saved {len(test_cache_proposals)} test proposal records to {test_proposals_cache_path}")

    # 3. BUILD TEST NPZ ROI CACHE USING TRAIN NORMALIZATION STATISTICS
    print("\n--- 3. Extracting Test Visual Pathways & Normalizing with Train Statistics ---")
    train_props_path = os.path.join("data", "cache", "v9", "phase1", "train_proposals_k5.pt")
    train_props = torch.load(train_props_path, weights_only=False)
    train_stats = compute_train_quality_statistics(train_props)
    
    test_npz_dir = os.path.join("data", "cache", "v9", "phase2", "test_npz")
    verify_or_build_v9_phase2_cache(
        df=full_test_df,
        proposals_cache_path=test_proposals_cache_path,
        target_cache_dir=test_npz_dir,
        split_name='test',
        train_stats=train_stats
    )

    # 4. LOAD LOCKED ARM G CLASSIFIER
    print("\n--- 4. Loading Locked Arm G Classifier ---")
    model = V9MultiCandidateClassifier(
        arm_type='Arm_G', k=5, dropout_rate=0.25, pretrained=False
    ).to(device)
    sd = torch.load(arm_g_path, map_location=device, weights_only=False)['model_state_dict']
    model.load_state_dict(sd)
    model.eval()

    # 5. DETERMINISM AUDIT (10 Test Cases Run Twice)
    print("\n--- 5. Running Determinism Audit on 10 Test Cases ---")
    audit_subset = primary_test_df.iloc[:10].reset_index(drop=True)
    audit_ds = V9MultiCandidateDataset(
        df=audit_subset, cache_dir=test_npz_dir, k=5, is_training=False,
        use_full_mammo=True, use_calc_patch=True, use_context_roi=True
    )
    audit_loader = DataLoader(audit_ds, batch_size=10, shuffle=False, num_workers=0)

    def run_logits_extraction(l_loader):
        all_logits = []
        with torch.inference_mode():
            for b in l_loader:
                c_crops = b['candidate_crops'].to(device)
                c_quals = b['candidate_qualities'].to(device)
                f_mammo = b['full_mammo'].to(device)
                c_patch = b['calc_patch'].to(device)
                ctx_roi = b['context_roi'].to(device)
                with autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    logits_2d, _ = model(c_crops, c_quals, f_mammo, c_patch, ctx_roi, chunk_size=8)
                all_logits.append(logits_2d.cpu().numpy())
        return np.concatenate(all_logits, axis=0)

    r1_logits_2d = run_logits_extraction(audit_loader)
    r2_logits_2d = run_logits_extraction(audit_loader)
    
    r1_bin_logits = r1_logits_2d[:, 1] - r1_logits_2d[:, 0]
    r2_bin_logits = r2_logits_2d[:, 1] - r2_logits_2d[:, 0]
    r1_probs = 1.0 / (1.0 + np.exp(-r1_bin_logits / FROZEN_TEMPERATURE))
    r2_probs = 1.0 / (1.0 + np.exp(-r2_bin_logits / FROZEN_TEMPERATURE))

    max_prob_diff = float(np.max(np.abs(r1_probs - r2_probs)))
    max_logit_diff = float(np.max(np.abs(r1_bin_logits - r2_bin_logits)))
    print(f"  Audit Result -> Max Prob Diff: {max_prob_diff:.8f}, Max Logit Diff: {max_logit_diff:.8f}")
    assert max_prob_diff < 1e-6, "Determinism check failed!"

    with open(os.path.join(results_dir, "v9_phase5_final_test_determinism_audit.json"), 'w') as f:
        json.dump({
            "status": "PASSED",
            "evaluated_scans": 10,
            "max_probability_difference": max_prob_diff,
            "max_logit_difference": max_logit_diff,
            "is_deterministic": True
        }, f, indent=2)

    # 6. PRIMARY TEST INFERENCE (N = 525) & COMPREHENSIVE (N = 526)
    print("\n--- 6. Running Final Locked Test Inference (Primary N = 525) ---")
    primary_test_ds = V9MultiCandidateDataset(
        df=primary_test_df, cache_dir=test_npz_dir, k=5, is_training=False,
        use_full_mammo=True, use_calc_patch=True, use_context_roi=True
    )
    primary_loader = DataLoader(primary_test_ds, batch_size=16, shuffle=False, num_workers=0)
    
    primary_logits_2d = run_logits_extraction(primary_loader)
    primary_bin_logits = primary_logits_2d[:, 1] - primary_logits_2d[:, 0]
    
    # Apply Frozen Temperature Scaling (T = 1.2128)
    primary_raw_probs = 1.0 / (1.0 + np.exp(-np.clip(primary_bin_logits, -30.0, 30.0)))
    primary_cal_probs = 1.0 / (1.0 + np.exp(-np.clip(primary_bin_logits / FROZEN_TEMPERATURE, -30.0, 30.0)))
    
    primary_y_true = primary_test_df['label'].values.astype(int)
    primary_y_pred = (primary_cal_probs >= FROZEN_THRESHOLD).astype(int)

    # Comprehensive Cohort (N = 526)
    comp_test_ds = V9MultiCandidateDataset(
        df=full_test_df, cache_dir=test_npz_dir, k=5, is_training=False,
        use_full_mammo=True, use_calc_patch=True, use_context_roi=True
    )
    comp_loader = DataLoader(comp_test_ds, batch_size=16, shuffle=False, num_workers=0)
    comp_logits_2d = run_logits_extraction(comp_loader)
    comp_bin_logits = comp_logits_2d[:, 1] - comp_logits_2d[:, 0]
    comp_cal_probs = 1.0 / (1.0 + np.exp(-np.clip(comp_bin_logits / FROZEN_TEMPERATURE, -30.0, 30.0)))
    comp_y_true = full_test_df['label'].values.astype(int)

    # 7. COMPUTE PRIMARY AND COMPREHENSIVE METRICS
    primary_metrics = compute_all_metrics(primary_y_true, primary_cal_probs, threshold=FROZEN_THRESHOLD)
    comp_metrics = compute_all_metrics(comp_y_true, comp_cal_probs, threshold=FROZEN_THRESHOLD)

    print("\n" + "=" * 85)
    print("V9 PRIMARY FINAL LOCKED TEST RESULTS (N = 525)")
    print("=" * 85)
    print(f"Accuracy:     {primary_metrics['accuracy']:.2f}%")
    print(f"Macro-F1:     {primary_metrics['macro_f1']:.2f}%")
    print(f"Binary F1:    {primary_metrics['f1_binary']:.2f}%")
    print(f"ROC-AUC:      {primary_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:       {primary_metrics['pr_auc']:.4f}")
    print(f"Sensitivity:  {primary_metrics['recall_sensitivity']:.2f}% ({primary_metrics['tp']}/{primary_metrics['tp'] + primary_metrics['fn']})")
    print(f"Specificity:  {primary_metrics['specificity']:.2f}% ({primary_metrics['tn']}/{primary_metrics['tn'] + primary_metrics['fp']})")
    print(f"Precision:    {primary_metrics['precision']:.2f}%")
    print(f"Brier Score:  {primary_metrics['brier_score']:.4f}")
    print(f"ECE (15 bin): {primary_metrics['ece']:.2f}%")
    print(f"NLL:          {primary_metrics['nll']:.4f}")
    print(f"Confusion Matrix: TN={primary_metrics['tn']}, FP={primary_metrics['fp']}, FN={primary_metrics['fn']}, TP={primary_metrics['tp']}")
    print("=" * 85)

    print("\n" + "=" * 85)
    print("V9 SECONDARY COMPREHENSIVE TEST RESULTS (N = 526)")
    print("=" * 85)
    print(f"Accuracy:     {comp_metrics['accuracy']:.2f}%")
    print(f"Macro-F1:     {comp_metrics['macro_f1']:.2f}%")
    print(f"ROC-AUC:      {comp_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:       {comp_metrics['pr_auc']:.4f}")
    print(f"Sensitivity:  {comp_metrics['recall_sensitivity']:.2f}%")
    print(f"Specificity:  {comp_metrics['specificity']:.2f}%")
    print("=" * 85)

    # 8. SAVE PREDICTIONS CSV
    df_preds = primary_test_df[['case_id', 'patient_id', 'abnormality_type', 'pathology', 'image_view', 'label']].copy()
    df_preds.rename(columns={'label': 'target'}, inplace=True)
    df_preds['binary_logit'] = primary_bin_logits
    df_preds['prob_raw'] = primary_raw_probs
    df_preds['prob_calibrated'] = primary_cal_probs
    df_preds['pred'] = primary_y_pred
    df_preds['is_correct'] = (df_preds['target'] == df_preds['pred']).astype(int)
    
    def get_err_type(row):
        t, p = row['target'], row['pred']
        if t == 1 and p == 1: return 'TRUE_POSITIVE'
        elif t == 0 and p == 0: return 'TRUE_NEGATIVE'
        elif t == 0 and p == 1: return 'FALSE_POSITIVE'
        else: return 'FALSE_NEGATIVE'
        
    df_preds['classification_type'] = df_preds.apply(get_err_type, axis=1)
    df_preds.to_csv(os.path.join(results_dir, "v9_phase5_final_test_predictions.csv"), index=False)

    # Save Metrics JSON
    metrics_bundle = {
        "primary_cohort_n525": primary_metrics,
        "comprehensive_cohort_n526": comp_metrics,
        "frozen_parameters": {
            "temperature_T": FROZEN_TEMPERATURE,
            "threshold": FROZEN_THRESHOLD,
            "k_proposals": 5,
            "arm_type": "Arm_G"
        }
    }
    with open(os.path.join(results_dir, "v9_phase5_final_test_metrics.json"), 'w') as f:
        json.dump(metrics_bundle, f, indent=2)

    # 9. PATIENT-CLUSTERED BOOTSTRAP (B = 1000)
    print("\n--- 7. Computing Patient-Clustered Bootstrap CIs (B = 1000) ---")
    boot_res = run_patient_clustered_bootstrap(df_preds, prob_col='prob_calibrated', n_bootstraps=1000, seed=SEED)
    
    boot_rows = []
    for mk in ['accuracy', 'macro_f1', 'f1_binary', 'roc_auc', 'pr_auc', 'recall_sensitivity', 'specificity', 'precision', 'brier_score', 'ece']:
        boot_rows.append({
            'Metric': mk.upper(),
            'Point Estimate': primary_metrics[mk],
            'Bootstrap Mean': boot_res[mk]['mean'],
            '95% CI Lower': boot_res[mk]['ci_lower'],
            '95% CI Upper': boot_res[mk]['ci_upper'],
            '95% CI Format': boot_res[mk]['ci_str']
        })
    pd.DataFrame(boot_rows).to_csv(os.path.join(results_dir, "v9_phase5_final_test_bootstrap_ci.csv"), index=False)

    # 10. SUBGROUP ANALYSES
    print("\n--- 8. Computing Subgroup Performance ---")
    subgroups = {
        'Overall Primary': primary_test_df,
        'Type: Mass': primary_test_df[primary_test_df['abnormality_type'] == 'MASS'],
        'Type: Calcification': primary_test_df[primary_test_df['abnormality_type'] == 'CALCIFICATION'],
        'View: CC': primary_test_df[primary_test_df['image_view'] == 'CC'],
        'View: MLO': primary_test_df[primary_test_df['image_view'] == 'MLO'],
        'Pathology: Benign': primary_test_df[primary_test_df['pathology'] == 'BENIGN'],
        'Pathology: Malignant': primary_test_df[primary_test_df['pathology'] == 'MALIGNANT']
    }

    subgroup_metrics_dict = {}
    for sg_name, sg_df in subgroups.items():
        case_ids = set(sg_df['case_id'].values)
        sub_p = df_preds[df_preds['case_id'].isin(case_ids)]
        m = compute_all_metrics(sub_p['target'].values, sub_p['prob_calibrated'].values)
        subgroup_metrics_dict[sg_name] = {
            'sample_count': len(sub_p),
            'positive_count': int(np.sum(sub_p['target'].values)),
            'metrics': m
        }
        print(f"  {sg_name:<22} (N={len(sub_p):<3}) | Acc: {m['accuracy']:>5.2f}% | Macro-F1: {m['macro_f1']:>5.2f}% | ROC-AUC: {m['roc_auc']:.4f} | Sens: {m['recall_sensitivity']:>5.2f}% | Spec: {m['specificity']:>5.2f}% | ECE: {m['ece']:>5.2f}%")

    with open(os.path.join(results_dir, "v9_phase5_final_test_subgroup_metrics.json"), 'w') as f:
        json.dump(subgroup_metrics_dict, f, indent=2)

    # 11. CALIBRATION REPORT ARTIFACT
    calib_summary = {
        "frozen_temperature": FROZEN_TEMPERATURE,
        "primary_brier_score": primary_metrics['brier_score'],
        "primary_ece": primary_metrics['ece'],
        "primary_nll": primary_metrics['nll'],
        "mean_predicted_confidence": primary_metrics['mean_confidence'],
        "empirical_positive_rate": primary_metrics['empirical_positive_rate'],
        "bin_details": primary_metrics['bin_details']
    }
    with open(os.path.join(results_dir, "v9_phase5_final_test_calibration.json"), 'w') as f:
        json.dump(calib_summary, f, indent=2)

    # 12. ERROR ANALYSIS ARTIFACT
    print("\n--- 9. Analyzing Test Classification Errors ---")
    fn_cases = df_preds[df_preds['classification_type'] == 'FALSE_NEGATIVE']
    fp_cases = df_preds[df_preds['classification_type'] == 'FALSE_POSITIVE']
    print(f"  Total Errors: {len(fn_cases) + len(fp_cases)} / 525 (False Negatives: {len(fn_cases)}, False Positives: {len(fp_cases)})")

    df_preds.to_csv(os.path.join(results_dir, "v9_phase5_final_test_error_analysis.csv"), index=False)

    # 13. AUDIT JSONS
    with open(os.path.join(results_dir, "v9_phase5_final_test_governance_audit.json"), 'w') as f:
        json.dump({
            "phase": "V9 Phase 5 Final Test Evaluation",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "governance_status": "PASSED",
            "primary_test_n": 525,
            "comprehensive_test_n": 526,
            "patient_overlap_train_test": len(train_pts.intersection(test_pts)),
            "patient_overlap_val_test": len(val_pts.intersection(test_pts)),
            "v9_arm_g_md5": arm_g_md5,
            "v6_ukan_md5": u_md5,
            "frozen_temperature": FROZEN_TEMPERATURE,
            "frozen_threshold": FROZEN_THRESHOLD,
            "tta_enabled": False,
            "overall_assessment": "COMPLETED"
        }, f, indent=2)

    with open(os.path.join(results_dir, "v9_phase5_final_test_checkpoint_integrity.json"), 'w') as f:
        json.dump({
            "locked_model": "Arm G (K=5 + Attention + Full Mammo + Calc Patch + Context ROI)",
            "checkpoint_path": arm_g_path,
            "md5_hash": arm_g_md5,
            "md5_verified": True,
            "status": "FINAL_TEST_VERIFIED"
        }, f, indent=2)

    # 14. GENERATE PUBLICATION-QUALITY FIGURES
    print("\n--- 10. Generating Publication-Quality Figures ---")

    # Figure 1: Confusion Matrix
    cm_arr = np.array([[primary_metrics['tn'], primary_metrics['fp']],
                       [primary_metrics['fn'], primary_metrics['tp']]])
    plt.figure(figsize=(5.5, 4.5), dpi=150)
    plt.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"V9 Phase 5 Final Test Confusion Matrix (N=525)\nAccuracy = {primary_metrics['accuracy']:.2f}% | Macro-F1 = {primary_metrics['macro_f1']:.2f}%")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Benign', 'Malignant'])
    plt.yticks(tick_marks, ['Benign', 'Malignant'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm_arr[i, j], 'd'), horizontalalignment="center",
                     color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black", fontsize=12, weight='bold')
    plt.ylabel('True Ground Truth')
    plt.xlabel('Predicted Class (Threshold = 0.50)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "phase5_confusion_matrix.png"))
    plt.close()

    # Figure 2: ROC Curve
    fpr, tpr, _ = sk_metrics.roc_curve(primary_y_true, primary_cal_probs)
    plt.figure(figsize=(6, 5), dpi=150)
    plt.plot(fpr, tpr, color='crimson', lw=2.2, label=f'V9 Arm G ROC (AUC = {primary_metrics["roc_auc"]:.4f} {boot_res["roc_auc"]["ci_str"]})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('V9 Final Locked Test ROC Curve (N=525)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "phase5_roc_curve.png"))
    plt.close()

    # Figure 3: PR Curve
    prec_vals, rec_vals, _ = sk_metrics.precision_recall_curve(primary_y_true, primary_cal_probs)
    plt.figure(figsize=(6, 5), dpi=150)
    plt.plot(rec_vals, prec_vals, color='darkorange', lw=2.2, label=f'V9 Arm G PR (AUC = {primary_metrics["pr_auc"]:.4f} {boot_res["pr_auc"]["ci_str"]})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title('V9 Final Locked Test Precision-Recall Curve (N=525)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "phase5_pr_curve.png"))
    plt.close()

    # Figure 4: Calibration Curve & Reliability Diagram
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    b_dets = primary_metrics['bin_details']
    confs = [b['avg_confidence'] for b in b_dets if b['count'] > 0]
    accs = [b['empirical_accuracy'] for b in b_dets if b['count'] > 0]
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration (y=x)')
    ax.plot(confs, accs, marker='o', lw=2, color='forestgreen', label=f'V9 Calibrated (T=1.2128, ECE={primary_metrics["ece"]:.2f}%)')
    ax.set_xlabel('Mean Predicted Confidence')
    ax.set_ylabel('Empirical Positive Fraction')
    ax.set_title('V9 Final Locked Test Reliability Diagram (15 Bins)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "phase5_calibration.png"))
    plt.close()

    # Figure 5: Confidence Histogram
    plt.figure(figsize=(6, 4.5), dpi=150)
    plt.hist(primary_cal_probs, bins=15, range=(0, 1), edgecolor='black', color='teal', alpha=0.75)
    plt.xlabel('Calibrated Malignant Confidence')
    plt.ylabel('Scan Count')
    plt.title(f'V9 Final Test Confidence Distribution (Mean = {primary_metrics["mean_confidence"]:.2f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "phase5_confidence_histogram.png"))
    plt.close()

    # Figure 6: Subgroup Comparison Bar Chart
    sg_keys = ['Overall Primary', 'Type: Mass', 'Type: Calcification', 'View: CC', 'View: MLO']
    f1s = [subgroup_metrics_dict[k]['metrics']['macro_f1'] for k in sg_keys]
    aucs = [subgroup_metrics_dict[k]['metrics']['roc_auc'] * 100.0 for k in sg_keys]
    sens = [subgroup_metrics_dict[k]['metrics']['recall_sensitivity'] for k in sg_keys]
    specs = [subgroup_metrics_dict[k]['metrics']['specificity'] for k in sg_keys]

    x = np.arange(len(sg_keys))
    w = 0.20
    plt.figure(figsize=(9, 4.5), dpi=150)
    plt.bar(x - 1.5*w, f1s, w, label='Macro-F1 (%)', color='royalblue')
    plt.bar(x - 0.5*w, aucs, w, label='ROC-AUC (x100)', color='darkorange')
    plt.bar(x + 0.5*w, sens, w, label='Sensitivity (%)', color='forestgreen')
    plt.bar(x + 1.5*w, specs, w, label='Specificity (%)', color='crimson')
    plt.xticks(x, [k.replace('Type: ', '').replace('View: ', '').replace('Primary', '') for k in sg_keys])
    plt.ylabel('Score (%)')
    plt.title('V9 Final Locked Test Subgroup Performance Comparison')
    plt.ylim([50, 100])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "phase5_subgroup_comparison.png"))
    plt.close()

    # Figure 7: Error Analysis Breakdown
    err_counts = [primary_metrics['tn'], primary_metrics['tp'], primary_metrics['fp'], primary_metrics['fn']]
    err_labels = [f'TN: Benign Correct\n({primary_metrics["tn"]})',
                  f'TP: Malignant Correct\n({primary_metrics["tp"]})',
                  f'FP: Benign Mistake\n({primary_metrics["fp"]})',
                  f'FN: Malignant Mistake\n({primary_metrics["fn"]})']
    err_colors = ['seagreen', 'forestgreen', 'coral', 'crimson']

    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.bar(err_labels, err_counts, color=err_colors, edgecolor='black', alpha=0.85)
    plt.ylabel('Number of Cases')
    plt.title('V9 Phase 5 Final Test Error & Classification Breakdown (N=525)')
    for i, v in enumerate(err_counts):
        plt.text(i, v + 4, str(v), ha='center', weight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "phase5_error_analysis.png"))
    plt.close()

    total_el = time.time() - t_start
    print("\n" + "=" * 85)
    print(f"V9 PHASE 5 FINAL LOCKED TEST EVALUATION COMPLETED IN {total_el:.1f}s")
    print("=" * 85)


if __name__ == "__main__":
    run_phase5_final_test()
