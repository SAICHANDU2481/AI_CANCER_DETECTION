"""
================================================================================
Breast Cancer AI Analysis - Configuration & Constants (V8 Final Frozen Model)
================================================================================
Academic Project:
"Advanced Breast Cancer Diagnosis Using Wiener Filtering, Vision Transformer,
and U-KAN-Based Medical Image Analysis"

GOVERNANCE:
- FINAL REPORTED MODEL: V8 (Phase 2 Arm F: Dual-Resolution U-KAN K=3 Proposals + Last 4 ViT Blocks Fine-Tuned).
- Locked V8 Test Benchmark (525 scans / 235 patients) is the PRIMARY reported test result.
- V9 is retained in the research record as a documented experimental extension.
- Cryptographic verification enforced for all model checkpoints.
- AI-assisted research prototype ONLY. NOT a clinical diagnostic system.
================================================================================
"""

import os
import hashlib
from typing import Dict, Any, List, Tuple

# -----------------------------------------------------------------------------
# Base Directories & System Paths
# -----------------------------------------------------------------------------
APP_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))

PROJECT_ROOTS = [
    PROJECT_ROOT,
    r"C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION",
    r"C:\Users\sys\Desktop\AI_CANCER_DETECTION"
]

def resolve_path(relative_or_abs_path: str) -> str:
    """Resolves path across primary OneDrive workspace and secondary local Desktop."""
    if os.path.isabs(relative_or_abs_path) and os.path.exists(relative_or_abs_path):
        return relative_or_abs_path
    
    # Try relative to current project root
    candidate = os.path.join(PROJECT_ROOT, relative_or_abs_path)
    if os.path.exists(candidate):
        return candidate
        
    for r in PROJECT_ROOTS:
        cand = os.path.join(r, relative_or_abs_path)
        if os.path.exists(cand):
            return cand
            
    return relative_or_abs_path

# -----------------------------------------------------------------------------
# Protected Model Checkpoints & Cryptographic MD5 Hashes
# -----------------------------------------------------------------------------
UKAN_V6_PATH = resolve_path(os.path.join("models", "final_v8", "ukan_v6_final.pth"))
if not os.path.exists(UKAN_V6_PATH):
    UKAN_V6_PATH = resolve_path(os.path.join("models", "v6", "segmentation", "final", "ukan_v6_arm_g_best.pth"))

VIT_V8_PATH = resolve_path(os.path.join("models", "final_v8", "vit_v8_final.pth"))
if not os.path.exists(VIT_V8_PATH):
    VIT_V8_PATH = resolve_path(os.path.join("models", "v8", "classification", "vit_v8_phase2_arm_f_best.pth"))

# Backward compatibility alias
VIT_V6_PATH = resolve_path(os.path.join("models", "v6", "classification", "vit_v6_arm_c_best.pth"))

EXPECTED_MD5 = {
    'ukan_v6': '68cfc38cfbddd93d6e4a6e692fffbb34',
    'vit_v8': '882313ce67adca01bffa699a180719c4',
    'vit_v6': '508250cfc1c07af1ea7d81cc2e5a7418'
}

def verify_file_md5(filepath: str, expected_hash: str) -> Tuple[bool, str]:
    """Computes MD5 hash of file and returns verification status."""
    if not os.path.exists(filepath):
        return False, "FILE_NOT_FOUND"
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096 * 1024), b""):
            h.update(chunk)
    actual_hash = h.hexdigest().lower()
    return (actual_hash == expected_hash.lower()), actual_hash

# -----------------------------------------------------------------------------
# Image Resolution & Model Dimensions
# -----------------------------------------------------------------------------
SEGMENTATION_INPUT_SIZE = (768, 768)
SEGMENTATION_WORKING_SIZE = (512, 512)
CLASSIFIER_CROP_SIZE = (224, 224)
PRIMARY_ROI_SCALES = {'local': 1.2, 'medium': 1.8, 'large': 2.5, 'context': 4.0}
TOP_K_PROPOSALS = 3
SEGMENTATION_THRESHOLD = 0.15

# -----------------------------------------------------------------------------
# Locked V8 Held-Out Independent Test Benchmark (FINAL REPORTED MODEL)
# -----------------------------------------------------------------------------
LOCKED_V8_TEST_METRICS = {
    "title": "Locked V8 Independent Held-Out Test Benchmark (FINAL REPORTED MODEL)",
    "description": "Formally locked evaluation on 525 independent test scans from 235 quarantined patients. Zero patient leakage.",
    "model_name": "V8 (Phase 2 Arm F: Dual-Resolution U-KAN K=3 Proposals + Last 4 ViT Blocks Tuned)",
    "status": "FROZEN",
    "n_scans": 525,
    "n_patients": 235,
    "patient_leakage": 0,
    "accuracy": 0.6457,
    "precision": 0.5569,
    "recall": 0.6605,
    "specificity": 0.6355,
    "binary_f1": 0.6043,
    "macro_f1": 0.6418,
    "roc_auc": 0.6947,
    "pr_auc": 0.5857,
    "brier_score": 0.3143,
    "ece": 0.2987,
    "confusion_matrix": {
        "tn": 197,
        "fp": 113,
        "fn": 73,
        "tp": 142
    },
    "ci_95": {
        "accuracy": "[59.06% - 69.79%]",
        "precision": "[47.88% - 63.27%]",
        "recall": "[58.85% - 72.93%]",
        "specificity": "[54.58% - 71.56%]",
        "binary_f1": "[54.03% - 66.09%]",
        "macro_f1": "[58.80% - 69.01%]",
        "roc_auc": "[0.6309 - 0.7474]",
        "pr_auc": "[0.5059 - 0.6723]"
    }
}

# Alias for compatibility
LOCKED_V6_TEST_METRICS = {
    "title": "Locked V6 Historical Test Benchmark",
    "description": "Evaluation on 526 independent test scans from 235 quarantined patients.",
    "n_scans": 526,
    "n_patients": 235,
    "patient_leakage": 0,
    "accuracy": 0.6445,
    "precision": 0.5538,
    "recall": 0.6698,
    "specificity": 0.6270,
    "binary_f1": 0.6063,
    "macro_f1": 0.6411,
    "roc_auc": 0.7087,
    "pr_auc": 0.5975,
    "confusion_matrix": {
        "tn": 195,
        "fp": 116,
        "fn": 71,
        "tp": 144
    }
}

# -----------------------------------------------------------------------------
# Comparative Evaluation Across Generations
# -----------------------------------------------------------------------------
LOCKED_MODEL_COMPARISON_TABLE = [
    {
        "model": "V5 (Arm I)",
        "accuracy": 65.02,
        "macro_f1": 64.73,
        "roc_auc": 0.6988,
        "pr_auc": 0.5916,
        "sensitivity": 68.37,
        "specificity": 62.70,
        "status": "Historical Baseline"
    },
    {
        "model": "V6 (Arm C)",
        "accuracy": 64.45,
        "macro_f1": 64.11,
        "roc_auc": 0.7087,
        "pr_auc": 0.5975,
        "sensitivity": 66.98,
        "specificity": 62.70,
        "status": "Historical Benchmark"
    },
    {
        "model": "V8 (Arm F)",
        "accuracy": 64.57,
        "macro_f1": 64.18,
        "roc_auc": 0.6947,
        "pr_auc": 0.5857,
        "sensitivity": 66.05,
        "specificity": 63.55,
        "status": "FINAL REPORTED MODEL (Frozen)"
    },
    {
        "model": "V9 (Arm G)",
        "accuracy": 62.86,
        "macro_f1": 62.86,
        "roc_auc": 0.7046,
        "pr_auc": 0.6129,
        "sensitivity": 76.28,
        "specificity": 53.55,
        "status": "Documented Experimental Extension"
    }
]

# -----------------------------------------------------------------------------
# CBIS-DDSM Dataset Cohort Statistics
# -----------------------------------------------------------------------------
DATASET_STATS = {
    "total_records": 3568,
    "total_patients": 1566,
    "splits": {
        "train": {"scans": 2501, "patients": 1096, "pct": "70.1%"},
        "validation": {"scans": 541, "patients": 235, "pct": "15.2%"},
        "test": {"scans": 525, "patients": 235, "pct": "14.7%"}
    },
    "patient_overlap": 0,
    "class_distribution": {
        "benign": 2111,
        "malignant": 1457,
        "benign_pct": "59.16%",
        "malignant_pct": "40.84%"
    },
    "abnormality_types": {
        "calcification": 1780,
        "mass": 1788
    },
    "modality": "Screening & Diagnostic Full-Field Digital Mammography (FFDM)",
    "views": ["CC (Craniocaudal)", "MLO (Mediolateral Oblique)"]
}

# -----------------------------------------------------------------------------
# Curated Benchmark Demo Cases
# -----------------------------------------------------------------------------
DEMO_SCANS = {
    "Select a preloaded clinical benchmark case...": None,
    "Case 1: MASS_P_00001_LEFT_CC (Malignant Spiculated Mass)": {
        "type": "Mass",
        "ground_truth": "MALIGNANT",
        "image_path": r"C:\Users\sys\Desktop\Capstone\datasets\jpeg\1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515\1-211.jpg",
        "alt_path": r"C:\Users\sys\OneDrive\Desktop\Capstone\datasets\jpeg\1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515\1-211.jpg",
        "desc": "High-density irregular mass with radiating spiculation and parenchymal architectural distortion."
    },
    "Case 2: MASS_P_00004_LEFT_CC (Benign Circumscribed Mass)": {
        "type": "Mass",
        "ground_truth": "BENIGN",
        "image_path": r"C:\Users\sys\Desktop\Capstone\datasets\jpeg\1.3.6.1.4.1.9590.100.1.2.89180046211022531834352631483669346540\1-250.jpg",
        "alt_path": r"C:\Users\sys\OneDrive\Desktop\Capstone\datasets\jpeg\1.3.6.1.4.1.9590.100.1.2.89180046211022531834352631483669346540\1-250.jpg",
        "desc": "Well-circumscribed oval mass with smooth boundaries, classic benign fibroadenoma appearance."
    },
    "Case 3: CALC_P_00005_RIGHT_CC (Malignant Microcalcifications)": {
        "type": "Calcification",
        "ground_truth": "MALIGNANT",
        "image_path": r"C:\Users\sys\Desktop\Capstone\datasets\jpeg\1.3.6.1.4.1.9590.100.1.2.47414316010368386519740343172775938548\1-188.jpg",
        "alt_path": r"C:\Users\sys\OneDrive\Desktop\Capstone\datasets\jpeg\1.3.6.1.4.1.9590.100.1.2.47414316010368386519740343172775938548\1-188.jpg",
        "desc": "Clustered fine pleomorphic microcalcifications with linear ductal branching distribution."
    },
    "Case 4: CALC_P_00011_LEFT_CC (Benign Punctate Calcifications)": {
        "type": "Calcification",
        "ground_truth": "BENIGN",
        "image_path": r"C:\Users\sys\Desktop\Capstone\datasets\jpeg\1.3.6.1.4.1.9590.100.1.2.389208391412680785736990872734216560014\1-198.jpg",
        "alt_path": r"C:\Users\sys\OneDrive\Desktop\Capstone\datasets\jpeg\1.3.6.1.4.1.9590.100.1.2.389208391412680785736990872734216560014\1-198.jpg",
        "desc": "Scattered round punctate microcalcifications consistent with benign fibrocystic/adenosis changes."
    }
}

# -----------------------------------------------------------------------------
# Research & Safety Disclaimer
# -----------------------------------------------------------------------------
SAFETY_DISCLAIMER_SHORT = (
    "AI-Assisted Breast Cancer Research Prototype only. This system is not a clinical diagnostic tool. "
    "Predictions are AI-assisted research outputs and must never replace evaluation by a qualified medical specialist."
)

SAFETY_DISCLAIMER_FULL = (
    "NOTICE: This web platform is an experimental academic prototype developed exclusively "
    "for peer-reviewed machine learning research and computer-assisted image analysis on the CBIS-DDSM dataset. "
    "It is NOT an FDA-approved or CE-marked medical device. All localized segmentation boundaries, "
    "risk likelihood estimates, and attention saliency heatmaps are algorithmic research artifacts "
    "and MUST NOT be utilized for medical triage, clinical decision-making, or treatment planning."
)
