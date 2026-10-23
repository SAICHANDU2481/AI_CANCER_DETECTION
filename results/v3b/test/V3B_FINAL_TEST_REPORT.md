# V3-B Final Locked Test Evaluation Report

## 1. Executive Summary & Governance
- **Research Experiment:** V3-B Cross-Attention Multi-Scale Machine-ROI ViT Classification
- **Upstream Segmentor:** Frozen V2-B MultiScaleUKAN (`models/v2b/segmentation/ukan_v2b_best.pth`, MD5: `a9a18b540cfd23114313b3e6d4276595`)
- **Evaluated Checkpoint:** `models/v3b/checkpoints/vit_v3b_best.pth` (MD5: `4dbe5ae7ef9362e44a6ab2472457801e`)
- **Locked Test Cohort:** 525 valid mammography scans from 235 strictly isolated patients (310 Benign, 215 Malignant)
- **Zero-Leakage Guarantee:** Zero patient overlap with train/validation sets; deterministic single-pass inference (no TTA, fixed threshold 0.50).

---

## 2. Overall Primary Test Metrics (N=525 Scans / 235 Patients)

| Metric | Point Estimate | 95% Bootstrap Confidence Interval |
|---|:---:|:---:|
| **Test Accuracy** | **64.57%** | [60.57%, 68.57%] |
| **Macro-F1** | **62.44%** | [58.15%, 66.36%] |
| **Binary F1 (Malignant)** | **53.50%** | — |
| **ROC-AUC** | **0.6579** | [0.6072, 0.7025] |
| **PR-AUC** | **0.5452** | [0.4799, 0.6145] |
| **Malignant Recall (Sensitivity)** | **49.77%** (107/215) | [43.05%, 56.37%] |
| **Specificity (Benign Recall)** | **74.84%** (232/310) | [70.10%, 79.52%] |
| **Malignant Precision (PPV)** | **57.84%** (107/185) | — |
| **Benign Precision (NPV)** | **68.24%** (232/340) | — |

### Confusion Matrix Counts:
- **True Positives (TP):** 107
- **True Negatives (TN):** 232
- **False Positives (FP):** 78
- **False Negatives (FN):** 108

---

## 3. Subgroup Performance Analysis

| Category | Subgroup | Count | Accuracy_% | Macro_F1_% | ROC_AUC | PR_AUC | Malignant_Recall_% | Specificity_% | Mean_Dice | Mean_IoU |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Abnormality_Type | MASS | 267 | 58.8 | 58.31 | 0.6173 | 0.5788 | 49.61 | 67.39 | 0.3989 | 0.3098 |
| Abnormality_Type | CALCIFICATION | 258 | 70.54 | 65.81 | 0.6815 | 0.5163 | 50.0 | 80.81 | 0.0485 | 0.0314 |
| ROI_Source | machine_roi | 380 | 63.16 | 62.01 | 0.6583 | 0.6158 | 50.0 | 74.27 | 0.3132 | 0.239 |
| ROI_Source | fallback | 145 | 68.28 | 61.98 | 0.6452 | 0.3613 | 48.78 | 75.96 | 0.0 | 0.0 |
| ROI_Area_Bin | Small (<2500 px) | 394 | 65.74 | 62.67 | 0.6754 | 0.5191 | 50.69 | 74.4 | 0.2078 | 0.1626 |
| ROI_Area_Bin | Medium (2500-10000 px) | 131 | 61.07 | 60.74 | 0.6246 | 0.6574 | 47.89 | 76.67 | 0.2835 | 0.2043 |

---

## 4. Comprehensive Error Taxonomy & False Negative Root Causes

### Error Taxonomy Across All 525 Test Cases:
| Error_Taxonomy_Class | Count | Percentage_% |
| --- | :---: | :---: |
| A. Severe Segmentation Failure | 180 | 34.29 |
| C. Fallback ROI | 145 | 27.62 |
| E. Good Segmentation + Correct Classification | 83 | 15.81 |
| B. Partial Lesion Capture | 68 | 12.95 |
| D. Good Segmentation but Classifier Failure | 49 | 9.33 |

### Root-Cause Breakdown for Malignant False Negatives (Total FN = 108):
| FN_Root_Cause | Count | Percentage_of_FN_% |
| --- | :---: | :---: |
| A. Severe Segmentation Failure | 36 | 33.33 |
| D. Good Segmentation but Classifier Failure | 28 | 25.93 |
| B. Partial Lesion Capture | 23 | 21.3 |
| C. Fallback ROI | 21 | 19.44 |

---

## 5. Artifact Verification & Checksum Log
- Checkpoint evaluated: `models/v3b/checkpoints/vit_v3b_best.pth` (MD5: `4dbe5ae7ef9362e44a6ab2472457801e`)
- Test predictions saved: `results/v3b/test/v3b_test_predictions.csv`
- Test metrics saved: `results/v3b/test/v3b_test_metrics.json`
- Figures generated:
  - Confusion matrix: `results/v3b/test/v3b_confusion_matrix.png`
  - ROC curve: `results/v3b/test/v3b_roc_curve.png`
  - PR curve: `results/v3b/test/v3b_pr_curve.png`

> [!NOTE]
> **Research Context:** All results represent AI-assisted medical image analysis research predictions evaluated on the benchmark DDSM/CBIS-DDSM dataset under strict patient-level quarantine.
