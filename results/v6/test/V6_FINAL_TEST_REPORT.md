# V6 Final Locked Test Evaluation Report
**Project:** Advanced Breast Cancer Diagnosis Using Wiener Filtering, Vision Transformer, and U-KAN-Based Medical Image Analysis  
**Protocol:** One-Time Final Held-Out Test Evaluation of Frozen V6 Research Pipeline  
**Date:** 2026-09-27 19:26:11  
**Evaluation Status:** **LOCKED & FINAL**

---

## 1. Executive Summary
This document presents the definitive, one-time test evaluation of the completely frozen **V6 Diagnostic Pipeline**. All hyperparameters, architecture configurations, and trained weights were frozen during Phase 4 validation prior to this evaluation.

- **Segmentation Model:** [`ukan_v6_arm_g_best.pth`](file:///c:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/models/v6/segmentation/final/ukan_v6_arm_g_best.pth) (MD5: `68cfc38cfbddd93d6e4a6e692fffbb34`)
- **Classification Model:** [`vit_v6_arm_c_best.pth`](file:///c:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/models/v6/classification/vit_v6_arm_c_best.pth) (MD5: `508250cfc1c07af1ea7d81cc2e5a7418`)
- **Test Dataset Cohort:** 526 total mammograms (311 Benign, 215 Malignant; 235 unique patients; 0 patient overlap with Train/Val).

---

## 2. Primary Test Results & 95% Confidence Intervals

| Metric | Measured Test Value | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | **64.45%** | [60.46%, 68.25%] |
| **Precision** | **55.38%** | [49.42%, 61.40%] |
| **Recall / Sensitivity** | **66.98%** | [60.38%, 73.61%] |
| **Specificity** | **62.70%** | [57.14%, 67.88%] |
| **Binary F1** | **60.63%** | [55.31%, 65.70%] |
| **Macro-F1** | **64.11%** | [59.82%, 67.93%] |
| **ROC-AUC** | **0.7087** | [0.6653, 0.7522] |
| **PR-AUC** | **0.5975** | [0.5338, 0.6707] |

### Confusion Matrix
- **True Positives (TP):** 144
- **False Positives (FP):** 116
- **True Negatives (TN):** 195
- **False Negatives (FN):** 71

$$\begin{pmatrix} \text{TN} = 195 & \text{FP} = 116 \\ \text{FN} = 71 & \text{TP} = 144 \end{pmatrix}$$

---

## 3. Historical Pipeline Evolution Comparison (Held-Out Test Set)

| Pipeline Version | Architecture & Approach | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | Malignant Recall | Specificity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1 Baseline** | ResNet Baseline + Basic Crops | 59.80% | 58.20% | 0.6310 | 0.5120 | 61.40% | 58.70% |
| **V2 Baseline** | ViT-Base + Single Scale ROI | 62.10% | 61.50% | 0.6650 | 0.5480 | 64.20% | 60.60% |
| **V3 Baseline** | Multi-Scale ViT (3 scales) | 63.50% | 63.00% | 0.6810 | 0.5690 | 66.50% | 61.40% |
| **V4 Baseline** | Concentric 4-Scale + Global Context | 64.26% | 63.88% | 0.6895 | 0.5794 | 67.44% | 62.06% |
| **V5 Arm-I Baseline** | V5 U-KAN + Top-3 + Calc Patch | 65.02% | 64.73% | 0.6988 | 0.5916 | 68.37% | 62.70% |
| **V6 Arm-C (Final Pipeline)** | **V6 U-KAN + Top-3 + Multi-Scale + Tuned ViT** | **64.45%** | **64.11%** | **0.7087** | **0.5975** | **66.98%** | **62.70%** |

---

## 4. Subgroup Performance Analysis

| Abnormality Subgroup | Total Scans | Accuracy (%) | Sensitivity / Recall (%) | Specificity (%) | Macro-F1 (%) | ROC-AUC | TP / FP / TN / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MASS** | 267 | **59.18%** | **68.99%** | **50.00%** | **58.95%** | **0.6427** | 89/69/69/40 |
| **CALCIFICATION** | 259 | **69.88%** | **63.95%** | **72.83%** | **67.44%** | **0.7611** | 55/47/126/31 |

---

## 5. Machine-ROI Quality on Held-Out Test Set
- **Machine ROI Success Rate:** **19.20%** (101/526 scans with machine-detected candidates)
- **Fallback Rate:** **80.80%** (425/526 scans)
- **Mean Test Dice:** **22.79%**
- **Mean Test IoU:** **16.31%**

---

## 6. Statistical Significance vs V5 Arm-I Reference
- **McNemar's Discordance Test:**
  - Cases correct in V6 only: **57**
  - Cases correct in V5 only: **60**
  - McNemar Statistic: **0.0342** ($p = 8.5331e-01$)
- **Wilcoxon Signed-Rank Test (Malignant Probability Calibration):**
  - Wilcoxon Statistic: **60070.00** ($p = 8.1261e-03$)

---

## 7. Error Analysis Breakdown (187 Total Misclassifications)

```
                      Error Category  Count  Percentage
                       Fallback Crop    156   83.422460
              Partial Lesion Capture     16    8.556150
         Severe Segmentation Failure     11    5.882353
Good Localization / Classifier Error      4    2.139037
```

### Malignant False Negatives Breakdown (71 Total FNs):
{
  "Fallback Crop": 63,
  "Partial Lesion Capture": 5,
  "Severe Segmentation Failure": 2,
  "Good Localization / Classifier Error": 1
}

---

## 8. Integrity Audit & Governance Verification
- **Patient Isolation:** 100% verified (0 patient overlap across Train, Val, and Test).
- **Test-Time Tuning:** 0 parameters tuned; threshold fixed at 0.50.
- **TTA:** 0 test-time augmentations used.
- **V6 U-KAN Checkpoint MD5:** `68cfc38cfbddd93d6e4a6e692fffbb34` (**100% MATCH**)
- **V6 ViT Arm-C Checkpoint MD5:** `508250cfc1c07af1ea7d81cc2e5a7418` (**100% MATCH**)
- **V5 Reference Checkpoints:** Verified untouched.

---

## 9. Research Prototype Disclaimer & Regulatory Notice
> [!NOTE]
> **Research Prototype Notice:** This software and pipeline represent an investigational academic research system developed for comparative evaluation of Wiener filtering, U-KAN medical image segmentation, and Vision Transformer classification. It is not certified as a medical device for standalone clinical diagnosis.
