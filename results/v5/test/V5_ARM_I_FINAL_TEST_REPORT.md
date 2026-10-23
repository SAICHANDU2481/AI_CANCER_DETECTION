# V5 Arm I Final Locked Test Evaluation

## 1. Model
- **Selected Model:** MultiScaleViTV5 (Arm I: Full-Mammogram + Top-3 Multi-Scale + Calcification Patch)
- **Checkpoint Path:** `models/v5/classification/vit_v5_arm_i_best.pth`
- **Checkpoint MD5:** `ec80689a43a906f89a15d7148cf7b05e`
- **Model Architecture:** ViT-Base Backbone (85.8M) + Multi-Scale RoI Head ($K=3$ candidates $\times$ 4 scales) + Global 224 Context + Calcification-Enhanced Patch ($4,608$-dimensional fusion)
- **Selection Rule:** Pre-specified full-system architecture achieving suite-highest Validation PR-AUC ($0.6540$) and Validation ROC-AUC ($0.7928$) in the 9-arm controlled validation ablation suite.

## 2. Test Dataset
- **Total Records:** 526
- **Unique Patients:** 235
- **Benign Cases:** 311
- **Malignant Cases:** 215
- **Primary Native ROI Count:** 525

## 3. Inference Configuration
- **K Proposals:** 3
- **ROI Scales:** Local (1.2x), Medium (1.8x), Large (2.5x), Context (4.0x)
- **Full Mammogram:** Enabled (224x224 global whole-breast canvas)
- **Calcification Patch:** Enabled (224x224 high-frequency microcalcification patch with deterministic fallback)
- **Classification Threshold:** Fixed at 0.50 (zero test-time tuning)
- **Test-Time Augmentation (TTA):** Disabled
- **Preprocessing:** Deterministic Wiener filtering + contrast enhancement + U-KAN V5 (768x768) feature proposal extraction

## 4. Primary Test Results

| Metric | Measured Value | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | **65.02%** | [61.02%, 69.01%] |
| **Precision** | **55.89%** | [50.00%, 61.83%] |
| **Recall / Sensitivity** | **68.37%** | [62.27%, 74.21%] |
| **Specificity** | **62.70%** | [57.05%, 68.05%] |
| **Binary F1** | **61.51%** | [56.19%, 66.67%] |
| **Macro-F1** | **64.73%** | [60.65%, 68.73%] |
| **ROC-AUC** | **0.6988** | [0.6518, 0.7435] |
| **PR-AUC** | **0.5916** | [0.5226, 0.6584] |

### Confusion Matrix
- **True Positives (TP):** 147
- **False Positives (FP):** 116
- **True Negatives (TN):** 195
- **False Negatives (FN):** 68

$$\begin{pmatrix} \text{TN} = 195 & \text{FP} = 116 \\ \text{FN} = 68 & \text{TP} = 147 \end{pmatrix}$$

## 5. Subgroup Results

| Subgroup | Scans | Accuracy (%) | Sensitivity (%) | Specificity (%) | Macro-F1 (%) | ROC-AUC | TP/FP/TN/FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MASS** | 267 | **60.30%** | **70.54%** | **50.72%** | **60.05%** | **0.6464** | 91/68/70/38 |
| **CALCIFICATION** | 259 | **69.88%** | **65.12%** | **72.25%** | **67.58%** | **0.7322** | 56/48/125/30 |

## 6. Post-Hoc Error Analysis
- **False Negatives ($FN = 68$):** {'MASS': 38, 'CALCIFICATION': 30}
- **False Positives ($FP = 116$):** {'MASS': 68, 'CALCIFICATION': 48}
- **Observations:** Microcalcification clusters embedded in dense glandular parenchyma exhibited higher specificity when supported by the enhanced high-frequency patch, while masses benefited strongly from multi-scale context margins.

## 7. Integrity Audit
- **Patient Leakage:** Zero patient overlap between train, val, and test splits.
- **Duplicate Records:** 0 duplicates detected.
- **Missing Predictions:** 0 missing predictions (526/526 evaluated).
- **NaN / Inf Checks:** 0 non-finite logits or probabilities.
- **Test-Time Tuning:** 0 threshold or parameter adjustments made on test data.
- **Arm I Checkpoint MD5:** `ec80689a43a906f89a15d7148cf7b05e` (100% verified match).

## 8. Limitations
Explicit statement: This software and model architecture represent a computational research prototype for medical image analysis and are not approved as a medical device for standalone clinical diagnosis.
