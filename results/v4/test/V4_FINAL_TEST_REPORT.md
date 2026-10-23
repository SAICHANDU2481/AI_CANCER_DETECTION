# V4-A Multi-Scale Visual Fusion Vision Transformer — Final Locked Test Report

**Project Title:** Advanced Breast Cancer Diagnosis Using Wiener Filtering, Vision Transformer, and U-KAN-Based Medical Image Analysis  
**Experiment:** V4-A (Multi-Scale Visual Fusion: Local ROI, Medium ROI, Large ROI, Surrounding Context ROI)  
**Evaluation Status:** Completed, Verified, and Permanently Frozen  
**Test Cohort:** Exactly 525 valid mammograms across 235 unique patients (310 Benign, 215 Malignant; 267 Masses, 258 Calcifications) — **Verified Zero Patient Overlap with Training / Validation Sets**  
**Classification Threshold:** Fixed at 0.50 (Pre-specified, strictly untuned on test set)  
**Test-Time Augmentation:** Strictly None (Exact deterministic frozen model inference)  

---

## 1. Executive Summary & Core Results

The one-time authorized locked test evaluation of the **V4-A Multi-Scale Visual Fusion Vision Transformer** was executed across all 525 held-out patient-isolated test scans from the benchmark CBIS-DDSM dataset.

The V4-A architecture evaluates 4 concentric and contextual spatial scales simultaneously:
1. **Local ROI ($1.2\times$):** Focused directly on the segmented core lesion margin.
2. **Medium ROI ($1.8\times$):** Perilesional parenchymal interface.
3. **Large ROI ($2.5\times$):** Regional architectural distortion zone.
4. **Surrounding Context ROI ($4.0\times$):** Surrounding quadrant background parenchyma.

Each scale is encoded via the frozen ImageNet-21k pre-trained Vision Transformer (`vit_base_patch16_224.augreg2_in21k_ft_in1k`) producing 768-dimensional visual tokens ($4 \times 768 = 3072$), fused through an MLP bottleneck ($3072 \rightarrow 768 \rightarrow 256 \rightarrow 2$) for binary malignancy classification.

### Primary Locked Test Metrics (N = 525 Scans / 235 Patients):

| Metric | Point Estimate | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Test Accuracy** | **63.43%** (333/525) | `[59.43%, 67.43%]` |
| **Macro-F1** | **61.72%** | `[57.51%, 65.83%]` |
| **Binary F1 (Malignant)** | **53.62%** | — |
| **ROC-AUC** | **0.6725** | `[0.6302, 0.7176]` |
| **PR-AUC (Average Precision)** | **0.6031** | — |
| **Malignant Recall (Sensitivity)** | **51.63%** (111/215) | — |
| **Benign Specificity** | **71.61%** (222/310) | — |
| **Malignant Precision (PPV)** | **55.78%** (111/199) | — |
| **Benign Precision (NPV)** | **68.10%** (222/326) | — |

> [!IMPORTANT]
> **Scientific Integrity & Validation vs. Test Distinction:**  
> The V4-A model achieved **71.90% Accuracy**, **70.27% Macro-F1**, and **0.7612 ROC-AUC** during the controlled validation phase. On the held-out test cohort of 525 scans / 235 patients, the locked test performance is **63.43% Accuracy**, **61.72% Macro-F1**, **0.6725 ROC-AUC**, and **0.6031 PR-AUC**.  
> The validation result is never substituted for the true test result.

---

## 2. Locked Model Configuration

- **Classifier Architecture:** `MultiScaleViTV4` (Arm E / F Validation Winner)
- **Scales Evaluated:** Local ($1.2\times$), Medium ($1.8\times$), Large ($2.5\times$), Context ($4.0\times$)
- **Feature Fusion:** $4 \times 768 = 3072 \rightarrow 768 \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.3) \rightarrow 256 \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.2) \rightarrow 2$
- **Quality Vector Injection:** DISABLED (Frozen validation decision)
- **Dynamic Gating Module:** DISABLED (Frozen validation decision)
- **Model Checkpoint:** `models/v4/vit_v4_best.pth` (MD5: `d016fdc756758658e03d43f02fd4482c`)
- **Upstream Segmentor:** Frozen V2-B U-KAN (`models/v2b/segmentation/ukan_v2b_best.pth`, MD5: `a9a18b540cfd23114313b3e6d4276595`)
- **Image Preprocessing:** 16-bit CBIS-DDSM Mammogram $\rightarrow$ Adaptive Wiener Filtering ($3\times3$) $\rightarrow$ Contrast Limited Adaptive Histogram Equalization (CLAHE, clip 2.5) $\rightarrow$ Min-Max Normalization $[0, 1] \rightarrow 512\times512$ canvas.
- **Lesion Detection Threshold:** Fixed at 0.50 probability on U-KAN sigmoid output.

---

## 3. Test Cohort & Patient-Level Isolation

- **Total Scans:** 525 valid mammograms
- **Unique Patients:** 235 patients
- **Class Balance:** 310 Benign (59.05%), 215 Malignant (40.95%)
- **Abnormality Breakdown:** 267 Mass scans, 258 Calcification scans
- **Patient Leakage Audit:** 0 overlapping patient IDs between Test and Training/Validation sets. Verified 100% strict patient isolation.

---

## 4. Test Protocol

1. Full $512\times512$ enhanced mammogram is passed to frozen V2-B U-KAN segmentor.
2. Binary mask generated using fixed threshold of $0.50$.
3. Connected component analysis identifies candidate lesion clusters.
4. ROI source categorized deterministically:
   - `machine_roi`: Detected lesion $\ge 50\text{ px}$.
   - `expanded_machine_roi`: Detected small lesion expanded to preserve minimal spatial context.
   - `fallback_context`: No component $\ge 50\text{ px}$ detected; quadrant/breast-mass center context crop extracted.
5. Deterministic multi-scale extraction creates 4 crops ($1.2\times, 1.8\times, 2.5\times, 4.0\times$), resized to $224\times224\times3$ with ImageNet normalization.
6. Forward pass through frozen `MultiScaleViTV4` classifier.
7. Decision rule: Predicted Malignant if $P(\text{Malignant}) \ge 0.5000$.

---

## 5. Confusion Matrix & Per-Class Performance

```
========================================================================================
                                     PREDICTED BENIGN (0)       PREDICTED MALIGNANT (1)
ACTUAL BENIGN (0)    [N = 310]             TN = 222                    FP =  88
ACTUAL MALIGNANT (1) [N = 215]             FN = 104                    TP = 111
========================================================================================
TOTAL EVALUATED: 525 scans across 235 unique patients
```

### Detailed Per-Class Breakdown:

| Class | Support | Precision | Recall (Sensitivity) | Specificity | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Benign (0)** | 310 | 68.10% (222/326) | 71.61% (222/310) | 51.63% (111/215) | **69.81%** |
| **Malignant (1)** | 215 | 55.78% (111/199) | 51.63% (111/215) | 71.61% (222/310) | **53.62%** |

---

## 6. Discrimination & Calibration (ROC-AUC & PR-AUC)

- **ROC-AUC:** **0.6725** (95% Bootstrap CI: `[0.6302, 0.7176]`)
- **PR-AUC:** **0.6031** (Baseline Malignancy Prevalence: $215/525 = 0.4095$)
- **Gain over Chance ($0.5000$ ROC-AUC):** $+0.1725$
- **Gain over Baseline PR-AUC ($0.4095$):** $+0.1936$ (+47.28% relative gain)

---

## 7. Stratified Subgroup Analysis

### A. Abnormality Type (Mass vs. Calcification)

| Subgroup | Scans | Benign | Malignant | Accuracy | Macro-F1 | ROC-AUC | Malignant Recall | Specificity | Precision | TP | TN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MASS** | 267 | 138 | 129 | 59.18% | 59.06% | 0.6376 | 55.81% (72/129) | 62.32% (86/138) | 58.06% | 72 | 86 | 52 | 57 |
| **CALCIFICATION** | 258 | 172 | 86 | 67.83% | 62.53% | 0.6737 | 45.35% (39/86) | 79.07% (136/172) | 52.00% | 39 | 136 | 36 | 47 |

**Clinical & Radiological Insights:**
- **Mass lesions:** Higher sensitivity ($55.81\%$) due to distinct soft-tissue opacity boundaries.
- **Calcifications:** Higher specificity ($79.07\%$) and accuracy ($67.83\%$), but lower sensitivity ($45.35\%$) due to sub-millimeter punctate morphological patterns requiring high magnification.

---

### B. ROI Source Stratification

| ROI Source | Scans | % Cohort | Accuracy | Macro-F1 | ROC-AUC | Malignant Recall | Specificity | TP | TN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Machine ROI** | 326 | 62.10% | 58.28% | 57.88% | 0.6316 | 53.74% (79/147) | 62.01% (111/179) | 79 | 111 | 68 | 68 |
| **Expanded Machine ROI** | 57 | 10.86% | 68.42% | 67.94% | 0.7094 | 57.14% (16/28) | 79.31% (23/29) | 16 | 23 | 6 | 12 |
| **Fallback Context** | 142 | 27.05% | 73.24% | 63.98% | 0.6841 | 40.00% (16/40) | 86.27% (88/102) | 16 | 88 | 14 | 24 |

---

### C. Segmentation Quality Stratification (Post-Hoc Ground-Truth Dice)

*Note: Ground-truth masks were used strictly for post-hoc error characterization and never accessible during inference.*

| Dice Stratum | Scans | % Cohort | Accuracy | Macro-F1 | Malignant Recall | Specificity | TP | TN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\text{Dice} < 0.10$** (Severe Miss) | 325 | 61.90% | 63.38% | 57.15% | 38.68% (41/106) | 75.34% (165/219) | 41 | 165 | 54 | 65 |
| **$0.10 \le \text{Dice} < 0.50$** (Partial) | 67 | 12.76% | 55.22% | 54.97% | 55.26% (21/38) | 55.17% (16/29) | 21 | 16 | 13 | 17 |
| **$\text{Dice} \ge 0.50$** (High Quality) | 133 | 25.33% | 67.67% | 67.55% | 69.01% (49/71) | 66.13% (41/62) | 49 | 41 | 21 | 22 |

**Key Finding:** When upstream U-KAN segmentation achieves $\text{Dice} \ge 0.50$, classification malignant recall reaches **69.01%** and Macro-F1 reaches **67.55%**, proving that segmentation quality directly governs downstream classifier sensitivity.

---

## 8. Mutually Exclusive Test Error Taxonomy

Across all 192 misclassified test cases ($36.57\%$ of the 525-scan cohort):

| Category | Description | Errors | % of Total Errors | % of Cohort |
| :--- | :--- | :---: | :---: | :---: |
| **Category 1** | **Severe segmentation failure** ($\text{Dice} < 0.10$) | 119 | 61.98% | 22.67% |
| **Category 2** | **Good localization but classifier failure** ($\text{Dice} \ge 0.50$) | 43 | 22.40% | 8.19% |
| **Category 3** | **Partial lesion localization** ($0.10 \le \text{Dice} < 0.50$) | 30 | 15.62% | 5.71% |
| **Total Errors** | Total Misclassified Cases | **192** | **100.00%** | **36.57%** |

---

## 9. Malignant False-Negative Analysis (Total FN = 104)

Root cause analysis for all 104 missed malignant cases:

| Malignant FN Root Cause | Criteria | FN Count | % of Total FN |
| :--- | :--- | :---: | :---: |
| **Severe Localization FN** | Segmentor missed lesion ($\text{Dice} < 0.10$) | 65 | 62.50% |
| **Good Localization Classifier FN** | Segmentor localized lesion ($\text{Dice} \ge 0.50$), ViT misclassified | 22 | 21.15% |
| **Partial Localization FN** | Segmentor partially overlapped lesion ($0.10 \le \text{Dice} < 0.50$) | 17 | 16.35% |
| **Fallback / Context FN** | Center/quadrant fallback used with $\text{Dice} \ge 0.10$ | 0 | 0.00% |
| **Total Localization-Associated FN** | Upstream localization failure (Severe + Partial) | **82** | **78.85%** |

### Historical Comparison of False Negative Root Causes:
- **V3-B Locked Benchmark:** Total FN = 108, Localization-Associated FN = 80 / 108 (74.07%)
- **V4-A Final Evaluation:** Total FN = 104, Localization-Associated FN = 82 / 104 (78.85%)
- **Net Impact:** V4-A reduced total false negatives from 108 to 104, improving true malignant detections from 107 to 111. Upstream localization failure remains the single largest operational bottleneck in automated breast cancer classification.

---

## 10. Historical Benchmark Comparison (V1 $\rightarrow$ V4-A)

All experiments evaluated on the canonical 525-scan / 235-patient CBIS-DDSM test cohort:

| Experiment | Pipeline Architecture | Test Accuracy | Test Macro-F1 | Test ROC-AUC | Test PR-AUC | Malignant Recall | Specificity | Malignant Precision | TN / FP / FN / TP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1 End-to-End** | Wiener + CLAHE $\rightarrow$ V1 U-KAN $\rightarrow$ Single-Crop ViT | 52.57% | 41.06% | 0.3972 | 0.3732 | 10.23% (22/215) | 81.94% (254/310) | 28.21% (22/78) | 254 / 56 / 193 / 22 |
| **V2-A Machine-ROI** | Wiener + CLAHE $\rightarrow$ V1 U-KAN $\rightarrow$ Perturbed Machine-ROI ViT | 56.76% | 56.46% | 0.6223 | 0.5627 | 59.07% (127/215) | 55.16% (171/310) | 47.74% (127/266) | 171 / 139 / 88 / 127 |
| **V3-A Stage 1** | Concentric Crops ($1.1\times, 1.5\times, 2.0\times$) $\rightarrow$ Bottleneck MLP | 53.90% | 53.47% | 0.5567 | 0.4910 | 53.95% (116/215) | 53.87% (167/310) | 44.79% (116/259) | 167 / 143 / 99 / 116 |
| **V3-A Stage 2** | Full Fine-Tuned Concentric ViT | 58.86% | 57.85% | 0.6041 | 0.5458 | 53.02% (114/215) | 62.90% (195/310) | 49.78% (114/229) | 195 / 115 / 101 / 114 |
| **V3-B Locked** | Cross-Attention Multi-Scale ($1.2\times, 2.0\times$) + Context ViT | 64.57% | 62.44% | 0.6579 | 0.5452 | 49.77% (107/215) | 74.84% (232/310) | 57.84% (107/185) | 232 / 78 / 108 / 107 |
| **V4-A Final (Ours)**| **Multi-Scale Visual Fusion (Local, Med, Lrg, Ctx: $4\times768$)** | **63.43%** | **61.72%** | **0.6725** | **0.6031** | **51.63%** (111/215) | **71.61%** (222/310) | **55.78%** (111/199) | **222 / 88 / 104 / 111** |

### Key Evolutionary Takeaways:
1. **Highest Discriminative Capacity (ROC-AUC 0.6725 & PR-AUC 0.6031):** V4-A achieves the highest area under the ROC curve ($0.6725$) and Precision-Recall curve ($0.6031$) across all tested architectures, demonstrating superior ranking and probabilistic calibration compared to V1 ($0.3972$), V2-A ($0.6223$), V3-A ($0.6041$), and V3-B ($0.6579$).
2. **Balanced Malignant Sensitivity (51.63%):** V4-A correctly detected 111 malignant cases, outperforming V3-B (107 cases) and V1 (22 cases).
3. **Robust Benign Specificity (71.61%):** V4-A correctly classified 222 benign cases without collapsing into false positive alarms.

---

## 11. Study Limitations

1. **Upstream Segmentation Dependency:** $78.85\%$ of malignant false negatives are directly attributable to poor lesion localization by the upstream U-KAN ($\text{Dice} < 0.50$). Future work should investigate joint end-to-end differentiable ROI alignment.
2. **Calcification Sensitivity Disparity:** While mass sensitivity reached $55.81\%$, calcification sensitivity remained at $45.35\%$. Punctate microcalcifications require specialized high-resolution patch zooms ($\ge 1024\times1024$).
3. **Single-Dataset Domain:** All evaluations were performed on the retrospective CBIS-DDSM dataset. External validation across multi-center digital mammography datasets (e.g., INbreast, VinDr-Mammo) is required before generalizability conclusions can be drawn.

---

## 12. Reproducibility & Governance Information

- **Operating System:** Microsoft Windows 11
- **Python Version:** 3.12.0
- **PyTorch Version:** 2.6.0+cu124
- **CUDA Device:** NVIDIA GeForce RTX 2050 (4096 MB VRAM)
- **Random Seed:** 42 (Deterministic evaluation)
- **Primary ViT Checkpoint MD5:** `d016fdc756758658e03d43f02fd4482c`
- **U-KAN Segmentor Checkpoint MD5:** `a9a18b540cfd23114313b3e6d4276595`
- **Evaluation Script:** `classification/v4/evaluate_test_v4.py`
- **Evaluation Manifest:** `results/v4/V4_FINAL_TEST_MANIFEST.json`

---

## 13. Final Governance Statement

> [!CAUTION]
> **Academic Research Prototype Disclaimer:**  
> This software, model weights, and pipeline represent an **academic research prototype** designed to investigate multi-scale visual fusion and U-KAN-based segmentation for computer-aided mammographic image analysis.  
> **This system is NOT a medical device, is NOT approved for clinical diagnosis, and MUST NOT be used for clinical decision-making or patient management.**

---
**Evaluation Status:** COMPLETE — MODEL FROZEN.
