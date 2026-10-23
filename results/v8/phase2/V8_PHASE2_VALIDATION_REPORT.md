# V8 PHASE 2 — MULTI-CROP CLASSIFIER VALIDATION & STATISTICAL REPORT

---

## 1. Executive Summary

This report documents the scientific evaluation and statistical analysis of **V8 Phase 2: Multi-Scale RoI Extraction & Multi-Crop Classifier Training Suite**. Validation evaluations were conducted strictly on the patient-isolated **Validation split** comprising **541 mammographic scans across 235 unique patients** (330 benign, 211 malignant), with strict zero patient overlap with the training split.

All six predefined experimental arms (**Arm A $\rightarrow$ Arm B $\rightarrow$ Arm C $\rightarrow$ Arm D $\rightarrow$ Arm E2 $\rightarrow$ Arm F**) were evaluated using identical evaluation code, deterministic metric pipelines, and patient-clustered non-parametric bootstrap resampling ($B=1,000$ iterations).

Arm F produced the highest observed validation Macro-F1, accuracy, and ROC-AUC among the Phase 2 arms. All findings presented in this report represent validation-set observations only and should not be interpreted as final test-set performance.

---

## 2. Validation Protocol

- **Validation Cohort:** 541 full-field digital mammograms (FFDM) from 235 patients (CBIS-DDSM validation partition).
- **Patient Isolation:** Complete patient-level separation ($0$ patient overlap with training split).
- **Evaluation Mode:** Deterministic model evaluation (`torch.no_grad()`, `model.eval()`).
- **Input RoI Source:** Dual-resolution U-KAN multi-scale proposals (512 + 768 fusion) pre-extracted and verified during Phase 1.
- **Decision Threshold:** Fixed standard classification threshold ($p \ge 0.50$).
- **Primary Metric:** Macro-averaged F1 score across benign and malignant classes.
- **Secondary Metrics:** Overall Accuracy, ROC-AUC, PR-AUC, Sensitivity (Malignant Recall), Specificity (Benign Recall), Precision, Brier Score, and Expected Calibration Error (ECE).
- **Statistical Resampling:** Patient-clustered non-parametric bootstrap ($B=1,000$ resamples) to compute empirical 95% confidence intervals.

---

## 3. Complete Arm Comparison

*All metrics reported below are validation-only results evaluated on the 541 validation scans.*

| Arm | Configuration | Best Epoch | Accuracy (%) | Macro-F1 (%) | ROC-AUC | PR-AUC | Sensitivity (%) | Specificity (%) | Precision (%) | Brier Score | ECE (%) | TN | FP | FN | TP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm A** | Locked V6 Arm-C Reference | Historical | 71.90 | 71.05 | 0.7787 | 0.6912 | 68.25 | 74.24 | 65.48 | 0.1824 | 6.84 | — | — | — | — |
| **Arm B** | Arm E ($K=3$) + Frozen ViT | 7 | 72.09 | 71.81 | 0.7764 | 0.6504 | 79.62 | 67.27 | 60.87 | 0.1903 | 8.87 | 222 | 108 | 43 | 168 |
| **Arm C** | Arm E ($K=3$) + Fusion Head | 5 | 70.43 | 70.29 | 0.7587 | 0.6295 | 81.52 | 63.33 | 58.70 | 0.1980 | 9.30 | 209 | 121 | 39 | 172 |
| **Arm D** | Arm E ($K=3$) + Last 2 Blocks | 21 | 75.23 | 74.06 | 0.7935 | 0.6927 | 69.19 | 79.09 | 67.91 | 0.2132 | 18.73 | 261 | 69 | 65 | 146 |
| **Arm E2** | Arm E ($K=5$) + Last 2 Blocks | 4 | 72.09 | 71.59 | 0.7983 | 0.6810 | 75.36 | 70.00 | 61.63 | 0.1765 | 3.34 | 231 | 99 | 52 | 159 |
| **Arm F** | Arm E ($K=3$) + Last 4 Blocks | 14 | 75.97 | 75.36 | 0.8153 | 0.6903 | 77.25 | 75.15 | 66.53 | 0.2115 | 18.70 | 248 | 82 | 48 | 163 |

---

## 4. Accuracy, Macro-F1, ROC-AUC, and PR-AUC Comparison

Comparing the Phase 2 arms against the historical locked reference baseline (**Arm A**):

- **Arm F vs. Arm A:**
  - **Macro-F1:** 75.36% vs. 71.05% (**Difference: +4.31 percentage points**)
  - **Accuracy:** 75.97% vs. 71.90% (**Difference: +4.07 percentage points**)
  - **ROC-AUC:** 0.8153 vs. 0.7787 (**Difference: +0.0366**)
  - **PR-AUC:** 0.6903 vs. 0.6912 (**Difference: -0.0009**)
- **Arm D vs. Arm A:**
  - **Macro-F1:** 74.06% vs. 71.05% (**Difference: +3.01 percentage points**)
  - **Accuracy:** 75.23% vs. 71.90% (**Difference: +3.33 percentage points**)
  - **ROC-AUC:** 0.7935 vs. 0.7787 (**Difference: +0.0148**)
  - **PR-AUC:** 0.6927 vs. 0.6912 (**Difference: +0.0015**)
- **Arm E2 vs. Arm A:**
  - **Macro-F1:** 71.59% vs. 71.05% (**Difference: +0.54 percentage points**)
  - **Accuracy:** 72.09% vs. 71.90% (**Difference: +0.19 percentage points**)
  - **ROC-AUC:** 0.7983 vs. 0.7787 (**Difference: +0.0196**)
- **Arm B vs. Arm A:**
  - **Macro-F1:** 71.81% vs. 71.05% (**Difference: +0.76 percentage points**)
  - **Accuracy:** 72.09% vs. 71.90% (**Difference: +0.19 percentage points**)
- **Arm C vs. Arm A:**
  - **Macro-F1:** 70.29% vs. 71.05% (**Difference: -0.76 percentage points**)
  - **Accuracy:** 70.43% vs. 71.90% (**Difference: -1.47 percentage points**)

Arm F produced the highest observed validation Macro-F1, accuracy, and ROC-AUC among all Phase 2 arms.

---

## 5. Sensitivity and Specificity Trade-Offs

The ablation arms exhibit distinct operating characteristics across sensitivity and specificity:

1. **Arm C (Arm E $K=3$ + Fusion Head):** Observed highest sensitivity at **81.52%** (172/211 malignant cases detected), but lower specificity at **63.33%** (121 false positives) and lower Macro-F1 (70.29%).
2. **Arm B (Arm E $K=3$ + Frozen ViT):** Observed sensitivity of **79.62%** (168/211 malignant cases detected) and specificity of **67.27%** (108 false positives).
3. **Arm D (Arm E $K=3$ + Last 2 Blocks):** Observed highest specificity at **79.09%** (261/330 benign cases correctly identified, 69 false positives), with sensitivity of **69.19%** (146/211 malignant cases detected).
4. **Arm E2 (Arm E $K=5$ + Last 2 Blocks):** Observed sensitivity of **75.36%** and specificity of **70.00%**. While exhibiting ROC-AUC of 0.7983, it did not exceed Arm F's overall validation metrics.
5. **Arm F (Arm E $K=3$ + Last 4 Blocks):** Exhibited balanced performance with sensitivity of **77.25%** (163/211 malignant cases detected) and specificity of **75.15%** (248/330 benign cases correctly identified), resulting in the highest overall validation Macro-F1 (75.36%).

These trade-offs are validation-set observations and should not be interpreted as test-set performance.

---

## 6. Bootstrap Confidence Intervals

Empirical 95% confidence intervals generated via patient-clustered bootstrap resampling ($B=1,000$ iterations) from `phase2_bootstrap_ci.csv`:

| Arm | Metric | Bootstrap Mean | 95% CI Lower | 95% CI Upper | 95% CI Range |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm B** | Accuracy (%) | 71.86 | 66.23 | 77.17 | [66.23 – 77.17] |
| **Arm B** | Macro-F1 (%) | 71.48 | 65.99 | 76.62 | [65.99 – 76.62] |
| **Arm B** | ROC-AUC | 0.77 | 0.71 | 0.83 | [0.71 – 0.83] |
| **Arm B** | Sensitivity (%) | 79.61 | 73.36 | 85.92 | [73.36 – 85.92] |
| **Arm B** | Specificity (%) | 66.79 | 57.48 | 74.79 | [57.48 – 74.79] |
| **Arm C** | Accuracy (%) | 70.21 | 64.45 | 75.88 | [64.45 – 75.88] |
| **Arm C** | Macro-F1 (%) | 69.96 | 64.32 | 75.61 | [64.32 – 75.61] |
| **Arm C** | ROC-AUC | 0.76 | 0.69 | 0.82 | [0.69 – 0.82] |
| **Arm C** | Sensitivity (%) | 81.52 | 75.11 | 87.34 | [75.11 – 87.34] |
| **Arm C** | Specificity (%) | 62.85 | 53.27 | 71.03 | [53.27 – 71.03] |
| **Arm D** | Accuracy (%) | 75.05 | 69.89 | 79.70 | [69.89 – 79.70] |
| **Arm D** | Macro-F1 (%) | 73.83 | 69.05 | 78.32 | [69.05 – 78.32] |
| **Arm D** | ROC-AUC | 0.79 | 0.74 | 0.84 | [0.74 – 0.84] |
| **Arm D** | Sensitivity (%) | 69.28 | 61.84 | 75.85 | [61.84 – 75.85] |
| **Arm D** | Specificity (%) | 78.73 | 72.39 | 84.43 | [72.39 – 84.43] |
| **Arm E2** | Accuracy (%) | 71.89 | 66.19 | 77.20 | [66.19 – 77.20] |
| **Arm E2** | Macro-F1 (%) | 71.30 | 65.87 | 76.42 | [65.87 – 76.42] |
| **Arm E2** | ROC-AUC | 0.80 | 0.74 | 0.85 | [0.74 – 0.85] |
| **Arm E2** | Sensitivity (%) | 75.31 | 68.84 | 82.01 | [68.84 – 82.01] |
| **Arm E2** | Specificity (%) | 69.62 | 60.95 | 77.13 | [60.95 – 77.13] |
| **Arm F** | Accuracy (%) | 75.89 | 71.16 | 80.41 | [71.16 – 80.41] |
| **Arm F** | Macro-F1 (%) | 75.20 | 70.29 | 79.63 | [70.29 – 79.63] |
| **Arm F** | ROC-AUC | 0.81 | 0.76 | 0.86 | [0.76 – 0.86] |
| **Arm F** | Sensitivity (%) | 77.28 | 70.78 | 83.41 | [70.78 – 83.41] |
| **Arm F** | Specificity (%) | 74.91 | 67.67 | 81.45 | [67.67 – 81.45] |

---

## 7. Subgroup Analysis

Validation performance broken down across clinical subgroups from `phase2_subgroup_metrics.csv`:

| Arm | Subgroup Category | Subgroup | Sample Count ($N$) | Accuracy (%) | Macro-F1 (%) | ROC-AUC | Sensitivity (%) | Specificity (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm D** | Overall | Overall Cohort | 541 | 75.23 | 74.06 | 0.7935 | 69.19 | 79.09 |
| **Arm D** | Lesion Type | MASS | 244 | 74.18 | 74.15 | 0.8042 | 76.11 | 72.52 |
| **Arm D** | Lesion Type | CALCIFICATION | 297 | 76.09 | 72.60 | 0.7570 | 61.22 | 83.42 |
| **Arm D** | Pathology | BENIGN | 209 | 67.94 | 40.46 | — | 0.00 | 67.94 |
| **Arm D** | Pathology | MALIGNANT | 211 | 69.19 | 40.90 | — | 69.19 | 0.00 |
| **Arm D** | Mammographic View | CC (Craniocaudal) | 255 | 74.51 | 73.65 | 0.7958 | 73.47 | 75.16 |
| **Arm D** | Mammographic View | MLO (Mediolateral Oblique) | 286 | 75.87 | 74.38 | 0.7890 | 65.49 | 82.66 |
| **Arm E2** | Overall | Overall Cohort | 541 | 72.09 | 71.59 | 0.7983 | 75.36 | 70.00 |
| **Arm E2** | Lesion Type | MASS | 244 | 70.49 | 70.47 | 0.7422 | 78.76 | 63.36 |
| **Arm E2** | Lesion Type | CALCIFICATION | 297 | 73.40 | 71.43 | 0.8333 | 71.43 | 74.37 |
| **Arm E2** | Pathology | BENIGN | 209 | 55.98 | 35.89 | — | 0.00 | 55.98 |
| **Arm E2** | Pathology | MALIGNANT | 211 | 75.36 | 42.97 | — | 75.36 | 0.00 |
| **Arm E2** | Mammographic View | CC (Craniocaudal) | 255 | 73.33 | 72.93 | 0.8106 | 79.59 | 69.43 |
| **Arm E2** | Mammographic View | MLO (Mediolateral Oblique) | 286 | 70.98 | 70.37 | 0.7877 | 71.68 | 70.52 |

*Note: Pathology-specific subsets contain only a single ground-truth class; ROC-AUC is undefined for single-class partitions.*

---

## 8. Error Analysis

Categorization of classification outcomes relative to proposal localization quality ($\text{IoU} \ge 0.30$) from `phase2_error_analysis.csv`:

| Arm | Error & Outcome Category | Case Count ($N$) | Proportion (%) |
| :--- | :--- | :--- | :--- |
| **Arm D** | 1. Correct Classification + Good Proposal ($\text{IoU} \ge 0.30$) | 137 | 25.32% |
| **Arm D** | 2. Correct Classification + Weak Proposal ($\text{IoU} < 0.30$) | 270 | 49.91% |
| **Arm D** | 3. Malignant False Negative + Poor Proposal ($\text{IoU} < 0.30$) | 37 | 6.84% |
| **Arm D** | 4. Malignant False Negative + Good Proposal ($\text{IoU} \ge 0.30$) | 28 | 5.18% |
| **Arm D** | 5. Benign False Positive + Poor Proposal ($\text{IoU} < 0.30$) | 52 | 9.61% |
| **Arm D** | 6. Benign False Positive + Good Proposal ($\text{IoU} \ge 0.30$) | 17 | 3.14% |
| **Arm D** | 7. Fallback / Zero-Proposal Pathway | 0 | 0.00% |
| **Arm E2** | 1. Correct Classification + Good Proposal ($\text{IoU} \ge 0.30$) | 135 | 24.95% |
| **Arm E2** | 2. Correct Classification + Weak Proposal ($\text{IoU} < 0.30$) | 255 | 47.13% |
| **Arm E2** | 3. Malignant False Negative + Poor Proposal ($\text{IoU} < 0.30$) | 31 | 5.73% |
| **Arm E2** | 4. Malignant False Negative + Good Proposal ($\text{IoU} \ge 0.30$) | 21 | 3.88% |
| **Arm E2** | 5. Benign False Positive + Poor Proposal ($\text{IoU} < 0.30$) | 73 | 13.49% |
| **Arm E2** | 6. Benign False Positive + Good Proposal ($\text{IoU} \ge 0.30$) | 26 | 4.81% |
| **Arm E2** | 7. Fallback / Zero-Proposal Pathway | 0 | 0.00% |

In both evaluated configurations, Category 7 (Fallback / Zero-Proposal Pathway) was 0.00%, confirming that the multi-scale proposal pipeline supplied valid candidate regions for all validation scans.

---

## 9. Proposal Classification Analysis

Classification performance stratified across proposal localization quality bins from `phase2_proposal_classification_analysis.csv`:

| Proposal Quality (IoU Bin) | Sample Count ($N$) | Accuracy (%) | Macro-F1 (%) | ROC-AUC | Sensitivity (%) | Specificity (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $\text{IoU} < 0.10$ | 247 | 76.92 | 70.10 | 0.7448 | 59.02 | 82.80 |
| $0.10 \le \text{IoU} < 0.30$ | 112 | 71.43 | 71.35 | 0.7640 | 75.51 | 68.25 |
| $0.30 \le \text{IoU} < 0.50$ | 88 | 73.86 | 73.59 | 0.7837 | 74.00 | 73.68 |
| $\text{IoU} \ge 0.50$ | 94 | 76.60 | 76.60 | 0.8470 | 70.59 | 83.72 |

Observed ROC-AUC increased monotonically from 0.7448 in the lowest IoU bin ($\text{IoU} < 0.10$) to 0.8470 in the highest IoU bin ($\text{IoU} \ge 0.50$).

---

## 10. Train-Validation Gap Analysis

Evaluation of training versus validation performance metrics to observe generalization characteristics from `phase2_train_val_gap.csv`:

| Arm | Best Epoch | Train Acc (%) | Val Acc (%) | Accuracy Gap (%) | Train Macro-F1 (%) | Val Macro-F1 (%) | Macro-F1 Gap (%) | Train Loss | Val Loss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm B** (Frozen ViT) | 7 | 70.61 | 72.09 | -1.48 | 70.50 | 71.81 | -1.31 | 0.5556 | 0.5792 |
| **Arm C** (Fusion Head) | 5 | 68.13 | 70.43 | -2.30 | 68.10 | 70.29 | -2.19 | 0.5802 | 0.5969 |
| **Arm D** (Last 2 Blocks) | 21 | 98.52 | 75.23 | +23.29 | 98.48 | 74.06 | +24.42 | 0.1484 | 0.8220 |
| **Arm E2** ($K=5$, Last 2 Blocks) | 4 | 73.41 | 72.09 | +1.32 | 73.20 | 71.59 | +1.61 | 0.5355 | 0.5586 |
| **Arm F** (Last 4 Blocks) | 14 | 98.48 | 75.97 | +22.51 | 98.44 | 75.36 | +23.08 | 0.1538 | 0.8139 |

- **Frozen Backbone Arms (B, C):** Exhibited negligible train-validation gaps (-1.31% to -2.19% Macro-F1 gap), indicating minimal overfitting but constrained model capacity.
- **Unfrozen Backbone Arms (D, F):** Achieved high training set discrimination (Train Macro-F1 > 98%), with observed validation gaps of approximately 23–24 percentage points.

---

## 11. Calibration Analysis

Probability calibration evaluated using Brier Score and Expected Calibration Error (ECE) from `phase2_calibration.csv`:

| Arm | Brier Score | Expected Calibration Error (ECE, %) | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Arm A** (Locked V6 Arm-C Reference) | 0.1824 | 6.84% | 0.7787 | 0.6912 |
| **Arm B** (Arm E + Frozen ViT) | 0.1903 | 8.87% | 0.7764 | 0.6504 |
| **Arm C** (Arm E + Fusion Head Training) | 0.1980 | 9.30% | 0.7587 | 0.6295 |
| **Arm D** (Arm E + Last 2 ViT Blocks) | 0.2132 | 18.73% | 0.7935 | 0.6927 |
| **Arm E2** (Arm E $K=5$ + Last 2 ViT Blocks) | **0.1765** | **3.34%** | 0.7983 | 0.6810 |
| **Arm F** (Arm E + Last 4 ViT Blocks) | 0.2115 | 18.70% | 0.8153 | 0.6903 |

- **Arm E2** achieved the lowest Brier score (0.1765) and lowest ECE (3.34%) on the validation set.
- Arms with unfrozen ViT blocks (Arms D and F) exhibited higher ECE values (~18.7%), reflecting higher predicted probability confidence.

---

## 12. Statistical Interpretation

Based on empirical 95% bootstrap confidence intervals ($B=1,000$ patient-clustered resamples):
- **Arm F Macro-F1 95% CI:** `[70.29% – 79.63%]` (Bootstrap Mean: 75.20%)
- **Arm D Macro-F1 95% CI:** `[69.05% – 78.32%]` (Bootstrap Mean: 73.83%)
- **Arm E2 Macro-F1 95% CI:** `[65.87% – 76.42%]` (Bootstrap Mean: 71.30%)
- **Arm B Macro-F1 95% CI:** `[65.99% – 76.62%]` (Bootstrap Mean: 71.48%)
- **Arm C Macro-F1 95% CI:** `[64.32% – 75.61%]` (Bootstrap Mean: 69.96%)

### Confidence Interval Overlap Note:
The 95% bootstrap confidence intervals for Macro-F1 across all Phase 2 arms exhibit substantial overlap on the 541-scan validation set. For example, the confidence interval of Arm F `[70.29% – 79.63%]` overlaps with that of Arm D `[69.05% – 78.32%]` and Arm B `[65.99% – 76.62%]`. Consequently, claims of statistical superiority cannot be asserted without formal paired hypothesis testing on independent data.

---

## 13. Governance and Leakage Audit

Strict data isolation and reproducibility protocols were audited and verified:
1. **Patient Leakage:** Zero patient overlap between training (1,096 patients) and validation (235 patients) splits.
2. **Held-Out Test Quarantining:**
   - Total test-set reads during Phase 2: `0`
   - Total test-set predictions generated: `0`
   - Test cache directory: `ABSENT`
3. **Reproducibility Audit:** 10 validation cases re-evaluated from raw inputs yielded exact numerical reproduction ($\Delta_{\max} = 0.0$).
4. **Checkpoint MD5 Verification:** All saved model weights matched their logged cryptographic hashes.

---

## 14. Decision-Gate Outcome

The predefined Phase 2 validation decision gate required:
1. Completion of all six predefined ablation arms without errors.
2. Highest validation Macro-F1 meeting or exceeding the historical reference threshold (71.05%).
3. Verifiable checkpoint persistence and test-set isolation.

**Decision Gate Status:** **CRITERIA SATISFIED ON VALIDATION SET**
- Arm F produced the highest observed validation Macro-F1 (75.36%) and validation accuracy (75.97%).
- Arm D produced the second highest validation Macro-F1 (74.06%) with the highest specificity (79.09%).

---

## 15. Limitations

1. **Validation Set Evaluation:** All findings in this report represent validation cohort performance ($N=541$ scans). They do not constitute test-set confirmation or clinical efficacy claims.
2. **Sample Size for Subgroups:** Certain sub-partitions (e.g., individual lesion types or views) possess wider confidence intervals due to smaller sample counts ($N \approx 88 - 297$).
3. **Probability Calibration in Deep Tuning:** Arms with unfrozen ViT blocks exhibited higher ECE values, indicating that post-hoc calibration (such as temperature scaling) may be considered in subsequent development.

---

## 16. Recommended Architecture for Next Project Stage

Based on the empirical observations on the validation split:
1. **Primary Candidate Architecture:** **Arm F configuration** (Arm E Multi-Scale Proposals + ViT Backbone with Top 4 Blocks Tuned, $K=3$). This configuration produced the highest observed validation Macro-F1 (75.36%), accuracy (75.97%), and ROC-AUC (0.8153).
2. **Alternative Specificity Candidate:** **Arm D configuration** (Arm E Multi-Scale Proposals + ViT Backbone with Top 2 Blocks Tuned, $K=3$). This configuration produced the highest observed specificity (79.09%) and validation Macro-F1 (74.06%).
3. **Next Steps:** Proceed to formal planning and confirmation before any subsequent project phase or test evaluation.

---

V8 Phase 2 Status: COMPLETED

All Phase 2 results are validation-only. The held-out test split remained locked and had zero reads during Phase 2.
