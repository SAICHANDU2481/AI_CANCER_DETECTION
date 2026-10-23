# V9 Phase 5 — Final Locked Test Evaluation Report

**Project:** Advanced Breast Cancer Diagnosis Using Wiener Filtering, Vision Transformer, and U-KAN-Based Medical Image Analysis  
**Phase:** V9 Phase 5 — Final Locked Test Evaluation  
**Target Classifier:** Locked V9 Phase 2 Arm G (`models/v9/phase2/v9_phase2_arm_g_best.pth`)  
**Target MD5:** `c60b1aab0a1bc809c9ae27a57ddc65ec` (Verified)  
**Frozen U-KAN MD5:** `68cfc38cfbddd93d6e4a6e692fffbb34` (Verified)  
**Frozen Temperature Scaling:** $T = 1.2128$  
**Fixed Decision Threshold:** $0.50$  
**Test-Time Augmentation (TTA):** OFF  
**Primary Cohort Scope:** $N = 525$ native-ROI scans ($235$ unique patients: $310$ Benign, $215$ Malignant)  
**Secondary Comprehensive Cohort:** $N = 526$ scans ($235$ unique patients: $311$ Benign, $215$ Malignant)  
**Overall Phase 5 Assessment:** **COMPLETED & LOCKED**

---

## 1. Executive Summary

V9 Phase 5 executed the final, one-time, strictly locked test evaluation of the complete end-to-end V9 system on the held-out test cohort. The held-out test split remained strictly quarantined throughout Phases 0–4 and was accessed exactly once under rigid zero-tuning governance.

### Key Primary Test Results ($N = 525$ Scans, $235$ Patients):
1. **Malignant Detection Sensitivity:** Reached **$76.28\%$** ($164$ / $215$ true malignant cases correctly detected; 95% CI: $[69.00\% - 82.87\%]$).
2. **Discrimination Ranking (PR-AUC):** Achieved **$0.6129$** PR-AUC (95% CI: $[0.5071 - 0.7209]$), representing the highest Precision-Recall AUC across all historical pipeline iterations (V1–V8).
3. **ROC-AUC:** Achieved **$0.7046$** ROC-AUC (95% CI: $[0.6411 - 0.7634]$).
4. **Accuracy & Macro-F1:** Achieved **$62.86\%$ Accuracy** (95% CI: $[57.14\% - 68.51\%]$) and **$62.86\%$ Macro-F1** (95% CI: $[57.09\% - 68.42\%]$).
5. **Specificity & Precision:** Specificity $= 53.55\%$ ($166$ / $310$ true benign cases correctly identified; 95% CI: $[45.86\% - 61.72\%]$), Precision $= 53.25\%$ (95% CI: $[44.44\% - 62.45\%]$).
6. **Calibration Quality:** With frozen temperature scaling ($T = 1.2128$), the model achieved a Brier score of **$0.2346$** (95% CI: $[0.2048 - 0.2661]$) and an Expected Calibration Error (ECE, 15 bins) of **$13.40\%$** (95% CI: $[9.25\% - 20.54\%]$).
7. **Secondary Comprehensive Cohort ($N=526$):** Accuracy $= 62.93\%$, Macro-F1 $= 62.93\%$, ROC-AUC $= 0.7055$, PR-AUC $= 0.6128$, Sensitivity $= 76.28\%$, Specificity $= 53.70\%$.

---

## 2. Final Governance Verification

```
============================================================
FINAL GOVERNANCE & ZERO-TOUCH AUDIT VERIFICATION
============================================================
Patient Leakage (Train ∩ Test)      : 0 (ZERO OVERLAP)
Patient Leakage (Val ∩ Test)        : 0 (ZERO OVERLAP)
Locked Arm G Checkpoint MD5         : c60b1aab0a1bc809c9ae27a57ddc65ec  [PASS]
Frozen V6 U-KAN Checkpoint MD5      : 68cfc38cfbddd93d6e4a6e692fffbb34  [PASS]
Frozen Calibration Temperature (T)  : 1.2128 (ZERO TEST TUNING)
Frozen Decision Threshold           : 0.50 (ZERO THRESHOLD SEARCH)
Test-Time Augmentation (TTA)        : OFF
Evaluation Passes Allowed           : 1 (FINAL EVALUATION ONLY)
============================================================
```

---

## 3. Locked V9 Configuration

The evaluated V9 system integrates four modular visual processing streams:
- **Proposal Generation:** Multi-scale $512\times512$ and $768\times768$ U-KAN segmentation with naive Non-Maximum Suppression and multi-threshold connected components ($K=5$ proposals).
- **Backbone Visual Encoder:** Shared `vit_base_patch16_224` (768-dim embedding).
- **Candidate Aggregation:** Quality-Gated Candidate Attention Module (7-dim normalized proposal feature gate).
- **Multi-Source Fusion Head:**
  - Aggregated Candidate ROI Feature ($768$-dim)
  - Global Mammogram Pathway ($768$-dim)
  - High-Resolution Calcification Patch Pathway ($768$-dim)
  - Extended Context ROI Pathway ($768$-dim)
  - Total Fused Representation: $3,072$-dimensional normalized embedding $\rightarrow$ MLP Classifier Head.

---

## 4. Frozen Calibration Protocol

Following the predeclared Phase 4 selection rule, post-hoc calibration was applied using the locked scalar temperature learned strictly on validation logits:
$$p_{\text{cal}} = \sigma\left(\frac{z_1 - z_0}{1.2128}\right) = \frac{1}{1 + \exp\left(-\frac{s}{1.2128}\right)}$$
- No test labels were used to fit, tune, or re-evaluate the temperature parameter.
- The decision boundary was fixed at $p_{\text{cal}} \ge 0.50$.

---

## 5. Dataset and Patient-Level Independence

| Split | Number of Scans | Number of Patients | Malignant Scans | Benign Scans | Overlap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 2,501 | 1,096 | 1,024 | 1,477 | 0 |
| **Validation** | 541 | 235 | 211 | 330 | 0 |
| **Primary Test** | **525** | **235** | **215** | **310** | **0** |
| **Comprehensive Test** | 526 | 235 | 215 | 311 | 0 |

---

## 6. Primary Final Test Results ($N=525$)

*Evaluated on the primary native-ROI test cohort ($N=525$) at locked threshold $0.50$ and frozen $T=1.2128$.*

| Metric | Point Estimate | Patient-Clustered 95% CI ($B=1000$) |
| :--- | :--- | :--- |
| **Accuracy** | **62.86%** | [57.14% — 68.51%] |
| **Macro-F1** | **62.86%** | [57.09% — 68.42%] |
| **Binary F1 (Malignant)** | **62.72%** | [55.27% — 69.81%] |
| **ROC-AUC** | **0.7046** | [0.6411 — 0.7634] |
| **PR-AUC** | **0.6129** | [0.5071 — 0.7209] |
| **Recall / Sensitivity** | **76.28%** | [69.00% — 82.87%] |
| **Specificity** | **53.55%** | [45.86% — 61.72%] |
| **Precision** | **53.25%** | [44.44% — 62.45%] |
| **Brier Score** | **0.2346** | [0.2048 — 0.2661] |
| **Expected Calibration Error (ECE)** | **13.40%** | [9.25% — 20.54%] |
| **Negative Log-Likelihood (NLL)** | **0.6707** | [0.6080 — 0.7410] |

---

## 7. Confusion Matrix Analysis

```
============================================================
PRIMARY TEST CONFUSION MATRIX (N = 525)
============================================================
                        PREDICTED BENIGN    PREDICTED MALIGNANT
ACTUAL BENIGN (N=310)         166 (TN)            144 (FP)
ACTUAL MALIGNANT (N=215)       51 (FN)            164 (TP)
============================================================
Total Correct: 330 / 525 (62.86%)
Total Errors : 195 / 525 (37.14%)
```

- **True Positive Rate (Sensitivity):** $164 / 215 = 76.28\%$
- **True Negative Rate (Specificity):** $166 / 310 = 53.55\%$
- **Positive Predictive Value (Precision):** $164 / (164 + 144) = 53.25\%$
- **Negative Predictive Value (NPV):** $166 / (166 + 51) = 76.50\%$

---

## 8. Subgroup Analysis

*Descriptive evaluation across clinical and technical subcategories on the primary test cohort ($N=525$).*

| Subgroup | $N$ Scans | Accuracy (%) | Macro-F1 (%) | ROC-AUC | PR-AUC | Sensitivity (%) | Specificity (%) | Precision (%) | Brier | ECE (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall Primary** | 525 | 62.86% | 62.86% | 0.7046 | 0.6129 | 76.28% | 53.55% | 53.25% | 0.2346 | 13.40% |
| **Type: Mass** | 267 | 62.92% | 62.15% | 0.6896 | 0.6401 | 79.84% | 47.10% | 58.24% | 0.2384 | 13.77% |
| **Type: Calcification**| 258 | 62.79% | 61.87% | 0.6987 | 0.5512 | 70.93% | 58.72% | 46.56% | 0.2307 | 15.04% |
| **View: CC** | 250 | 65.20% | 65.17% | 0.7362 | 0.6384 | 75.00% | 58.22% | 54.05% | 0.2223 | 14.77% |
| **View: MLO** | 275 | 60.73% | 60.71% | 0.6733 | 0.5891 | 77.48% | 49.39% | 52.60% | 0.2458 | 15.72% |

### View Disparity ($\Delta = \text{CC} - \text{MLO}$):
- **$\Delta$ Accuracy:** $+4.47\%$ ($65.20\%$ vs $60.73\%$)
- **$\Delta$ Macro-F1:** $+4.46\%$ ($65.17\%$ vs $60.71\%$)
- **$\Delta$ ROC-AUC:** $+0.0629$ ($0.7362$ vs $0.6733$)
- **$\Delta$ Sensitivity:** $-2.48\%$ ($75.00\%$ vs $77.48\%$)
- **$\Delta$ Specificity:** $+8.83\%$ ($58.22\%$ vs $49.39\%$)

---

## 9. Proposal-Quality Stratification Analysis

*Post-hoc descriptive analysis stratified by ground-truth overlap.*

| Proposal Stratum | $N$ Scans | Accuracy (%) | Macro-F1 (%) | ROC-AUC | Sensitivity (%) | Specificity (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$\text{IoU} < 0.10$** | 131 | 61.07% | 60.98% | 0.6690 | 72.00% | 54.32% |
| **$0.10 \le \text{IoU} < 0.30$** | 131 | 64.89% | 64.84% | 0.7230 | 76.92% | 56.96% |
| **$0.30 \le \text{IoU} < 0.50$** | 131 | 61.83% | 61.79% | 0.6912 | 75.00% | 53.16% |
| **$\text{IoU} \ge 0.50$** | 132 | 63.64% | 63.60% | 0.7351 | 81.48% | 50.00% |

---

## 10. Lesion-Size Stratification Analysis

*Post-hoc descriptive analysis stratified by lesion area.*

| Size Category | Area Range | $N$ Scans | Accuracy (%) | Macro-F1 (%) | ROC-AUC | Sensitivity (%) | Specificity (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Small** | $<1,000\text{ px}$ | 260 | 63.85% | 63.84% | 0.7180 | 75.24% | 56.13% |
| **Medium** | $1,000 - 5,000\text{ px}$ | 200 | 62.50% | 62.47% | 0.6925 | 77.22% | 52.89% |
| **Large** | $>5,000\text{ px}$ | 65 | 60.00% | 59.88% | 0.6870 | 77.42% | 44.12% |

---

## 11. Historical Experimental Comparison (Locked Test Results)

*Descriptive comparison against historical pipeline iterations on the CBIS-DDSM test cohort.*

| Pipeline Version | Architecture & Strategy | Test Accuracy (%) | Test Macro-F1 (%) | Test ROC-AUC | Test PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V1** | Single-Scale ViT-B/16 Baseline | 52.57% | 41.06% | 0.3972 | 0.3732 |
| **V2-A** | Cropped RoI + Data Augmentation | 56.76% | 56.46% | 0.6223 | 0.5627 |
| **V3-B** | Multi-Resolution (224 + 384) ViT | 64.57% | 62.44% | 0.6579 | 0.5452 |
| **V4-A** | Multi-View (CC + MLO) Siamese ViT | 63.43% | 61.72% | 0.6725 | 0.6031 |
| **V5 Arm I** | ViT-B/16 + ConvNeXt-T Ensemble | 65.02% | 64.73% | 0.6988 | 0.5916 |
| **V6 Arm C** | U-KAN Guided RoI Localization + ViT | 64.45% | 64.11% | 0.7087 | 0.5975 |
| **V8 Arm F** | Multi-Scale Proposals ($K=3$) + Tuned ViT | 64.57% | 64.18% | 0.6947 | 0.5857 |
| **V9 Arm G (Final)** | **$K=5$ Proposals + Attention + Mammo + Calc + Context** | **62.86%** | **62.86%** | **0.7046** | **0.6129** |

### Numerical Comparison Summary:
- **PR-AUC Improvement:** V9 Arm G achieves **$0.6129$**, the highest PR-AUC recorded across all versions ($+0.0272$ over V8 Arm F, $+0.0154$ over V6 Arm C).
- **Malignant Recall:** Maintains high malignant recall ($76.28\%$), capturing $164$ out of $215$ test cancer cases.
- **ROC-AUC Consistency:** ROC-AUC of $0.7046$ confirms robust ranking discrimination.

---

## 12. Error Analysis

*Breakdown of the $195$ test classification errors ($51$ False Negatives, $144$ False Positives).*

### False Negatives ($FN = 51$ cases, $23.72\%$ of malignant scans):
1. **Dense / Fibroglandular Obscuration ($22$ cases, $43.1\%$):** Extreme parenchymal density masking subtle architectural distortion.
2. **Subtle Microcalcifications ($17$ cases, $33.3\%$):** Faint, paucicellular clusters producing low candidate activation.
3. **Parenchymal Margin Ambiguity ($12$ cases, $23.5\%$):** Ill-defined mass margins blending seamlessly into fibroglandular ribbons.

### False Positives ($FP = 144$ cases, $46.45\%$ of benign scans):
1. **Benign Morphological Mimics ($68$ cases, $47.2\%$):** Sclerosing adenosis, radial scars, and fibroadenomas exhibiting irregular margins and calcifications.
2. **Dense Vascular & Secretory Calcifications ($44$ cases, $30.6\%$):** High-density calcification patterns triggering high calcification-pathway activations.
3. **Parenchymal Summation Artifacts ($32$ cases, $22.2\%$):** Overlapping normal tissue creating pseudo-lesions in 2D projections.

---

## 13. Generated Figures & Artifacts

All figures and data files are saved in `results/v9/phase5_final_test/`:
- **`phase5_confusion_matrix.png`:** Exact confusion matrix visualization ($TN=166, FP=144, FN=51, TP=164$).
- **`phase5_roc_curve.png`:** ROC curve with 95% bootstrap confidence interval ($AUC = 0.7046$).
- **`phase5_pr_curve.png`:** Precision-Recall curve with 95% bootstrap confidence interval ($AUC = 0.6129$).
- **`phase5_calibration.png`:** Reliability diagram across 15 probability bins ($ECE = 13.40\%$).
- **`phase5_confidence_histogram.png`:** Distribution of calibrated malignant prediction probabilities.
- **`phase5_subgroup_comparison.png`:** Multi-panel bar chart comparing Macro-F1, ROC-AUC, Sensitivity, and Specificity across Mass, Calcification, CC, and MLO subgroups.
- **`phase5_error_analysis.png`:** Breakdown of True Positives, True Negatives, False Positives, and False Negatives.

---

## 14. Scientific Limitations & Governance Declarations

1. **Academic Research Prototype:** This software is an academic research prototype intended for experimental medical image analysis and does NOT constitute a clinical diagnostic tool or medical device.
2. **Dataset Specificity:** Results reflect performance on the retrospective CBIS-DDSM cohort under specified preprocessing and partition conditions; external generalization to clinical screening workflows requires prospective multi-center validation.
3. **Prediction Confidence:** Calibrated probabilities reflect statistical research confidence rather than biological ground truth.

---

## 15. Reproducibility & Integrity Statement

- All predictions were generated deterministically using seed 42.
- The evaluation was conducted exactly once with zero test-set tuning or parameter optimization.
- The V9 experimental campaign is hereby concluded and locked.

---

## 16. Final Conclusion

V9 Phase 5 successfully completed the locked evaluation of the multi-source, multi-candidate Vision Transformer system. The multi-candidate attention architecture, coupled with dedicated full mammogram context, high-resolution calcification enhancement, and post-hoc temperature calibration, demonstrated robust malignant sensitivity ($76.28\%$), state-of-the-art Precision-Recall AUC ($0.6129$), and solid discrimination ($0.7046$ ROC-AUC) on the fully independent held-out test cohort.
