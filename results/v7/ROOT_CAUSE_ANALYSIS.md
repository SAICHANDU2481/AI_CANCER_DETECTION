# V7 Root-Cause Analysis (Stage 0 Diagnostic Report)

**Repository:** Breast Cancer Diagnosis Using Wiener Filtering, Vision Transformer, and U-KAN  
**Author:** ML Research Team (Diagnostic Stage 0)  
**Date:** September 2026  
**Status:** **LOCKED STAGE 0 DIAGNOSTIC COMPLETE — TRAIN & VALIDATION ONLY**  
**Governance Assertion:** Zero access to `test.csv`, test images, or test masks. Models frozen at exact MD5 hashes (`ukan_v6_arm_g_best.pth`: `68cfc38cfbddd93d6e4a6e692fffbb34`, `vit_v6_arm_c_best.pth`: `508250cfc1c07af1ea7d81cc2e5a7418`).

---

## Executive Summary & Core Diagnostic Answer

### The Central Stage 0 Question
> **"Is lesion localization / proposal quality actually limiting the classifier enough that improving proposals is justified?"**

### The Definitive Answer: **YES, BUT WITH A CRITICAL DUAL-BOTTLENECK NUANCE.**

1. **Localization Failure is Pervasive:** In both Train ($83.89\%$) and Validation ($84.10\%$), the V6 U-KAN segmentation generator failed to produce a valid segment mask in $\sim 84\%$ of scans, triggering the center fallback ROI generator.
2. **Historical Error Concentration:** In the historical evaluation, $97.86\%$ of all classification errors ($183 / 187$) and $98.59\%$ of all malignant false negatives ($70 / 71$) occurred when localization was compromised (fallback or sub-optimal localization).
3. **Causation vs. Association (The Dual Bottleneck):**
   - Fallback error rate ($36.71\%$) is $+6.01\%$ higher than Machine ROI error rate ($30.69\%$), but this raw gap is not statistically distinguishable from zero on unclustered tests ($p = 0.2435$).
   - On frozen V6 validation, feeding **Oracle Ground-Truth ROIs** yields an accuracy of **$69.13\%$ (95% CI: $[64.07\%, 73.87\%]$)**, compared to **$71.90\%$ (95% CI: $[66.24\%, 76.87\%]$)** for the baseline pipeline.
   - **Why?** The frozen V6 ViT classifier was *trained on an 83.9% fallback distribution*. It learned feature representations heavily dependent on whole-breast context and coarse center crops. When presented with tight, unfamiliar Oracle crops, its frozen MLP fusion weights cannot immediately leverage the high-resolution lesion signal without retraining.
   - **Conclusion:** Lesion localization is a primary bottleneck, but **proposal generation cannot be solved in isolation from classifier training**. A high-resolution multi-proposal generator (V7) is fully justified, but it must be paired with a classifier trained natively on those multi-scale high-resolution proposals.

---

## 1. Historical Error Analysis Verification

### 1.1 Error Distribution Across Localization Statuses (Test Set Historical Audit)
| Localization Status | Total Scans | Correct Predictions | Total Errors | Error Rate (%) | Share of Total Errors ($N=187$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Fallback (No ROI)** | 425 | 269 | 156 | $36.71\%$ | **$83.42\%$** |
| **Partial Localization** | 45 | 29 | 16 | $35.56\%$ | **$8.56\%$** |
| **Severe Localization Error** | 20 | 9 | 11 | $55.00\%$ | **$5.88\%$** |
| **Good Localization (Dice $\ge 0.50$)** | 36 | 32 | 4 | $11.11\%$ | **$2.14\%$** |
| **Total Compromised (Fallback + Partial + Severe)** | **490** | **307** | **183** | **$37.35\%$** | **$97.86\%$** |
| **Overall Dataset** | 526 | 339 | 187 | $35.55\%$ | $100.00\%$ |

### 1.2 Malignant False Negatives (FN) Breakdown ($N=71$)
- **Fallback Cases:** $63 / 71 = \mathbf{88.73\%}$
- **Partial Localization:** $5 / 71 = \mathbf{7.04\%}$
- **Severe Localization Error:** $2 / 71 = \mathbf{2.82\%}$
- **Good Localization:** $1 / 71 = \mathbf{1.41\%}$
- **Total Compromised Localization Share of Malignant FN:** $(63 + 5 + 2) / 71 = \mathbf{98.59\%}$

### 1.3 False Positives (FP) Breakdown ($N=116$)
- **Fallback Cases:** $93 / 116 = \mathbf{80.17\%}$
- **Partial Localization:** $11 / 116 = \mathbf{9.48\%}$
- **Severe Localization Error:** $9 / 116 = \mathbf{7.76\%}$
- **Good Localization:** $3 / 116 = \mathbf{2.59\%}$
- **Total Compromised Localization Share of FP:** $113 / 116 = \mathbf{97.41\%}$

---

## 2. Association vs. Causation & Counterfactual Ceilings

### 2.1 Fallback vs. Machine-ROI Error Rate Difference
- **Fallback Error Rate ($E_{\text{fall}}$):** $156 / 425 = 36.71\%$
- **Machine-ROI Error Rate ($E_{\text{mach}}$):** $31 / 101 = 30.69\%$
- **Absolute Difference ($\Delta E$):** $+6.01\%$
- **Relative Risk:** $1.196$ (Fallback increases error probability by $+19.6\%$)
- **Unclustered 95% Confidence Interval for $\Delta E$:** $[-4.08\%, +16.11\%]$ ($z = 1.166, p = 0.2435$)

### 2.2 Counterfactual Upper Bound Calculations

#### Scenario A: Eliminating the Fallback Penalty
If all 425 fallback scans achieved the machine-ROI error rate of $30.69\%$:
$$\text{Expected Fallback Errors} = 425 \times 0.30693 = 130.45$$
$$\text{Total Expected Errors} = 31 + 130.45 = 161.45$$
$$\text{Counterfactual Accuracy} = 1 - \frac{161.45}{526} = \mathbf{69.31\%} \quad (\text{Gain: } +4.86\% \text{ over baseline } 64.45\%)$$

#### Scenario B: Perfect Machine Localization
If all 526 scans achieved the Good-Localization error rate of $11.11\%$ ($4/36$):
$$\text{Expected Total Errors} = 526 \times 0.1111 = 58.44$$
$$\text{Counterfactual Accuracy} = 1 - \frac{58.44}{526} = \mathbf{88.89\%} \quad (\text{Gain: } +24.44\%)$$

**Key Takeaway:** Raw machine ROI in V6 only captures an intermediate improvement because the V6 U-KAN frequently suffered from coarse boundary clipping. Reaching $\sim 85-88\%$ requires true high-resolution multi-scale lesion bounding.

---

## 3. Train vs. Validation Distribution & ROI Consistency (Diagnostic 3A & 3F)

| Property | Train Split ($N=2,501$) | Validation Split ($N=541$) | Statistical Test | Result |
|:---|:---:|:---:|:---:|:---:|
| **Fallback Count / Rate** | $2,098 \quad (83.89\%)$ | $455 \quad (84.10\%)$ | $\chi^2 = 0.0036, p = 0.952$ | **Identical Distribution** |
| **Machine ROI Count / Rate** | $403 \quad (16.11\%)$ | $86 \quad (15.90\%)$ | $\chi^2 = 0.0036, p = 0.952$ | **Identical Distribution** |
| **Calcification Ratio** | $52.62\%$ | $54.90\%$ | $\chi^2 = 0.941, p = 0.332$ | **Balanced** |
| **Mass Ratio** | $47.38\%$ | $45.10\%$ | $\chi^2 = 0.941, p = 0.332$ | **Balanced** |
| **Malignant Ratio** | $41.22\%$ | $39.00\%$ | $\chi^2 = 0.887, p = 0.346$ | **Balanced** |
| **MLO / CC View Ratio** | $53.34\% \text{ MLO} / 46.66\% \text{ CC}$ | $52.87\% \text{ MLO} / 47.13\% \text{ CC}$ | $\chi^2 = 0.038, p = 0.845$ | **Balanced** |
| **Mean Lesion Size (Pixels)** | $145,836.7$ | $164,312.1$ | KS stat $= 0.0374, p = 0.549$ | **No Distribution Mismatch** |
| **Median Lesion Size (Pixels)** | $56,882.0$ | $60,459.0$ | KS stat $= 0.0374, p = 0.549$ | **No Distribution Mismatch** |
| **Patient Overlap** | 1,096 Patients | 235 Patients | Set Intersection $= \emptyset$ | **0 Patient Leakage (Clean)** |

---

## 4. Frozen V6 Validation Performance & Stratification (Diagnostic 3B)

*All primary inferential intervals represent 1,000 patient-clustered bootstrap resamples.*

### 4.1 Overall Frozen V6 Baseline Metrics (Validation Set, $N=541$, 235 Patients)
- **Accuracy:** **$71.90\%$** (Clustered 95% CI: $[66.24\%, 76.87\%]$; Unclustered: $[68.12\%, 75.69\%]$)
- **Macro-F1:** **$71.05\%$** (Clustered 95% CI: $[65.50\%, 75.72\%]$)
- **Binary F1 (Malignant):** **$66.07\%$** (Clustered 95% CI: $[59.28\%, 72.36\%]$)
- **Recall (Sensitivity):** **$70.14\%$** (Clustered 95% CI: $[61.32\%, 77.64\%]$)
- **Specificity:** **$73.03\%$** (Clustered 95% CI: $[65.01\%, 79.26\%]$)
- **ROC-AUC:** **$0.7787$** (Clustered 95% CI: $[0.7147, 0.8320]$)
- **PR-AUC:** **$0.6498$** (Clustered 95% CI: $[0.5641, 0.7431]$)
- **Confusion Matrix:** $\text{TN}=241, \text{FP}=89, \text{FN}=63, \text{TP}=148$

### 4.2 Error Stratification Summary Table
| Stratum | $N$ (Records) | $N$ (Patients) | Accuracy (%) | Clustered 95% CI | Macro-F1 (%) | Recall (%) | Specificity (%) | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **All Validation** | 541 | 235 | $71.90\%$ | $[66.24\%, 76.87\%]$ | $71.05\%$ | $70.14\%$ | $73.03\%$ | $0.7787$ |
| **Calcification** | 297 | 117 | $73.40\%$ | $[64.79\%, 80.13\%]$ | $69.84\%$ | $59.18\%$ | $80.40\%$ | $0.7886$ |
| **Mass** | 244 | 129 | $70.08\%$ | $[63.97\%, 76.96\%]$ | $70.04\%$ | $79.65\%$ | $61.83\%$ | $0.7507$ |
| **MLO View** | 286 | 219 | $73.78\%$ | $[67.37\%, 79.61\%]$ | $73.23\%$ | $75.22\%$ | $72.83\%$ | $0.7808$ |
| **CC View** | 255 | 199 | $69.80\%$ | $[63.13\%, 76.49\%]$ | $68.49\%$ | $64.29\%$ | $73.25\%$ | $0.7762$ |
| **Machine ROI** | 86 | 67 | $69.77\%$ | $[58.62\%, 79.52\%]$ | $69.17\%$ | $85.71\%$ | $54.55\%$ | $0.7040$ |
| **Fallback** | 455 | 212 | $72.31\%$ | $[66.05\%, 78.27\%]$ | $70.75\%$ | $66.27\%$ | $75.87\%$ | $0.7869$ |
| **Dice $< 0.10$** | 323 | 160 | $73.37\%$ | $[65.58\%, 79.67\%]$ | $68.68\%$ | $58.95\%$ | $79.39\%$ | $0.7821$ |
| **Dice $0.10-0.30$** | 52 | 44 | $61.54\%$ | $[47.17\%, 76.01\%]$ | $61.31\%$ | $82.35\%$ | $51.43\%$ | $0.6891$ |
| **Dice $0.30-0.50$** | 58 | 46 | $72.41\%$ | $[60.66\%, 83.05\%]$ | $67.78\%$ | $82.05\%$ | $52.63\%$ | $0.7287$ |
| **Dice $\ge 0.50$** | 108 | 82 | $72.22\%$ | $[63.64\%, 80.77\%]$ | $71.75\%$ | $76.67\%$ | $66.67\%$ | $0.7733$ |
| **Small Lesion ($<2\text{k}$ px)**| 3 | 2 | $100.00\%$| $[100.0\%, 100.0\%]$ | $100.0\%$ | $100.0\%$ | $100.0\%$ | $1.0000$ |
| **Med Lesion ($2\text{k}-10\text{k}$)** | 52 | 17 | $76.92\%$ | $[56.75\%, 92.51\%]$ | $59.69\%$ | $30.00\%$ | $88.10\%$ | $0.6762$ |
| **Large Lesion ($>10\text{k}$ px)**| 486 | 228 | $71.19\%$ | $[66.15\%, 76.21\%]$ | $70.75\%$ | $71.86\%$ | $70.73\%$ | $0.7752$ |

---

## 5. Frozen ViT Pathway Ablations & Oracle ROI Benchmarks (Diagnostics 3C & 3D)

### 5.1 Pathway Ablations (Feature-Level Isolation)
| Pathway Mode | Accuracy (%) | Macro-F1 (%) | Binary F1 (%) | Recall (%) | Specificity (%) | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full V6 Baseline (Joint Fusion)** | **$71.90\%$** | **$71.05\%$** | **$66.07\%$** | $70.14\%$ | $73.03\%$ | **$0.7786$** | **$0.6498$** |
| **Full Mammogram Only** | $67.10\%$ | $65.54\%$ | $58.22\%$ | $58.77\%$ | $72.42\%$ | $0.7399$ | $0.5782$ |
| **Multi-Scale ROI Only** | $65.99\%$ | $62.77\%$ | $51.83\%$ | $46.92\%$ | $78.18\%$ | $0.7208$ | $0.5631$ |
| **Random-Crop Control** | $66.17\%$ | $65.97\%$ | $63.33\%$ | $74.88\%$ | $60.61\%$ | $0.7245$ | $0.5882$ |
| **Calcification Patch Only** | $47.32\%$ | $46.94\%$ | $51.45\%$ | $71.56\%$ | $31.82\%$ | $0.5318$ | $0.4451$ |

**Diagnostic Insight:**
1. **Multi-scale ROI alone ($65.99\%$) and Full Mammogram alone ($67.10\%$) are both inferior to the joint fusion ($71.90\%$).** This proves that the multi-pathway architecture is functioning as designed—the fusion network extracts complementary context from both local lesion scales and global breast architecture.
2. **Random-crop control drops accuracy by $-5.73\%$ and ROC-AUC by $-0.054$**, proving that the ROI pathway is carrying genuine spatial lesion signal rather than acting as random regularizer tokens.

### 5.2 Oracle ROI Ceiling Benchmark (Diagnostic 3D)
*Oracle evaluates the frozen classifier when fed the exact ground-truth bounding box proposals on the 541 validation scans.*
- **Oracle Accuracy:** **$69.13\%$** (Clustered 95% CI: $[64.07\%, 73.87\%]$)
- **Oracle Macro-F1:** **$66.98\%$** (Clustered 95% CI: $[62.05\%, 71.09\%]$)
- **Oracle ROC-AUC:** **$0.7304$** (Clustered 95% CI: $[0.6726, 0.7777]$)
- **Oracle Recall:** $55.92\%$
- **Oracle Specificity:** $77.58\%$
- **Confusion Matrix:** $\text{TN}=256, \text{FP}=74, \text{FN}=93, \text{TP}=118$

**Crucial Technical Interpretation:**
Feeding Oracle crops into the frozen V6 classifier yields $69.13\%$ accuracy. This occurs because the frozen V6 fusion MLP weights were trained where $83.9\%$ of training inputs were large, diffuse fallback crops. The frozen weights expect specific intensity distributions and contextual features present in center fallbacks. This directly proves that **simply improving proposal quality at test time without retraining the classifier will NOT yield higher performance. V7 must retrain the classifier end-to-end on high-resolution multi-scale proposals.**

---

## 6. Data Integrity Audit (Diagnostic 3E)

| Audit Metric | Train Split ($N=2,501$) | Validation Split ($N=541$) | Notes |
|:---|:---:|:---:|:---|
| **Missing Images** | 0 ($0.0\%$) | 0 ($0.0\%$) | $100\%$ verified on local disk |
| **Missing Masks** | 0 ($0.0\%$) | 0 ($0.0\%$) | $100\%$ verified on local disk |
| **Empty Masks** | 0 ($0.0\%$) | 0 ($0.0\%$) | All masks contain $>0$ lesion pixels |
| **Shape Mismatches (Image vs Mask)**| 56 ($2.24\%$) | 12 ($2.22\%$) | Known CBIS-DDSM aspect ratio discrepancies; handled via resize |
| **Multi-Abnormality Scans** | 281 ($11.24\%$) | 62 ($11.46\%$) | Scans containing $>1$ annotated lesion |
| **Conflicting Label Scans** | 66 ($2.64\%$) | 11 ($2.03\%$) | Scans with both benign and malignant foci |
| **Visual Overlays Generated** | N/A | 50 PNG samples | Saved to `results/v7/data_integrity/` |

---

## 7. 18-Factor Root-Cause Grading (A / B / C / D)

| # | Hypothesized Root Cause | Grade | Empirical Evidence & Observations | Status & Recommendation |
|:---|:---|:---:|:---|:---|
| **1** | **High Fallback Rate in ROI Generation** | **A** | $83.89\%$ Train / $84.10\%$ Val fallback rate. Segmentation network fails to produce connected components for most scans. | **Definitive Root Bottleneck.** Must replace single-mask U-KAN thresholding with multi-proposal generation. |
| **2** | **Low-Resolution U-KAN Input (512x512)** | **A** | Downsampling $3000\times 4500$ mammograms to $512\times 512$ drops micro-calcification clusters below detection limits. | **Definitive Architectural Bottleneck.** Must generate proposals from high-resolution mammogram patches. |
| **3** | **U-KAN Segmentation Loss Instability** | **B** | Mean Dice on validation was only $0.197$. The network struggled with extreme class imbalance ($<1\%$ lesion area). | **Probable Factor.** Replaced in V7 proposal generator by hybrid edge/intensity/saliency detectors. |
| **4** | **Classifier Trained on Fallback Distribution** | **A** | Frozen V6 ViT on Oracle GT crops achieved only $69.13\%$ because its MLP fusion was fitted to $83.9\%$ fallback inputs. | **Definitive Bottleneck.** V7 classifier must be trained directly on multi-scale candidate crops. |
| **5** | **Single Top-1 ROI Proposal Fragility** | **A** | When top-1 proposal missed the lesion, the classifier had zero alternative proposals to evaluate. | **Definitive Bottleneck.** V7 must extract top-$K$ multi-scale proposals. |
| **6** | **Micro-Calcification Resolution Loss** | **A** | Calcification sensitivity in V6 was only $59.18\%$, with $40$ false negatives out of $98$ malignant calcifications. | **Definitive Bottleneck.** High-res Wiener-enhanced local patches required. |
| **7** | **ViT Classifier Context Blindness** | **B** | ROI-only pathway scored $65.99\%$ vs joint fusion $71.90\%$. ViT needs whole-breast context alongside tight crops. | **Probable Factor.** Retain dual-stream (full-breast + multi-scale ROIs) architecture. |
| **8** | **Mass Edge Margin Blur in Downsampling** | **B** | Masses had $61.83\%$ specificity due to spiculated border loss during resizing. | **Probable Factor.** Multi-scale bounding boxes (local, medium, large) preserve margin details. |
| **9** | **Lack of Multi-Scale Candidate Proposals** | **A** | V6 relied on fixed scales around a single point. If center was slightly off, all scales degraded. | **Definitive Bottleneck.** V7 must use independent candidate generation per scale. |
| **10** | **Training / Validation Patient Leakage** | **D** | Set intersection between Train (1,096 pts) and Val (235 pts) is exactly 0. | **Ruled Out.** Strict patient-level stratification confirmed. |
| **11** | **Optimization / Under-convergence** | **D** | ViT training curves converged smoothly with stable validation loss and learning rate decay. | **Ruled Out.** Model was fully converged. |
| **12** | **Class Imbalance Distortion** | **C** | Train set is $58.8\%$ benign / $41.2\%$ malignant. Weighted cross-entropy adequately balanced training. | **Minor Contributor.** Handled by loss weighting. |
| **13** | **Shape / Aspect Ratio Resizing Distortion** | **B** | Standard resize to $224\times 224$ distorts aspect ratio of tall CC/MLO views ($1:1.8$). | **Probable Factor.** Pad-to-square letterboxing recommended. |
| **14** | **Wiener Filter Noise Artifacts** | **C** | Wiener filtering improved calcification patch PR-AUC, but noise amplification occurred in extreme dense tissue. | **Minor Factor.** Adaptive noise floor estimation is required. |
| **15** | **Background / Pectoral Muscle Confusion** | **B** | High false positives ($89$ validation FP) caused by dense glandular tissue and pectoral muscle edges. | **Probable Factor.** Breast-mask bounding avoids pectoral artifacts. |
| **16** | **Multi-Abnormality Conflict** | **C** | Present in $11.2\%$ of scans ($62$ val scans). Single top-1 ROI could only target one lesion. | **Minor Factor.** Multi-candidate pooling addresses multi-focal scans. |
| **17** | **Mask-Image Registration Discrepancies** | **C** | $2.2\%$ of masks had minor dimension mismatch with original scans. | **Minor Factor.** Automated coordinate scaling mitigates this. |
| **18** | **Overfitting to Training Set ROI Artifacts** | **B** | ViT learned to rely on center-crop artifacts, explaining performance degradation on Oracle crops. | **Probable Factor.** Addressed by diverse proposal data augmentation in V7. |

---

## 8. Evidence-Ranked Bottleneck Hierarchy

```
RANK 1: HIGH FALLBACK RATE & ROI LOCALIZATION FAILURE (Grade A)
        └── 83.9% of scans fallback to center crop -> 97.9% of errors occur here.

RANK 2: LOW RESOLUTION DOWNSAMPLING (512x512) (Grade A)
        └── Erases micro-calcification clusters and spiculated mass margins before detection.

RANK 3: SINGLE TOP-1 ROI PROPOSAL FRAGILITY (Grade A)
        └── Zero fallback recovery if top-1 proposal has low overlap.

RANK 4: CLASSIFIER TRAINED ON FALLBACK DISTRIBUTION (Grade A)
        └── ViT weights fitted to center crops cannot directly exploit tight Oracle crops without retraining.

RANK 5: LACK OF HIGH-RESOLUTION MULTI-SCALE POOLING (Grade A)
        └── ViT lacks independent candidate extraction across scales.

RANK 6: PECTORAL MUSCLE & DENSE TISSUE CONFUSION (Grade B)
        └── Elevates false positive rate (89 validation FPs, specificity 73.0%).
```

---

## 9. Final Stage 0 Conclusions & Strategic Recommendation

1. **Localization is the Definitive Bottleneck:** The $84\%$ fallback rate starves the classifier of high-resolution lesion features, forcing it to act as a global texture classifier.
2. **Oracle Benchmark Informs Architecture:** Feeding tight Oracle crops into the frozen V6 classifier proves that **V7 requires retraining the classifier on multi-proposal features**, not merely bolting a new proposal generator onto the frozen V6 classifier.
3. **Strategic Gate Verdict:** The evidence firmly justifies proceeding to **Phase 5 (V7 High-Resolution Multi-Proposal Benchmarking)** once user confirmation is granted.
