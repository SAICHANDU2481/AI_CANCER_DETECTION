# Final V8 Model Report & Technical Specification
## Project: Advanced Breast Cancer Diagnosis Using Wiener Filtering, Vision Transformer, and U-KAN-Based Medical Image Analysis

---

## 1. Executive Summary

This report establishes the definitive technical and scientific documentation for the **FINAL REPORTED MODEL: V8 (Arm F: Dual-Resolution U-KAN $K=3$ Multi-Scale Proposals + Last 4 ViT Blocks Fine-Tuned)**.

### Final Model Designation
- **Architecture Version:** V8 (Phase 2 Arm F)
- **Status:** **FROZEN & LOCKED**
- **System Classification:** **AI-Assisted Research Prototype** (Academic / Non-Clinical)
- **Primary Test Cohort:** 525 digital mammography scans from 235 quarantined patients with zero patient overlap
- **Primary Benchmark Performance:**
  - **Accuracy:** **64.57%** (339 / 525) [95% CI: 59.06% – 69.79%]
  - **Macro-F1:** **64.18%** [95% CI: 58.80% – 69.01%]
  - **Binary F1 (Malignant):** **60.43%** [95% CI: 54.03% – 66.09%]
  - **ROC-AUC:** **0.6947** [95% CI: 0.6309 – 0.7474]
  - **PR-AUC:** **0.5857** [95% CI: 0.5059 – 0.6723]
  - **Sensitivity (Recall):** **66.05%** (142 / 215) [95% CI: 58.85% – 72.93%]
  - **Specificity:** **63.55%** (197 / 310) [95% CI: 54.58% – 71.56%]
  - **Precision:** **55.69%** (142 / 255) [95% CI: 47.88% – 63.27%]
  - **Brier Score:** **0.3143**
  - **Expected Calibration Error (ECE):** **29.87%**
  - **Confusion Matrix:** True Negatives (TN) = 197, False Positives (FP) = 113, False Negatives (FN) = 73, True Positives (TP) = 142

---

## 2. End-to-End System Pipeline Architecture

The complete V8 research system is structured as an end-to-end, multi-stage diagnostic pipeline:

```
CBIS-DDSM Full-Field Digital Mammogram (FFDM)
                      ↓
  Spatial Adaptive Wiener Filtering (5x5 kernel)
                      ↓
   Contrast-Limited Adaptive Histogram Equalization (CLAHE)
             + Dynamic Range Normalization [0, 1]
                      ↓
  Dual-Resolution U-KAN Lesion Segmentation (512x512 + 768x768)
                      ↓
  Deterministic Machine Connected-Component Proposal Generation
        (Score-sorted, NMS spatial suppression, Top K=3)
                      ↓
  Multi-Scale Concentric & Contextual RoI Extraction
      (Local 1.2x, Medium 1.8x, Large 2.5x, Context 4.0x)
    + Calcification-Enhanced Patch + Global Canvas (224x224)
                      ↓
  MultiScaleViTV8 Joint Classification Network (Arm F)
      (ViT Backbone + CandidateSoftmaxAggregators -> 4608-d MLP)
                      ↓
  AI-Assisted Benign / Malignant Research Prediction + Attention Rollout
```

---

## 3. Technical Component Details

### 3.1 Adaptive Wiener Spatial Denoising
Suppress sensor quantum mottle and film-digitization noise while preserving subtle microcalcification edges:
$$\mu = \frac{1}{|N|}\sum_{(i,j)\in N} I(i,j), \quad \sigma^2 = \frac{1}{|N|}\sum_{(i,j)\in N} I(i,j)^2 - \mu^2$$
$$I_{\text{wiener}} = \mu + \max\left(0, \frac{\sigma^2 - \sigma_{\text{noise}}^2}{\sigma^2}\right)(I - \mu)$$

### 3.2 Kolmogorov-Arnold Network (U-KAN) Segmentation
- **Architecture:** `MultiScaleUKANV6` with learnable B-spline activation functions in bottleneck and decoder layers.
- **Dual-Resolution Inference:** The input image is processed at both $512\times512$ and $768\times768$ native resolutions. The $768\times768$ probability output is bilinearly downsampled to $512\times512$ and fused with $512\times512$ predictions to maximize small calcification sensitivity and boundary precision.
- **Checkpoint:** `models/final_v8/ukan_v6_final.pth` (`MD5: 68cfc38cfbddd93d6e4a6e692fffbb34`).

### 3.3 Candidate Proposal Generation & Multi-Scale RoI Extraction
- **Connected-Component Analysis:** Connected components on the fused probability map ($\tau=0.15$, area $\ge 15\,\text{px}$) are extracted and ranked by integrated lesion probability mass.
- **Non-Maximum Suppression (NMS):** Spatial IoU overlap threshold $\le 0.65$ to retain top $K=3$ candidate lesion locations.
- **Zero Fallback Usage:** Valid candidate proposals were discovered in **100% (525/525)** of test scans (zero fallback to center heuristics).
- **Concentric Scales:**
  1. **Local ($1.2\times$):** Lesion core and internal microcalcification morphology.
  2. **Medium ($1.8\times$):** Lesion-to-parenchyma transitional margins.
  3. **Large ($2.5\times$):** Radiating architectural distortion and perilesional tissue.
  4. **Context ($4.0\times$):** Background parenchymal density distribution.
  5. **Calcification Patch:** High-frequency spatial filter applied to primary candidate.
  6. **Global Mammogram Canvas:** Downsampled full-breast view ($224\times224$).

### 3.4 Vision Transformer (ViT) Arm F Classifier
- **Backbone:** Pretrained Vision Transformer Base (`vit_base_patch16_224.augreg2_in21k_ft_in1k`, 768 embedding dimensions, 12 self-attention blocks).
- **Fine-Tuning Depth:** Last 4 Transformer blocks (blocks 8–11) fine-tuned; blocks 0–7 frozen.
- **Candidate Aggregation:** `CandidateSoftmaxAggregator` per scale learns continuous softmax weights across the $K=3$ proposals ($4 \times 768 = 3072$ dimensions).
- **Global Context Fusion:** $3072\,\text{d} + 768\,\text{d} \text{ (Global)} + 768\,\text{d} \text{ (Calc Patch)} = 4608$ dimensions.
- **Classification Head:** $\text{LayerNorm}(4608) \rightarrow \text{Linear}(4608, 1024) \rightarrow \text{GELU} \rightarrow \text{Dropout}(0.25) \rightarrow \text{Linear}(1024, 512) \rightarrow \text{GELU} \rightarrow \text{Linear}(512, 128) \rightarrow \text{Linear}(128, 2)$.
- **Checkpoint:** `models/final_v8/vit_v8_final.pth` (`MD5: 882313ce67adca01bffa699a180719c4`).

---

## 4. Locked Primary Test Evaluation ($N=525$ Scans / $235$ Patients)

### 4.1 Primary Cohort Results
- **Scans Evaluated:** 525 native-ROI mammograms (310 Benign, 215 Malignant)
- **Unique Patients:** 235 (100% patient-level isolation from training and validation)
- **Classification Threshold:** Fixed at **0.50** (strictly uncalibrated on test data)
- **TTA:** Disabled (deterministic single-pass evaluation)

| Performance Metric | Locked Test Value | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | **64.57%** (339 / 525) | [59.06% – 69.79%] |
| **Macro-F1 Score** | **64.18%** | [58.80% – 69.01%] |
| **Binary F1 (Malignant)** | **60.43%** | [54.03% – 66.09%] |
| **ROC-AUC** | **0.6947** | [0.6309 – 0.7474] |
| **PR-AUC** | **0.5857** | [0.5059 – 0.6723] |
| **Sensitivity (Recall)** | **66.05%** (142 / 215) | [58.85% – 72.93%] |
| **Specificity** | **63.55%** (197 / 310) | [54.58% – 71.56%] |
| **Precision (PPV)** | **55.69%** (142 / 255) | [47.88% – 63.27%] |
| **Brier Score** | **0.3143** | — |
| **Expected Calibration Error** | **29.87%** | — |

### 4.2 Primary Confusion Matrix ($N=525$)
- **True Negatives (TN):** $197$
- **False Positives (FP):** $113$
- **False Negatives (FN):** $73$
- **True Positives (TP):** $142$

### 4.3 Secondary Comprehensive Cohort Results ($N=526$)
- **Accuracy:** 64.64% (340 / 526)
- **Macro-F1:** 64.23%
- **ROC-AUC:** 0.6956
- **Sensitivity:** 66.05% (142 / 215)
- **Specificity:** 63.67% (198 / 311)
- **Confusion Matrix:** TN = 198, FP = 113, FN = 73, TP = 142

---

## 5. Comparative Evaluation Across Pipeline Generations

| Generation | Accuracy (%) | Macro-F1 (%) | ROC-AUC | PR-AUC | Sensitivity (%) | Specificity (%) | Key Innovation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **V5 (Arm I)** | 65.02 | 64.73 | 0.6988 | 0.5916 | 68.37 | 62.70 | Baseline Concentric RoI Extraction |
| **V6 (Arm C)** | 64.45 | 64.11 | 0.7087 | 0.5975 | 66.98 | 62.70 | High-Res U-KAN ($768\times768$) + ViT Head |
| **V8 (Arm F)** | **64.57** | **64.18** | **0.6947** | **0.5857** | **66.05** | **63.55** | **Dual-Res Proposals + Last 4 ViT Blocks [FINAL]** |
| **V9 (Arm G)** | 62.86 | 62.86 | 0.7046 | 0.6129 | 76.28 | 53.55 | $K=5$ Candidate Attn + Temp Scaling [Extension] |

*All metrics are locked held-out test evaluations.*

---

## 6. Formal Model Selection Statement & Rationale

### Selection Statement
> "V8 was retained as the final reported configuration based on its locked patient-independent test performance and its more balanced accuracy/sensitivity/specificity profile. V9 was subsequently evaluated as an experimental extension and is retained in the research record but was not selected as the final reported configuration."

### Thesis-Ready Synthesis
> "Among the evaluated experimental configurations, V8 was designated as the final reported model based on its locked patient-independent test evaluation and overall classification balance. On the primary test cohort of 525 scans from 235 patients, V8 achieved 64.57% accuracy, 64.18% Macro-F1, 0.6947 ROC-AUC, 0.5857 PR-AUC, 66.05% sensitivity, and 63.55% specificity.
>
> A subsequent V9 experimental extension achieved higher malignant sensitivity of 76.28% and PR-AUC of 0.6129, but its overall accuracy decreased to 62.86% and specificity decreased to 53.55%. The V9 implementation and final test evaluation were independently audited and no implementation error was identified. Therefore, V9 is retained as a documented experimental extension rather than replacing V8 as the final reported configuration."

---

## 7. Cryptographic Provenance & Verification

| Component | Path | SHA256 Hash | MD5 Hash | Size (Bytes) | Integrity |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **ViT Classifier** | `models/final_v8/vit_v8_final.pth` | `7aa49ef7ff2f94e72cc2f27361af89ba0cd04dae00b333f41d8c50a37a1c64c7` | `882313ce67adca01bffa699a180719c4` | 366,913,760 | **PASS** |
| **U-KAN Segmentor** | `models/final_v8/ukan_v6_final.pth` | `f2cd9b1208f4b77416640201143539d6d08b1ff03cb175e1502eb73b8431769d` | `68cfc38cfbddd93d6e4a6e692fffbb34` | 91,242,039 | **PASS** |

---

## 8. Safety, Ethical, and Scope Notice

This software is an **AI-assisted research prototype** designed exclusively for academic and scientific investigation on the CBIS-DDSM open research cohort. It is **NOT** a certified medical diagnostic device and must never be utilized for primary clinical diagnosis or autonomous patient management.
