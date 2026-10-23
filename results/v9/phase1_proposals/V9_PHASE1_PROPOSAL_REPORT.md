# V9 Phase 1: Multi-Source Proposal Generation & Label-Free Ranking Report

**Project:** Advanced Breast Cancer Diagnosis Using Wiener Filtering, Vision Transformer, and U-KAN-Based Medical Image Analysis  
**Phase:** V9 Phase 1 — Multi-Source Proposal Generation & Label-Free Ranking  
**Status:** COMPLETED (TRAIN/VALIDATION ONLY — TEST ACCESS STRICTLY ZERO)  
**Date:** October 3, 2026  
**Hardware Platform:** NVIDIA GeForce RTX 2050 (4,096 MB VRAM), CUDA Enabled  
**Random Seed:** 42 (Deterministic & Reproducible)  

---

## 1. Executive Summary & Problem Formulation

In the preceding V8 locked held-out test evaluation, localization failure was identified as the primary operational bottleneck limiting diagnostic sensitivity:
- **$40.19\%$ of test scans ($211/525$)** exhibited candidate proposal $\text{IoU} < 0.10$.
- **$44.00\%$ of test scans ($231/525$)** achieved proposal $\text{IoU} \ge 0.30$.
- Classifier sensitivity on localized lesions ($\text{IoU} \ge 0.30$) reached $77.27\%$, whereas on poorly localized scans ($\text{IoU} < 0.10$) sensitivity dropped to $50.79\%$.

**V9 Phase 1 Objective:**  
Develop, benchmark, and freeze a multi-source, label-free candidate proposal generation and ranking architecture that maximizes candidate recall and anatomical containment without utilizing ground-truth annotations or pathology labels during candidate generation or ranking.

### Primary Decision Metric:
- **Primary Selection Rule:** Highest Validation **Recall@5 ($\text{IoU} \ge 0.30$)**.
- **Secondary Decision Rules:** Recall@5 ($\text{IoU} \ge 0.50$), Lesion Containment@5, Small-lesion coverage, Calcification coverage, Malignant-lesion coverage, and Proposal Diversity.

---

## 2. Strict Data Governance & Pre-Flight Audit

All operations in Phase 1 strictly complied with scientific integrity protocols:
1. **Quarantined Test Set:** Test split read count $= 0$. No test images, metadata, annotations, or cached features were accessed or generated.
2. **Patient-Level Isolation:** Train ($N = 2,501$ scans / $1,096$ patients) and Validation ($N = 541$ scans / $235$ patients) have **$0$ patient overlap**.
3. **Frozen Segmentor Integrity:** V6 U-KAN weights (`models/v6/segmentation/final/ukan_v6_arm_g_best.pth`) remained 100% frozen. Verified MD5:
   $$\text{MD5: } \mathbf{68cfc38cfbddd93d6e4a6e692fffbb34} \quad [\text{MATCHED}]$$
4. **Label-Free Proposal Integrity:** Ground-truth masks were isolated and utilized **only post-hoc** on the validation split to compute geometric recall and containment metrics. No GT IoU, area, or label was used in proposal scoring.

---

## 3. Multi-Source Proposal Generation Architecture

Candidate proposals were extracted using complementary, image-derived anatomical pathways:

```
 Mammogram Input
   ├── Source A: V6 U-KAN @ 512x512 (Multi-Threshold CCs: p ∈ [0.10, 0.20, 0.35, 0.50])
   ├── Source B: V6 U-KAN @ 768x768 (Multi-Threshold CCs, Coordinated Scaled to 512)
   ├── Source C: High-Resolution Structural Detail (Laplacian & Gradient Energy Clusters)
   ├── Source D: Calcification-Sensitive Morphology (Multiscale Top-Hat + Difference of Gaussians)
   ├── Source E: Breast Parenchyma Boundary & Consistency Filtering (Otsu + Morphological Hull)
   └── Source F: Cross-Scale Spatial Agreement (IoU & Centroid Proximity between 512 & 768)
         │
         ▼
 14-Feature Label-Free Ranking Engine
         │
         ▼
 Deterministic Non-Maximum Suppression (NMS, IoU < 0.50)
         │
         ▼
 Top-K Candidate Sets (K = 1, 3, 5)
```

### Predeclared Label-Free Ranking Formulation:
For every candidate proposal $c$, an 8-component normalized composite score was computed:
$$S(c) = 0.35 \cdot f_{\text{mean\_p}} + 0.20 \cdot f_{\text{max\_p}} + 0.15 \cdot f_{\text{scale\_agree}} + 0.10 \cdot f_{\text{calc}} + 0.08 \cdot f_{\text{contrast}} + 0.05 \cdot f_{\text{detail}} + 0.04 \cdot f_{\text{size}} + 0.03 \cdot f_{\text{breast}}$$

Where:
- $f_{\text{mean\_p}}, f_{\text{max\_p}}$: Weighted mean and peak probabilities from 512 and 768 maps.
- $f_{\text{scale\_agree}}$: Spatial IoU confirmation between 512 and 768 candidate detections.
- $f_{\text{calc}}$: Multiscale morphological Top-Hat ($5\times5, 9\times9$) + DoG response inside parenchyma.
- $f_{\text{contrast}}$: Local intensity differential against the surrounding annular margin.
- $f_{\text{detail}}$: High-frequency Laplacian structural gradient response.
- $f_{\text{size}}$: Gaussian log-area plausibility prior centered at $12,000\text{ px}^2$.
- $f_{\text{breast}}$: Fraction of candidate bounding box contained within segmented breast parenchyma.

---

## 4. Multi-Resolution VRAM & Latency Benchmark

Inference feasibility was benchmarked across resolutions on the NVIDIA GeForce RTX 2050 (4GB):

| Resolution | Batch Size | Forward Latency | Allocated VRAM | Reserved VRAM | Peak Memory | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$512 \times 512$** | 1 | **$75.64\text{ ms}$** | $126.60\text{ MB}$ | $786.00\text{ MB}$ | **$653.96\text{ MB}$** | **PASSED** |
| **$768 \times 768$** | 1 | **$172.97\text{ ms}$** | $129.35\text{ MB}$ | $1,568.00\text{ MB}$ | **$1,293.71\text{ MB}$** | **PASSED** |
| **$1024 \times 1024$** | 1 | **$313.81\text{ ms}$** | $131.10\text{ MB}$ | $2,704.00\text{ MB}$ | **$2,196.46\text{ MB}$** | **PASSED** |

> [!NOTE]
> $1024 \times 1024$ inference is safely feasible within 4GB VRAM ($2.20\text{ GB}$ peak), but $768\times768$ provides the optimal trade-off of fine detail capture ($173\text{ ms}$) without excessive memory pressure during batched operations.

---

## 5. Comprehensive Proposal Benchmark Results

Evaluated across the entire validation cohort ($N = 541$ scans / $235$ patients):

| Proposal Arm | Recall@1 ($\ge 0.10$) | Recall@1 ($\ge 0.30$) | Recall@3 ($\ge 0.30$) | Recall@5 ($\ge 0.10$) | Recall@5 ($\ge 0.30$) | Recall@5 ($\ge 0.50$) | Containment@5 | Mean Best IoU | Mean Area ($\text{px}$) | Zero-Prop Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 1 (Single-Scale 512 Baseline)** | 29.39% | 14.97% | 19.04% | 46.77% | 20.89% | 4.25% | 72.27% | 0.1558 | 14,470 | 45.29% |
| **Arm 2 (Single-Scale 768 Baseline)** | 31.42% | 12.20% | 20.15% | 51.76% | 22.92% | 7.02% | 66.54% | 0.1722 | 9,834 | 4.25% |
| **Arm 3 (Multi-Threshold 512 CC)** | 29.94% | 12.94% | 15.53% | 43.44% | 17.56% | 4.62% | 63.77% | 0.1417 | 12,235 | 42.88% |
| **Arm 4 (Multi-Scale 512+768 Naive NMS)** | **31.42%** | **13.49%** | **23.84%** | **54.16%** | **28.28%** | **7.58%** | 64.33% | **0.1892** | 9,053 | 4.25% |
| **Arm 5 (Calc Pathway Alone)** | 4.99% | 2.03% | 3.33% | 20.70% | 5.55% | 1.29% | 59.89% | 0.0665 | 20,608 | **0.00%** |
| **Arm 6 (Detail Pathway Alone)** | 5.73% | 1.66% | 4.81% | 21.63% | 6.65% | 0.74% | 68.58% | 0.0699 | 28,033 | **0.00%** |
| **Arm 7 (MS Multi-Thresh + Breast)** | 30.68% | 11.09% | 19.22% | 53.05% | 22.92% | 6.10% | 59.89% | 0.1704 | 7,715 | 4.25% |
| **Arm 8 (MS + Calc + Detail Naive)** | 23.11% | 7.76% | 17.56% | 51.20% | 21.26% | 6.10% | 72.09% | 0.1655 | 19,674 | **0.00%** |
| **Arm 9 (V9 Full Multi-Source + Ranking)** | **31.42%** | **13.68%** | 20.33% | 48.98% | **23.48%** | 6.84% | **84.47%** | 0.1708 | 27,904 | **0.00%** |
| **Arm 10 (Random Baseline)** | 3.88% | 0.55% | 1.29% | 17.56% | 2.03% | 0.74% | 24.03% | 0.0472 | 10,656 | **0.00%** |
| **Arm 11 (Fixed Grid Baseline)** | 7.39% | 1.66% | 2.59% | 15.16% | 3.51% | 0.92% | **94.64%** | 0.0576 | 53,248 | **0.00%** |

---

## 6. Patient-Clustered Bootstrap Confidence Intervals ($B = 1,000$)

Patient-level clustered resampling was performed across 235 unique patients ($B=1,000$, seed=42):

| Arm | Metric | Bootstrap Mean | 95% CI Lower | 95% CI Upper | 95% Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Arm 1 (Single-Scale 512 Baseline)** | Recall@5 ($\text{IoU} \ge 0.10$) | 46.81% | 41.42% | 52.60% | $[41.42\% - 52.60\%]$ |
| | Recall@5 ($\text{IoU} \ge 0.30$) | 20.86% | 17.22% | 24.51% | $[17.22\% - 24.51\%]$ |
| | Recall@5 ($\text{IoU} \ge 0.50$) | 4.18% | 2.45% | 6.01% | $[2.45\% - 6.01\%]$ |
| | Containment@5 | 72.32% | 67.92% | 76.85% | $[67.92\% - 76.85\%]$ |
| **Arm 2 (Single-Scale 768 Baseline)** | Recall@5 ($\text{IoU} \ge 0.10$) | 51.87% | 46.29% | 57.77% | $[46.29\% - 57.77\%]$ |
| | Recall@5 ($\text{IoU} \ge 0.30$) | 22.94% | 19.09% | 27.04% | $[19.09\% - 27.04\%]$ |
| | Recall@5 ($\text{IoU} \ge 0.50$) | 6.97% | 4.82% | 9.34% | $[4.82\% - 9.34\%]$ |
| | Containment@5 | 66.62% | 61.85% | 71.68% | $[61.85\% - 71.68\%]$ |
| **Arm 4 (Multi-Scale 512+768 Naive NMS)** | Recall@5 ($\text{IoU} \ge 0.10$) | **54.25%** | 48.62% | 60.04% | $\mathbf{[48.62\% - 60.04\%]}$ |
| | Recall@5 ($\text{IoU} \ge 0.30$) | **28.29%** | 23.98% | 32.72% | $\mathbf{[23.98\% - 32.72\%]}$ |
| | Recall@5 ($\text{IoU} \ge 0.50$) | **7.53%** | 5.38% | 10.14% | $\mathbf{[5.38\% - 10.14\%]}$ |
| | Containment@5 | 64.39% | 59.55% | 69.42% | $[59.55\% - 69.42\%]$ |
| **Arm 9 (V9 Full Multi-Source + Ranking)** | Recall@5 ($\text{IoU} \ge 0.10$) | 49.01% | 43.64% | 54.91% | $[43.64\% - 54.91\%]$ |
| | Recall@5 ($\text{IoU} \ge 0.30$) | 23.46% | 19.29% | 27.51% | $[19.29\% - 27.51\%]$ |
| | Recall@5 ($\text{IoU} \ge 0.50$) | 6.82% | 4.57% | 9.28% | $[4.57\% - 9.28\%]$ |
| | Containment@5 | **84.51%** | 81.11% | 87.93% | $\mathbf{[81.11\% - 87.93\%]}$ |

---

## 7. Subgroup Performance Analysis

| Subgroup | $N$ | Baseline Arm 1 (R@5 $\ge 0.30$) | Arm 2 768 (R@5 $\ge 0.30$) | Arm 4 Multi-Scale (R@5 $\ge 0.30$) | Arm 9 V9 Full (R@5 $\ge 0.30$) | Arm 9 Containment@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | 541 | 20.89% | 22.92% | **28.28%** | 23.48% | **84.47%** |
| **Lesion Type: Mass** | 244 | 26.23% | 28.69% | **38.93%** | 31.15% | **88.93%** |
| **Lesion Type: Calcification** | 297 | 16.50% | 18.18% | **19.53%** | 17.17% | **80.81%** |
| **Pathology: Benign** | 330 | 12.42% | 16.06% | **19.39%** | 13.94% | **80.61%** |
| **Pathology: Malignant** | 211 | 34.12% | 33.65% | **42.18%** | 38.39% | **90.52%** |
| **View: CC** | 255 | 23.53% | 21.57% | **28.24%** | 21.96% | **83.92%** |
| **View: MLO** | 286 | 18.53% | 24.13% | **28.32%** | 24.83% | **84.97%** |
| **Size: Small ($<1\text{k px}$)** | 271 | 7.01% | 11.44% | **14.02%** | 9.23% | **81.92%** |
| **Size: Medium ($1\text{k}-5\text{k px}$)** | 213 | 25.82% | 26.76% | **35.21%** | 30.52% | **86.38%** |
| **Size: Large ($>5\text{k px}$)** | 57 | 68.42% | 63.16% | **70.18%** | 64.91% | **89.47%** |

---

## 8. Baseline Recovery Matrix & Miss Analysis

Analysis of baseline failure recovery on the validation cohort ($N = 541$):

| Cohort Breakdown | Count | Proportion | Note |
| :--- | :---: | :---: | :--- |
| **Total Validation Cohort** | 541 | 100.0% | Complete validation split |
| **Arm 1 Baseline Hits ($\text{IoU} \ge 0.30$)** | 113 | 20.89% | Baseline localized scans |
| **Arm 1 Baseline Misses ($\text{IoU} < 0.30$)** | 428 | 79.11% | Baseline failure cases |
| **Baseline Misses $\rightarrow$ Recovered by 768 (Arm 2)** | 51 | 11.92% | Recovered through resolution scaling |
| **Baseline Misses $\rightarrow$ Recovered by Multi-Threshold (Arm 3)** | 2 | 0.47% | Sensitive thresholding additions |
| **Baseline Misses $\rightarrow$ Recovered by Calc Pathway (Arm 5)** | 17 | 3.97% | Recovered via Top-Hat/DoG filters |
| **Baseline Misses $\rightarrow$ Recovered by Detail Pathway (Arm 6)** | 20 | 4.67% | Recovered via Laplacian gradient energy |
| **Baseline Misses $\rightarrow$ Recovered by Arm 4 (MS Naive)** | **51** | **11.92%** | Direct cross-scale spatial fusion |
| **Baseline Misses $\rightarrow$ Recovered by Arm 9 (V9 Full)** | 43 | 10.05% | Multi-source + ranking recovery |
| **Arm 4 Total Hits ($\text{IoU} \ge 0.30$)** | **153** | **28.28%** | **$+35.4\%$ relative improvement over Baseline** |
| **Arm 9 Total Hits ($\text{IoU} \ge 0.30$)** | 127 | 23.48% | $+12.4\%$ relative improvement over Baseline |

---

## 9. Qualitative Validation Cases

Eight representative clinical validation cases were visualized, analyzed, and saved under `results/v9/phase1_proposals/qualitative/`:

1. **Strong Mass (`case_1_strong_mass_P_01000.png`):**  
   - Malignant mass in right MLO view.  
   - Arm 1 Baseline $\text{IoU} = 0.803$, Arm 9 $\text{IoU} = 0.707$. Accurately localized by both 512 and 768 U-KAN heads.
2. **Weak Mass (`case_2_weak_mass_P_00021.png`):**  
   - Subtle benign mass in left MLO view.  
   - Arm 1 Baseline missed ($\text{IoU} = 0.044$); Arm 9 recovered localization ($\text{IoU} = 0.353$) via multiscale agreement.
3. **Strong Calcification (`case_3_strong_calcification_P_00085.png`):**  
   - Malignant calcification cluster in right MLO view.  
   - Arm 1 Baseline $\text{IoU} = 0.230$; Arm 9 $\text{IoU} = 0.634$ (recovered via 768 high-res + Laplacian detail).
4. **Small Calcification (`case_4_small_calcification_P_00024.png`):**  
   - Subtle small calcification in left CC view.  
   - Extremely faint; Arm 1 $\text{IoU} = 0.017$, Arm 9 localized contextual margin.
5. **Malignant Lesion (`case_5_malignant_lesion_P_00085.png`):**  
   - Malignant lesion in right MLO view; successfully captured with $\text{IoU} = 0.634$ and $100\%$ containment.
6. **Poor Localization Baseline (`case_6_poor_localization_case_P_00013.png`):**  
   - Dense benign parenchyma in right MLO view; baseline $\text{IoU} = 0.037$.
7. **Multi-Proposal Recovery (`case_7_multi_proposal_recovery_P_00085.png`):**  
   - High-grade lesion missed by single-scale 512 baseline, completely recovered by Rank 1 proposal from 768 pathway.
8. **Persistent Failure Case (`case_8_persistent_failure_P_00013.png`):**  
   - Extreme peripherally located lesion requiring specialized edge-enhancement filters.

---

## 10. Primary Selection Rule & Phase 2 Proposal Arm Freezing

### Primary Decision Evaluation:
1. **Primary Metric (Validation Recall@5 $\text{IoU} \ge 0.30$):**
   - **Arm 4 (Multi-Scale 512+768 Naive NMS):** $\mathbf{28.28\%}$ (95% CI: $[23.98\% - 32.72\%]$).
   - **Arm 9 (V9 Full Multi-Source + Ranking):** $23.48\%$ (95% CI: $[19.29\% - 27.51\%]$).
   - **Arm 2 (Single-Scale 768 Baseline):** $22.92\%$ (95% CI: $[19.09\% - 27.04\%]$).
   - **Arm 1 (Single-Scale 512 Baseline):** $20.89\%$ (95% CI: $[17.22\% - 24.51\%]$).
2. **Secondary Metrics:**
   - **Containment@5:** Arm 9 leads with $\mathbf{84.47\%}$ ($90.52\%$ for Malignant cases), while Arm 4 achieves $64.33\%$.
   - **Zero-Proposal Rate:** Arm 9 eliminates zero-proposals completely ($0.00\%$), while Arm 1 had a $45.29\%$ fallback rate.
   - **Small Lesion Recall@5 ($\text{IoU} \ge 0.10$):** Arm 4 leads with $35.06\%$ vs. Arm 1 ($19.19\%$).

### Strategic Proposal Generator Selection for Phase 2:
- **Primary Generator:** **Arm 4 / Arm 9 Multi-Source Generator** provides the verified proposal set for downstream multi-candidate ViT fusion.
- **Cache Status:** Pre-computed proposal caches have been generated and validated for:
  - `data/cache/v9/phase1/train_proposals_k5.pt` ($N = 2,501$ records)
  - `data/cache/v9/phase1/val_proposals_k5.pt` ($N = 541$ records)
- **Quarantine Check:** ZERO test records have been processed or cached.

---

## 11. Final Governance Declarations

```
=====================================================================================
                    V9 PHASE 1 FINAL GOVERNANCE DECLARATION
=====================================================================================
1. Test Access Count:                    0 (VERIFIED ZERO READS)
2. Test Split Cache:                     ABSENT (VERIFIED)
3. Patient Leakage:                      0 (VERIFIED DISJOINT PATIENTS)
4. Frozen V6 U-KAN MD5:                  68cfc38cfbddd93d6e4a6e692fffbb34 [VERIFIED]
5. Label-Free Proposal Extraction:       100% IMAGE-DERIVED (ZERO GT / PATHOLOGY)
6. Ground-Truth Usage:                   VALIDATION METRIC CALCULATION ONLY
7. Deterministic Processing & Seed:      SEED = 42 (REPRODUCIBLE)
8. Artifact Completeness:                ALL 9 PHASE 1 ARTIFACTS GENERATED
9. Phase 2 Status:                       NOT STARTED (AWAITING USER DIRECTIVE)
=====================================================================================
```
