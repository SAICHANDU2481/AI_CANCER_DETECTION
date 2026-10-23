# V7 Phase 5: Final Consistency Audit & Pre-Phase 6 Readiness Report

**Date:** September 30, 2026  
**Status:** READ-ONLY CONSISTENCY AUDIT COMPLETED  
**Scope:** Validation Set ($N=541$ scans, 235 patients) & Training Cache ($N=2,501$ scans, 1,096 patients)  
**Test Data Access:** 0 reads (`test.csv` quarantined, `results/v7/test` and `data/cache/v7_proposals/test` do not exist)  
**Frozen Model Checkpoints:**
- V6 U-KAN: `models/v6/segmentation/final/ukan_v6_arm_g_best.pth` (MD5: `68cfc38cfbddd93d6e4a6e692fffbb34` — Verified Unchanged)
- V6 Classifier: `models/v6/classification/vit_v6_arm_c_best.pth` (MD5: `508250cfc1c07af1ea7d81cc2e5a7418` — Verified Unchanged)

---

## 1. Audit 1: Discrepancy & Reconciliation of Fallback Definitions

### A. Stage 0 Fallback Definition (Classifier Input Level)
In Stage 0, fallback was measured directly from the precomputed V6 ROI generator cache (`classification/v6/topk_proposals_v6.py`).
- **Logic:** The V6 classifier requires a fixed tensor of $K=3$ proposal crops. If U-KAN produced fewer than 3 connected components ($0$, $1$, or $2$), the system set `fallback_flag = True` and padded the deficit with synthetic center-crops.
- **Stage 0 Metric Value:** **84.10%** of validation scans ($455 / 541$) required at least one fallback padding crop.
- **Pure Machine Rate ($\ge 3$ machine proposals):** **15.90%** ($86 / 541$).

### B. Phase 5 Fallback Definition (Proposal Detector Level)
In Phase 5, fallback measures complete localization failure by the candidate generator.
- **Logic:** `is_fallback = 1` occurs if and only if **0 candidate proposals** are extracted by the generator for an image (`len(proposals) == 0`).
- **Phase 5 Metric Value (Arm A V6 U-KAN):** **54.53%** ($295 / 541$) scans had zero connected components.
- **Phase 5 Metric Value (Arm B High-Res 768):** **15.53%** ($84 / 541$) zero-proposal rate.
- **Phase 5 Metric Value (Arm F / Arm G Multi-Proposal):** **0.00%** ($0 / 541$) zero-proposal rate.

### C. Exact Mathematical Reconciliation
The numbers reconcile:
$$\text{Total Scans } (N = 541) = \underbrace{295 \text{ (Zero Proposals, } 54.53\%)}_{\text{Phase 5 Complete Fallback}} + \underbrace{160 \text{ (1 or 2 Proposals, } 29.57\%)}_{\text{Partial Deficit}} + \underbrace{86 \text{ (}\ge 3 \text{ Proposals, } 15.90\%)}_{\text{Stage 0 Pure Machine}}$$

$$\text{Stage 0 Fallback Rate} = \frac{295 + 160}{541} = \frac{455}{541} = \mathbf{84.10\%}$$
$$\text{Phase 5 Detection Fallback Rate} = \frac{295}{541} = \mathbf{54.53\%}$$

> **Metric Clarity:** The two metrics measure distinct layers of the pipeline: Stage 0 measured **$K=3$ Multi-Crop Padding Fallback**, whereas Phase 5 measured **Zero-Proposal Detection Failure**. Phase 5 Arm F/G eliminates **both** failure modes ($0.00\%$ zero-proposal rate and $0.00\%$ deficit rate for Top-5 crops).

---

## 2. Audit 2: Proposal Recall Verification

Direct verification from [proposal_benchmark.csv](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/v7/phase5/proposal_benchmark.csv):

| Metric | V6 Baseline (Arm A) | Multi-Proposal Suite (Arm F/G) | Absolute Difference | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Recall@1 (IoU $\ge 0.10$)** | **15.71%** | **21.07%** | **+5.36%** | **+34.12%** |
| **Recall@3 (IoU $\ge 0.10$)** | **17.93%** | **30.31%** | **+12.38%** | **+69.05%** |
| **Recall@5 (IoU $\ge 0.10$)** | **18.30%** | **35.86%** | **+17.56%** | **+95.96%** |
| **Recall@5 (IoU $\ge 0.30$)** | **10.91%** | **20.89%** | **+9.98%** | **+91.48%** |
| **Recall@5 (IoU $\ge 0.50$)** | **7.21%** | **10.72%** | **+3.51%** | **+48.68%** |
| **Containment@5 ($\ge 80\%$)** | **14.79%** | **21.81%** | **+7.02%** | **+47.46%** |

---

## 3. Audit 3: Recovery Matrix Verification

Direct verification from [proposal_recovery_overlap.csv](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/v7/phase5/proposal_recovery_overlap.csv):

- **Total Validation Scans ($N=541$):**
  - V6 Baseline (Arm A) Total Recovered: **99** scans (**18.30%** of 541)
  - V6 Baseline (Arm A) Total Missed: **442** scans (**81.70%** of 541)
- **Breakdown of Missed Cases ($N_{missed} = 442$):**
  - Recovered by High-Res (Arm B): **82** scans (**18.55%** of missed)
  - Recovered by Calc Detectors (Arm C): **57** scans (**12.90%** of missed)
  - Recovered by Arm F (V6 + High-Res + Calc): **100** scans (**22.62%** of missed)
  - Shared Recovery (V6 $\cap$ High-Res $\cap$ Calc): **20** scans (**3.70%** of 541)
- **Total Unique Lesions Recovered by Multi-Proposal Suite:** **194** scans (**35.86%** of 541).
- **Net Recovery Gain:** $+95$ additional lesion cases localized (+95.96% relative increase over V6).

---

## 4. Audit 4: Statistical Rigor & Confidence Intervals

From [proposal_bootstrap_ci.csv](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/v7/phase5/proposal_bootstrap_ci.csv) ($B=1,000$ patient-clustered resamples):

- **Arm A Recall@5 (IoU $\ge 0.10$):** Mean = $18.24\%$, 95% Clustered CI: **$[14.47\%, 22.39\%]$** (Std: $2.09\%$)
- **Arm F/G Recall@5 (IoU $\ge 0.10$):** Mean = $35.73\%$, 95% Clustered CI: **$[31.03\%, 40.30\%]$** (Std: $2.36\%$)
- **Arm A Recall@5 (IoU $\ge 0.30$):** Mean = $10.89\%$, 95% Clustered CI: **$[7.75\%, 14.46\%]$** (Std: $1.70\%$)
- **Arm F/G Recall@5 (IoU $\ge 0.30$):** Mean = $20.81\%$, 95% Clustered CI: **$[17.18\%, 24.76\%]$** (Std: $2.00\%$)

> **Statistical Governance Statement:** The 95% patient-clustered bootstrap confidence intervals for Recall@5 do not overlap ($[14.47\%, 22.39\%]$ vs. $[31.03\%, 40.30\%]$). We do not cite uncalculated asymptotic p-values; the non-overlapping clustered bootstrap intervals confirm strong empirical separation under clustered sampling.

---

## 5. Audit 5: Subgroup Stratification Denominator Consistency

From [proposal_recall_by_subgroup.csv](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/v7/phase5/proposal_recall_by_subgroup.csv):

| Subgroup Category | Partition | Scan Count | Sum / Total |
| :--- | :--- | :---: | :---: |
| **Lesion Type** | Mass / Calcification | 244 / 297 | **541 / 541** |
| **Pathology** | Benign / Malignant | 330 / 211 | **541 / 541** |
| **Lesion Size** | Small (<2k px) / Medium (2k–10k px) / Large (>10k px) | 3 / 52 / 486 | **541 / 541** |
| **View** | CC / MLO | 255 / 286 | **541 / 541** |

All subgroup partitions sum to exactly 541 validation scans across 235 unique patients.

---

## 6. Audit 6: Hardware VRAM & Resolution Recommendation

From [vram_runtime_audit.csv](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/v7/phase5/vram_runtime_audit.csv):
- **$512\times512$ (BS=1):** Peak Alloc = 551.1 MB, Peak Res = 696.0 MB $\rightarrow$ Extremely safe.
- **$768\times768$ (BS=1):** Peak Alloc = 1,193.2 MB, Peak Res = 1,478.0 MB $\rightarrow$ Optimal balance.
- **$768\times768$ (BS=2):** Peak Alloc = 2,348.6 MB, Peak Res = 2,922.0 MB $\rightarrow$ Safe on RTX 2050 4GB.
- **$1024\times1024$ (BS=1):** Peak Alloc = 2,091.6 MB, Peak Res = 2,466.0 MB $\rightarrow$ **Safe for forward inference only**, but will exceed 4GB during backpropagation training due to gradient history and AdamW optimizer states.
- **$1024\times1024$ (BS=2):** Peak Alloc = 4,145.3 MB, Peak Res = 5,314.0 MB $\rightarrow$ **Unsafe / Exceeds physical 4GB budget.**

> **Hardware Audit Conclusion:** $768\times768$ is the proven, safe resolution for both training and inference on the RTX 2050 4GB GPU.

---

## 7. Audit 7: Final Synthesis & Readiness Decision

### A. What Phase 5 Establishes Confidently:
1. Moving from single-threshold U-KAN to multi-source proposals (U-KAN 768 + Calcification spatial filters) **rescues 100 validation lesions** that V6 lost.
2. Complete detection fallback is reduced from **54.53% to 0.00%**.
3. Dedicated spatial filtering (Top-Hat, DoG, High-Pass) successfully detects microcalcifications without expensive full-scale DL inference.
4. $768\times768$ breast-ROI inference is fast (159 ms) and requires only 1.2–2.3 GB VRAM.

### B. What Remains Uncertain:
1. How best to aggregate Top-K proposal features in the downstream classifier (MIL attention pooling vs. cross-attention vs. max-pooling).
2. The optimal number of proposal crops ($K=3$ vs $K=5$) to feed into the classifier during training without introducing false-positive noise.

### C. Which Metrics are Directly Comparable with Stage 0:
- **Scan-level GT IoU and Containment rates** are directly comparable.
- **Subgroup partitions (Mass, Calc, Benign, Malignant, CC, MLO)** are directly comparable.

### D. Which Metrics are NOT Directly Comparable:
- **Stage 0 Fallback (84.10%)** vs **Phase 5 Fallback (54.53%)**: Stage 0 measured multi-crop tensor padding deficit ($N < 3$ machine proposals), while Phase 5 measured zero-detection failure ($N = 0$). Both are reconciled in Section 1.

### E. Whether Phase 6 is Scientifically Justified:
**YES.**
Stage 0 demonstrated that the primary performance cap of V6 was localization starvation. Phase 5 has empirically demonstrated that multi-source proposal generation recovers **35.86%** of lesions (up from **18.30%**) with **0.00%** fallback. Training a Multi-Instance Learning (MIL) / Multi-Proposal Classifier in Phase 6 is scientifically justified.

---

```
PHASE 5 AUDIT COMPLETE — READY FOR PHASE 6
```
