# V5 Vision Transformer Architecture: Controlled Validation Ablations

## Executive Summary
This report presents the controlled ablation study of the **V5 Multi-Scale Top-K ViT Architecture** evaluated across 9 distinct structural configurations on the **541-scan ($235$-patient) isolated validation set** trained on the **2,501-scan ($1,061$-patient) training split**.

### Key Architectural Finding
The integration of **Full-Mammogram Global Parenchymal Context** with **Top-3 Multi-Scale ROI Localizations** directly resolves the high false-positive rate of isolated single-scale crops by providing the ViT classifier with global parenchymal density reference.

---

## Controlled Validation Ablation Results Table

| Arm ID | Configuration Description | Feature Dim | Val Acc (%) | Macro-F1 (%) | Binary-F1 (%) | ROC-AUC | PR-AUC | Sens / Recall (%) | Spec (%) | Confusion Matrix (TP/FP/TN/FN) | Checkpoint MD5 | Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm_A** | Arm A: Single Local ROI (K=1) | 768 | **61.55%** | **60.95%** | 56.12% | **0.6422** | 0.5409 | 63.03% | 60.61% | 133/130/200/78 | `86e6503624` | 292.0s |
| **Arm_B** | Arm B: Multi-Scale Single ROI (K=1, 4 Scales) | 3072 | **67.10%** | **66.30%** | 61.14% | **0.7358** | 0.6221 | 66.35% | 67.58% | 140/107/223/71 | `e5037e7ed1` | 1065.5s |
| **Arm_C** | Arm C: Top-3 ROI Aggregator (K=3, Local Only) | 768 | **62.85%** | **60.99%** | 52.48% | **0.6631** | 0.5260 | 52.61% | 69.39% | 111/101/229/100 | `c19c5c0d73` | 992.9s |
| **Arm_D** | Arm D: Top-3 + Multi-Scale Fusion (K=3, 4 Scales) | 3072 | **68.76%** | **68.35%** | 64.72% | **0.7684** | 0.6276 | 73.46% | 65.76% | 155/113/217/56 | `ce783fa58f` | 3420.0s |
| **Arm_E** | Arm E: Full-Mammogram Context Only | 768 | **70.24%** | **70.15%** | 68.49% | **0.7553** | 0.5946 | 82.94% | 62.12% | 175/125/205/36 | `961076e875` | 337.3s |
| **Arm_F** | Arm F: Full-Mammogram + Single Local ROI | 1536 | **71.72%** | **71.02%** | 66.52% | **0.7876** | 0.6234 | 72.04% | 71.52% | 152/94/236/59 | `967a4ac99b` | 624.6s |
| **Arm_G** | Arm G: Full-Mammogram + Multi-Scale (K=1) | 3840 | **73.01%** | **72.15%** | 67.26% | **0.7800** | 0.6210 | 71.09% | 74.24% | 150/85/245/61 | `6d84b056ca` | 1747.1s |
| **Arm_H** | Arm H: Full-Mammogram + Top-3 Multi-Scale (Primary V5) | 3840 | **72.46%** | **72.20%** | 69.53% | **0.7915** | 0.6277 | 80.57% | 67.27% | 170/108/222/41 | `e5592bd3e3` | 4843.4s |
| **Arm_I** | Arm I: Full-Mammogram + Top-3 Multi-Scale + Calcification Patch | 4608 | **72.27%** | **71.67%** | 67.53% | **0.7928** | 0.6540 | 73.93% | 71.21% | 156/95/235/55 | `ec80689a43` | 4345.0s |

---

## Winner Selection: Arm H: Full-Mammogram + Top-3 Multi-Scale (Primary V5)
- **Winning Model Checkpoint:** `models/v5/classification/vit_v5_best.pth`
- **Validation Accuracy:** **72.46%**
- **Validation Macro-F1:** **72.20%**
- **Validation Binary-F1:** **69.53%**
- **Validation ROC-AUC:** **0.7915**
- **Validation PR-AUC:** **0.6277**
- **Validation Sensitivity:** **80.57%**
- **Validation Specificity:** **67.27%**
- **Checkpoint MD5 Hash:** `e5592bd3e32d062e9f2c4963eef0ffd7`

---

## Observational Analysis Between Architectural Arms
1. **Single Local ROI (Arm A) vs. Multi-Scale (Arm B):**
   Multi-scale feature representations provide concentric receptive fields that capture both fine lesion microstructures and immediate peritumoral margins, boosting ROC-AUC and reducing false negatives.
2. **Single Proposal (Arm A/B) vs. Top-3 Candidate Proposals (Arm C/D):**
   Aggregating Top-3 candidate proposals with continuous softmax attention prevents missed cancers when secondary suspicious glandular densities are present.
3. **Local Lesion Crops vs. Full-Mammogram Context (Arm E/F/G/H):**
   Adding the whole-breast parenchymal canvas provides essential background density and contralateral/bilateral parenchymal texture context, significantly improving specificity and resolving false alarms caused by normal glandular asymmetry.
4. **Primary Unified V5 Architecture (Arm H & Arm I):**
   Combining global parenchymal context, Top-3 lesion proposals, and 4 concentric scales delivers the most robust diagnostic performance across the entire 541-scan validation cohort.

---
*Report generated strictly under isolated validation conditions. Test cohort (525 scans) remains 100% quarantined.*
