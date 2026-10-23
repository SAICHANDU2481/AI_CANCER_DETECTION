# Arm I Final Validation Report: Full-Mammogram + Top-3 Multi-Scale + Calcification Patch

## Executive Summary
This document provides the complete, isolated validation audit and performance analysis for **Arm I** ($4,608$-dimensional feature fusion) within the V5 Vision Transformer ablation suite. Training was conducted strictly on the **2,501-scan ($1,061$-patient) training split** and evaluated on the **541-scan ($235$-patient) isolated validation split**.

---

## 1. Arm I Architectural Configuration
- **Architecture Name:** Multi-Scale Vision Transformer V5 (Arm I)
- **Backbone:** `vit_base_patch16_224.augreg2_in21k_ft_in1k` (ImageNet-21k Pretrained ViT-Base)
- **Local Lesion Receptive Scales:** 4 Concentric Scales (`'local'` 1.2x, `'medium'` 1.8x, `'large'` 2.5x, `'context'` 4.0x)
- **Top-K Proposals:** $K = 3$ Candidates aggregated via Continuous Softmax Attention (`CandidateSoftmaxAggregator`)
- **Global Context:** Full-Mammogram 224x224 Parenchymal Canvas Enabled
- **Microcalcification Patch:** High-Frequency Calcification-Enhanced Patch ($224 \times 224$) Enabled with deterministic local fallback
- **Total Feature Fusion Dimension:** **4,608** ($768 \times 4$ ROI + $768$ Global Context + $768$ Calcification Patch)
- **Total Parameters:** **91,712,902**
- **Stage 1 Trainable Parameters:** **5,914,246**
- **Stage 2 Trainable Parameters:** **20,091,526**

---

## 2. Dataset & Cohort Demographics
- **Training Set:** **2,501 scans** across **1,061 patients** (`data/processed/train.csv`, cache: `F:\BreastCancer_V5_Cache\train\`)
- **Validation Set:** **541 scans** across **235 patients** (`data/processed/val.csv`, cache: `F:\BreastCancer_V5_Cache\val\`)
- **Patient Isolation:** **100% Guaranteed** (Zero patient overlap across cohorts)
- **Test Quarantine:** **100% Intact** (`data/processed/test.csv` untouched, 0 reads, `results/v5/test/` empty)

---

## 3. Best Epoch & Checkpoint Details
- **Best Stage:** **Stage 1** (Fusion Head + Softmax Aggregators Training)
- **Best Epoch:** **Epoch 4 / 8**
- **Best Checkpoint Path:** `models/v5/classification/vit_v5_arm_i_best.pth`
- **Checkpoint File Size:** `366,927,754 bytes` ($349.93$ MB)
- **Checkpoint MD5 Hash:** `ec80689a43a906f89a15d7148cf7b05e`
- **Checkpoint Status:** Verified, loadable, and saved on disk

---

## 4. Measured Validation Performance Metrics

| Metric | Measured Value | Benchmark Context |
| :--- | :---: | :--- |
| **Validation Accuracy** | **72.27%** | $391 / 541$ correct predictions |
| **Validation Macro-F1** | **71.67%** | Balanced harmonic mean |
| **Validation Binary-F1** | **67.53%** | Positive class (Malignant) F1 score |
| **ROC-AUC** | **0.7928** | Suite-leading discrimination ability ($0.7928$) |
| **PR-AUC** | **0.6540** | Highest Precision-Recall AUC across all 9 arms |
| **Malignant Recall / Sensitivity** | **73.93%** | $156 / 211$ true malignant cases detected |
| **Benign Specificity** | **71.21%** | $235 / 330$ true benign cases correctly ruled out |
| **Precision (PPV)** | **62.15%** | $156 / 251$ positive predictions confirmed malignant |

---

## 5. Confusion Matrix Analysis (541 Validation Scans)

$$\begin{pmatrix} \text{TN} = 235 & \text{FP} = 95 \\ \text{FN} = 55 & \text{TP} = 156 \end{pmatrix}$$

- **True Positives (TP):** 156
- **True Negatives (TN):** 235
- **False Positives (FP):** 95
- **False Negatives (FN):** 55

---

## 6. Training Runtime & Compute Telemetry
- **Total Training Runtime:** **4,345.0 seconds** ($72.4$ minutes)
- **Compute Hardware:** NVIDIA GeForce RTX 2050 Laptop GPU ($4,096$ MiB VRAM)
- **Peak VRAM Allocated:** `2,297 MiB` ($56.1\%$ capacity)
- **Precision:** Mixed Precision AMP FP16 (`torch.amp.autocast('cuda')`)
- **Loss Stability / NaN / Inf:** **0 NaN / Inf occurrences** (Loss and gradients fully finite throughout)
- **CUDA / OOM Errors:** **0 errors encountered**

---

## 7. Comparative Context Across 9-Arm Ablation Suite

| Arm | Description | Dim | Val Acc (%) | Macro-F1 (%) | ROC-AUC | PR-AUC | Sens (%) | Spec (%) | Runtime | MD5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A** | Single Local ROI | 768 | 61.55% | 60.95% | 0.6422 | 0.5409 | 63.03% | 60.61% | 292s | `86e65036` |
| **Arm B** | Multi-Scale Single ROI | 3072 | 67.10% | 66.30% | 0.7358 | 0.6221 | 66.35% | 67.58% | 1066s | `e5037e7e` |
| **Arm C** | Top-3 ROI Aggregator | 768 | 62.85% | 60.99% | 0.6631 | 0.5260 | 52.61% | 69.39% | 993s | `c19c5c0d` |
| **Arm D** | Top-3 Multi-Scale | 3072 | 68.76% | 68.35% | 0.7684 | 0.6276 | 73.46% | 65.76% | 3420s | `ce783fa5` |
| **Arm E** | Full-Mammogram Context | 768 | 70.24% | 70.15% | 0.7553 | 0.5946 | 82.94% | 62.12% | 337s | `961076e8` |
| **Arm F** | Full-Mammo + Local ROI | 1536 | 71.72% | 71.02% | 0.7876 | 0.6234 | 72.04% | 71.52% | 625s | `967a4ac9` |
| **Arm G** | Full-Mammo + Multi-Scale ($K=1$) | 3840 | **73.01%** | 72.15% | 0.7800 | 0.6210 | 71.09% | **74.24%** | 1747s | `6d84b056` |
| **Arm H** | Full-Mammo + Top-3 Multi-Scale | 3840 | 72.46% | **72.20%** | 0.7915 | 0.6277 | 80.57% | 67.27% | 4843s | `e5592bd3` |
| **Arm I** | Full-Mammo + Top-3 MS + Calc Patch | 4608 | 72.27% | 71.67% | **0.7928** | **0.6540** | 73.93% | 71.21% | 4345s | `ec80689a` |

---

## 8. Governance & Integrity Audit
- **Arms A–H Checkpoints:** Verified 100% intact with matching historical MD5 checksums.
- **Cache File Count:** Exactly 2,501 training NPZs and 541 validation NPZs on Drive F: (0 modified or deleted).
- **Test Quarantine:** Verified 0 reads of `data/processed/test.csv`, 0 test predictions generated, `results/v5/test/` empty.
