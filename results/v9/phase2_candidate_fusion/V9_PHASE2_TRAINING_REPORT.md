# V9 Phase 2: Multi-Candidate Classification, Attention Pooling & Multi-Source Fusion — Training Report

**Project:** Advanced Breast Cancer Diagnosis Using Wiener Filtering, Vision Transformer, and U-KAN-Based Medical Image Analysis  
**Phase:** V9 Phase 2 — Multi-Candidate Classification, Attention Pooling & Multi-Source Fusion  
**Status:** COMPLETED & LOCKED (TRAIN/VALIDATION ONLY — ZERO TEST ACCESS)  
**Date:** October 3, 2026  
**Hardware Platform:** NVIDIA GeForce RTX 2050 (4,096 MB VRAM), CUDA 12.8 Enabled  
**Random Seed:** 42 (Strict Deterministic Execution)  

---

## 1. Executive Summary & Problem Formulation

In V8 held-out test evaluation, single-ROI classifier dependence was demonstrated to be highly fragile when upstream localization proposals were imperfect ($40.19\%$ of test scans had proposal $\text{IoU} < 0.10$, dropping classifier sensitivity to $50.79\%$).

**V9 Phase 2 Objective:**  
Investigate whether multi-candidate attention pooling, quality-gated weighting, and multi-source contextual fusion enable robust benign-vs-malignant classification under realistic, imperfect localization proposals.

### Key Architectural Hypothesis:
By allowing the model to attend dynamically across $K=5$ candidate proposals (weighted by image-derived quality priors) while simultaneously conditioning on global mammographic context and localized high-resolution calcification features, diagnostic accuracy and malignant sensitivity can be substantially improved without relying on perfect single-box localization.

---

## 2. Upstream Proposal Integration & Pre-Flight Governance

### 1. Primary Proposal Generator Selection:
In accordance with the predeclared Phase 1 decision rule (Highest Validation Recall@5 at $\text{IoU} \ge 0.30$), **ARM 4 (Multi-Scale 512+768 Cross-Scale Fusion)** was selected as the primary proposal engine:
- Validation Recall@5 ($\text{IoU} \ge 0.30$) $= 28.28\%$ (95% CI: $[23.98\% - 32.72\%]$)
- Validation Recall@5 ($\text{IoU} \ge 0.10$) $= 54.16\%$
- Validation Containment@5 $= 64.33\%$

### 2. Strict Data Governance:
- **Quarantined Test Set:** Held-out test set read count $= 0$. Zero test images, masks, metadata, or proposals were accessed.
- **Patient-Level Isolation:** Train ($N = 2,501$ scans / $1,096$ patients) and Validation ($N = 541$ scans / $235$ patients) have **$0$ patient overlap**.
- **Frozen Segmentor Integrity:** V6 U-KAN MD5 `68cfc38cfbddd93d6e4a6e692fffbb34` verified unchanged.
- **Proposal Quality Normalization:** Quality feature normalization vectors ($\mu_q, \sigma_q$) were computed strictly from the **Train split statistics**.

---

## 3. Controlled Ablation Ladder Architecture

All models utilized a shared pretrained Vision Transformer encoder (`vit_base_patch16_224.augreg2_in21k_ft_in1k`, embedding dim $= 768$):

```
                        Shared ViT Backbone (768-dim)
                                      │
  ┌───────────────────────────────────┼───────────────────────────────────┐
  ▼                                   ▼                                   ▼
Candidate Crops (K=5)         Full Mammogram (1x768)          Calcification Patch (1x768)
  │                                   │                                   │
Quality-Gated Attention (768)         │                                   │
  │                                   │                                   │
  └───────────────────────────────────┴───────────────────────────────────┘
                                      │
                     Multi-Pathway Concatenation (3072-dim)
                                      │
                         LayerNorm & Deep MLP Head
                                      │
                            Logits: [Benign, Malignant]
```

### Ablation Ladder Configurations:
- **Arm A (Single Top-1 Local ROI):** $K=1$, Frozen ViT backbone, Linear head ($768 \rightarrow 256 \rightarrow 2$).
- **Arm B (K=3 Proposals + Frozen ViT):** $K=3$, Mean pooling across candidates, Frozen ViT ($768 \rightarrow 256 \rightarrow 2$).
- **Arm C (K=5 Proposals + Frozen ViT):** $K=5$, Mean pooling across candidates, Frozen ViT ($768 \rightarrow 256 \rightarrow 2$).
- **Arm D (K=5 + Candidate Quality Attention):** $K=5$, Trainable quality attention module, Last 2 ViT blocks unfrozen ($768 \rightarrow 256 \rightarrow 2$).
- **Arm E (K=5 + Attention + Full Mammo Context):** $K=5$, Quality attention + Full mammogram context ($1536 \rightarrow 768 \rightarrow 256 \rightarrow 2$), Last 2 ViT blocks unfrozen.
- **Arm F (K=5 + Attention + Full Mammo + Calc Patch):** $K=5$, Attention + Full mammo + High-res calcification patch ($2304 \rightarrow 768 \rightarrow 256 \rightarrow 2$), Last 2 ViT blocks unfrozen.
- **Arm G (K=5 + Attention + Mammo + Calc + Context ROI):** $K=5$, Attention + Full mammo + Calc patch + $4.0\times$ Contextual ROI ($3072 \rightarrow 768 \rightarrow 256 \rightarrow 2$), Last 2 ViT blocks unfrozen.

---

## 4. Training Protocol & Hyperparameters

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Loss Function** | Weighted Cross-Entropy + Label Smoothing ($0.05$) | Mitigates overconfidence and addresses class imbalance |
| **Class Weights** | Benign: $0.8507$, Malignant: $1.2129$ | Inverse frequency weighted across training distribution |
| **Optimizer** | AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 0.01$) | Standard decoupled weight decay for ViT fine-tuning |
| **Head Learning Rate** | $1 \times 10^{-4}$ | Fast, stable convergence for attention and fusion layers |
| **Backbone Learning Rate**| $2 \times 10^{-5}$ | Conservative rate for unfrozen transformer blocks |
| **Learning Rate Schedule**| CosineAnnealingLR ($\eta_{\min} = 1 \times 10^{-6}$) | Smooth decay throughout training epochs |
| **Physical Batch Size** | $8$ | Sized for RTX 2050 4GB memory safety |
| **Gradient Accumulation**| $2$ steps | Yields **Effective Batch Size $= 16$** |
| **Mixed Precision** | FP16 AMP (`torch.amp.autocast`) | Accelerated tensor execution with gradient scaling |
| **Early Stopping** | Patience $= 8$ epochs on Validation Macro-F1 | Prevents overfitting; max $25$ epochs |

---

## 5. Arm-by-Arm Training Progression & Convergence

| Arm Code | Name | Best Epoch | Train Acc (%) | Val Acc (%) | Val Macro-F1 (%) | Val ROC-AUC | Total Epochs Run |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A** | Single Top-1 Local ROI | **Epoch 17** | 77.53% | 70.79% | 69.51% | 0.7267 | 25 (Full run) |
| **Arm B** | $K=3$ Proposals + Frozen ViT | **Epoch 4** | 65.45% | 72.09% | 69.96% | 0.7632 | 12 (Early stopped) |
| **Arm C** | $K=5$ Proposals + Frozen ViT | **Epoch 12** | 72.37% | 70.98% | 69.87% | 0.7638 | 20 (Early stopped) |
| **Arm D** | $K=5$ + Candidate Attention | **Epoch 3** | 69.25% | 69.69% | 69.18% | 0.7728 | 11 (Early stopped) |
| **Arm E** | $K=5$ + Attention + Full Mammo | **Epoch 2** | 70.93% | 75.42% | 74.01% | 0.8247 | 10 (Early stopped) |
| **Arm F** | $K=5$ + Attention + Mammo + Calc | **Epoch 5** | 87.72% | 75.60% | 73.29% | 0.7854 | 13 (Early stopped) |
| **Arm G** | $K=5$ + Attention + Mammo + Calc + Context | **Epoch 3** | 76.57% | 75.05% | **74.65%** | **0.8303** | 11 (Early stopped) |

---

## 6. Hardware Profiling & VRAM Telemetry

Measured on NVIDIA GeForce RTX 2050 (4,096 MB VRAM):

| Arm Code | Training Time | Peak VRAM | Allocated VRAM | Reserved VRAM | Memory Headroom | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A** | $1,617.4\text{ s}$ ($27.0\text{ min}$) | $1,348.8\text{ MB}$ | $142.1\text{ MB}$ | $1,622.0\text{ MB}$ | $67.1\%$ Free | **PASSED** |
| **Arm B** | $1,328.6\text{ s}$ ($22.1\text{ min}$) | $1,684.2\text{ MB}$ | $145.3\text{ MB}$ | $1,894.0\text{ MB}$ | $58.9\%$ Free | **PASSED** |
| **Arm C** | $2,184.1\text{ s}$ ($36.4\text{ min}$) | $1,970.5\text{ MB}$ | $148.2\text{ MB}$ | $2,180.0\text{ MB}$ | $51.9\%$ Free | **PASSED** |
| **Arm D** | $1,215.3\text{ s}$ ($20.3\text{ min}$) | $2,014.1\text{ MB}$ | $151.7\text{ MB}$ | $2,250.0\text{ MB}$ | $50.8\%$ Free | **PASSED** |
| **Arm E** | $2,341.8\text{ s}$ ($39.0\text{ min}$) | $2,128.6\text{ MB}$ | $158.4\text{ MB}$ | $2,390.0\text{ MB}$ | $48.0\%$ Free | **PASSED** |
| **Arm F** | $3,458.2\text{ s}$ ($57.6\text{ min}$) | $2,284.0\text{ MB}$ | $164.9\text{ MB}$ | $2,540.0\text{ MB}$ | $44.2\%$ Free | **PASSED** |
| **Arm G** | $3,212.8\text{ s}$ ($53.5\text{ min}$) | $2,396.4\text{ MB}$ | $171.2\text{ MB}$ | $2,680.0\text{ MB}$ | $41.5\%$ Free | **PASSED** |

---

## 7. Checkpoint Inventory & Integrity Hash Verification

All models are saved under `models/v9/phase2/` with verified MD5 checksums:

| Arm Name | Filename | File Size | Best Epoch | MD5 Checksum |
| :--- | :--- | :---: | :---: | :--- |
| **Arm A** | `v9_phase2_arm_a_best.pth` | $344.0\text{ MB}$ | 17 | `f8456f8583b8da3181e942199b662553` |
| **Arm B** | `v9_phase2_arm_b_best.pth` | $344.0\text{ MB}$ | 4 | `6f3d669ffe61941b8c58f1796e64b45a` |
| **Arm C** | `v9_phase2_arm_c_best.pth` | $344.0\text{ MB}$ | 12 | `ac328a4266cd70d09cc791920d3b86ba` |
| **Arm D** | `v9_phase2_arm_d_best.pth` | $344.8\text{ MB}$ | 3 | `eb345d7e04ceb41df003e19620ef52d4` |
| **Arm E** | `v9_phase2_arm_e_best.pth` | $347.2\text{ MB}$ | 2 | `e522b9112def4173f62e86762601b943` |
| **Arm F** | `v9_phase2_arm_f_best.pth` | $349.5\text{ MB}$ | 5 | `e5412e79047bc23eab2a6a8c34f65853` |
| **Arm G** | `v9_phase2_arm_g_best.pth` | $351.9\text{ MB}$ | 3 | `c60b1aab0a1bc809c9ae27a57ddc65ec` |

---

## 8. Governance Declarations

```
=====================================================================================
                    V9 PHASE 2 TRAINING GOVERNANCE DECLARATION
=====================================================================================
1. Test Access Count:                    0 (ZERO READS VERIFIED)
2. Test Cache Status:                    ABSENT (VERIFIED)
3. Patient Leakage:                      0 (VERIFIED DISJOINT PATIENTS)
4. Frozen V6 U-KAN MD5:                  68cfc38cfbddd93d6e4a6e692fffbb34 [VERIFIED]
5. Checkpoint Protection:                NO OVERWRITES OF HISTORICAL MODELS
6. Deterministic Processing:             SEED = 42 (100% REPRODUCIBLE)
7. Training Mode:                        STRICTLY TRAIN / VALIDATION ONLY
=====================================================================================
```
