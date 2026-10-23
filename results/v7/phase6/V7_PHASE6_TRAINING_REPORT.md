# V7 Phase 6: Machine-ROI-Aware Classifier Training Report

## 1. Governance & Training Environment
- **Scope:** 100% Validation and Training only.
- **Test Quarantine:** Strictly Enforced (0 test reads; `test.csv` and test data completely unaccessed).
- **Split Governance:** Patient-level split ($N=2,501$ train / 1,096 patients; $N=541$ validation / 235 patients; 0 patient overlap).
- **Hardware:** NVIDIA GeForce RTX 2050 (4.00 GB VRAM), PyTorch AMP FP16, Sequential Chunked ViT Engine.
- **Model Selection Rule:** Best epoch selected strictly via **Validation Macro-F1**.

---

## 2. Controlled Classification Arms Summary (Validation Split)

| Arm | Architecture / Unfrozen Depth | Best Epoch | Validation Accuracy | Validation Macro-F1 | Validation Binary F1 | Validation ROC-AUC | Validation PR-AUC | Malignant Recall | Specificity | Checkpoint MD5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Arm A** | V6 Arm-C Benchmark Reference (Single ROI) | Reference | 71.90% | 71.05% | 64.59% | 0.7787 | 0.7225 | 60.14% | 79.34% | `508250cfc1c07af1ea7d81cc2e5a7418` |
| **Arm B** | Multi-Proposal ($K=3$) + Frozen ViT | Epoch 13 | 71.16% | 70.69% | 66.95% | 0.7671 | 0.6807 | 74.88% | 68.79% | `9ad187a2040e28d2f2e9dbf186735781` |
| **Arm C** | Multi-Proposal ($K=3$) + Last 2 ViT Blocks | Epoch 11 | **78.56%** | **77.13%** | **71.43%** | 0.8248 | 0.7576 | 68.72% | **84.85%** | `955531a8ce548cc710b247b5efba49b8` |
| **Arm D** | Multi-Proposal ($K=3$) + Last 4 ViT Blocks | Epoch 5 | 75.42% | 74.75% | 70.64% | **0.8423** | **0.7781** | **75.83%** | 75.15% | `a04739bfc063c01e87ddaaf6c3684f57` |
| **Arm E2** | Multi-Proposal ($K=5$) + Last 2 ViT Blocks | Epoch 1 | 74.68% | 73.74% | 68.79% | 0.8186 | 0.7170 | 71.56% | 76.67% | `fc26c3ca602b2626d9cec5536cdb917e` |

---

## 3. Training Dynamics & Epoch Selection
- **Arm D (Last 4 Blocks):**
  - Best Validation Macro-F1 (**74.75%**, Accuracy **75.42%**, ROC-AUC **0.8423**) achieved at **Epoch 5**.
  - Early stopping triggered at Epoch 13 after 8 consecutive epochs without higher Macro-F1.
- **Arm E2 ($K=5$, Last 2 Blocks):**
  - Best Validation Macro-F1 (**73.74%**, Accuracy **74.68%**, ROC-AUC **0.8186**) achieved at **Epoch 1**.
  - *Note on Selection Rule:* Although Epoch 3 achieved a higher raw ROC-AUC (0.8518), Epoch 1 was deterministically selected because the pre-specified primary selection metric was **Validation Macro-F1** (73.74% vs 68.23%).
  - Early stopping triggered at Epoch 9 after 8 consecutive epochs without exceeding Epoch 1 Macro-F1.

---

## 4. Protected Checkpoint Integrity Verification
All 7 model checkpoints are preserved and cryptographically verified:
1. `models/v6/segmentation/final/ukan_v6_arm_g_best.pth` -> `68cfc38cfbddd93d6e4a6e692fffbb34` (MATCH)
2. `models/v6/classification/vit_v6_arm_c_best.pth` -> `508250cfc1c07af1ea7d81cc2e5a7418` (MATCH)
3. `models/v5/classification/vit_v5_arm_i_best.pth` -> `ec80689a43a906f89a15d7148cf7b05e` (MATCH)
4. `models/v7/phase6/v7_vit_arm_b_frozen_best.pth` -> `9ad187a2040e28d2f2e9dbf186735781` (MATCH)
5. `models/v7/phase6/v7_vit_arm_c_last2_best.pth` -> `955531a8ce548cc710b247b5efba49b8` (MATCH)
6. `models/v7/phase6/v7_vit_arm_d_last4_best.pth` -> `a04739bfc063c01e87ddaaf6c3684f57` (MATCH)
7. `models/v7/phase6/v7_vit_arm_e2_k5_best.pth` -> `fc26c3ca602b2626d9cec5536cdb917e` (MATCH)
