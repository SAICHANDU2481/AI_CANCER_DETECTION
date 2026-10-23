# V9 PHASE 0 — PRE-FLIGHT GOVERNANCE, CHECKPOINT AUDIT & EXPERIMENT MANIFEST REPORT

---

## 1. Executive Summary

This report establishes the verified scientific baseline and pre-flight governance audit for **V9: Targeted Localization + Multi-Candidate Fusion + Generalization + Calibration**. 

The V9 development lifecycle is formulated specifically to address the six demonstrated failure modes of the V8 locked test evaluation:
1. **Weak lesion localization and proposal coverage:** 40.19% of test scans had IoU < 0.10.
2. **Over-reliance on top-ranked single ROI:** Downstream classifier vulnerability when candidate 1 misses the true lesion margin.
3. **Validation-to-test generalization gap:** Macro-F1 fell from 75.36% on validation to 64.18% on locked test (11.18 percentage points gap).
4. **Calcification classification difficulty:** Calcification sensitivity remained at 55.81%.
5. **Classifier failure despite good localization:** 34.25% of malignant false negatives occurred despite IoU >= 0.30.
6. **Probability miscalibration:** Expected Calibration Error (ECE) reached 29.87% with strong overconfidence.

All historical checkpoints (V6 U-KAN, V6 ViT, V8 ViT, V5 ViT) have been cryptographically audited with exact hash matches. The held-out test split is strictly quarantined with zero access.

---

## 2. Checkpoint Integrity Audit

| Checkpoint | Role | Path | Expected MD5 | Actual MD5 | Integrity Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V6 U-KAN (Arm G Best)** | Frozen Lesion Segmentor | `models/v6/segmentation/final/ukan_v6_arm_g_best.pth` | `68cfc38cfbddd93d6e4a6e692fffbb34` | `68cfc38cfbddd93d6e4a6e692fffbb34` | **PASS (EXACT MATCH)** |
| **V6 ViT (Arm C Best)** | Reference Classifier | `models/v6/classification/vit_v6_arm_c_best.pth` | `508250cfc1c07af1ea7d81cc2e5a7418` | `508250cfc1c07af1ea7d81cc2e5a7418` | **PASS (EXACT MATCH)** |
| **V8 ViT (Arm F Best)** | Reference Classifier | `models/v8/classification/vit_v8_phase2_arm_f_best.pth` | `882313ce67adca01bffa699a180719c4` | `882313ce67adca01bffa699a180719c4` | **PASS (EXACT MATCH)** |
| **V5 ViT (Arm I Best)** | Historical Baseline | `models/v5/classification/vit_v5_arm_i_best.pth` | `ec80689a43a906f89a15d7148cf7b05e` | `ec80689a43a906f89a15d7148cf7b05e` | **PASS (EXACT MATCH)** |

---

## 3. Dataset Splits & Strict Patient-Level Isolation Audit

| Split Name | Scan Count ($N_{\text{scans}}$) | Patient Count ($N_{\text{pts}}$) | Benign Scans | Malignant Scans | Masses | Calcifications |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Set** | 2,501 | 1,096 | 1,470 | 1,031 | 1,234 | 1,267 |
| **Validation Set** | 541 | 235 | 330 | 211 | 244 | 297 |
| **Primary Test Set** | 525 | 235 | 310 | 215 | 267 | 258 |
| **Comprehensive Test Set** | 526 | 235 | 311 | 215 | 267 | 259 |

### Disjointness Verification:
- **Train $\cap$ Validation Overlap:** `0` patients
- **Train $\cap$ Test Overlap:** `0` patients
- **Validation $\cap$ Test Overlap:** `0` patients
- **Audit Outcome:** **PASS (100% Strict Patient Isolation)**

---

## 4. Hardware & Runtime Configuration

- **GPU Device:** NVIDIA GeForce RTX 2050 (4095.5 MB Dedicated VRAM)
- **CUDA Stack:** PyTorch 2.6.0+cu124, CUDA 12.4
- **Host RAM:** 5984.6 MB available / 16108.1 MB total
- **Host Pagefile:** 5877.1 MB available / 25883.0 MB total
- **Random Seed:** `42` (Fixed across Python random, NumPy, PyTorch CPU/CUDA)

---

## 5. V8 Locked Baseline Reference (Primary Test Benchmark, $N=525$)

| Metric | V8 Arm F Locked Test Result | Primary Observation |
| :--- | :--- | :--- |
| **Accuracy** | **64.57%** (339/525) | Base classification accuracy |
| **Macro-F1** | **64.18%** | Balanced harmonic mean |
| **Binary F1** | **60.43%** | Malignant class F1 |
| **ROC-AUC** | **0.6947** | Diagnostic ranking discrimination |
| **PR-AUC** | **0.5857** | Precision-Recall area (+17.62% over 40.95% prevalence) |
| **Sensitivity (Recall)** | **66.05%** (142/215) | True positive detection rate |
| **Specificity** | **63.55%** (197/310) | Benign identification rate |
| **Precision** | **55.69%** (142/255) | Positive predictive value |
| **Brier Score** | **0.3143** | Mean squared probability error |
| **ECE** | **29.87%** | Expected Calibration Error |
| **Confusion Matrix** | **TN: 197 \| FP: 113 \| FN: 73 \| TP: 142** | Error distribution |
| **$\text{IoU} < 0.10$ Cohort** | $N=211$ (40.19%) | Sensitivity: 50.79%, Specificity: 68.24% |
| **$\text{IoU} \ge 0.30$ Cohort** | $N=231$ (44.00%) | Sensitivity: 77.27%, Specificity: 60.33% |

---

## 6. V9 Phase Architecture & Sequential Execution Plan

| Phase | Title | Core Objective | Key Deliverables & Validation Criteria |
| :--- | :--- | :--- | :--- |
| **Phase 0** | Pre-Flight Governance & Checkpoint Audit | Audit all baseline models, splits, directories, and hardware. | `V9_PHASE0_AUDIT_REPORT.md`, governance audit JSON, experiment manifest. |
| **Phase 1** | Multi-Source Proposal Generation | Improve candidate lesion coverage before classifier changes using label-free ranking (contrast, area, eccentricity, boundary distance, multi-scale agreement). | Primary: Recall@5 (IoU >= 0.30) on Validation. Secondary: Containment, small lesion & calcification coverage. |
| **Phase 2** | Multi-Candidate ViT Fusion | Shared ViT backbone with candidate attention / gated pooling ($K=1, 3, 5$), full mammogram feature, and calcification patch pathway (Arms A–G). | Primary: Validation Macro-F1. Secondary: ROC-AUC, PR-AUC, Sensitivity. |
| **Phase 3** | Controlled Generalization & Regularization | Bridge train-validation generalization gap using controlled augmentations, hard-example weighting, and representation consistency loss. | Primary: Reduce train-val generalization gap while maintaining high Validation Macro-F1. |
| **Phase 4** | Validation-Only Probability Calibration | Fit temperature scaling ($T$) on validation data only; freeze $T$ before test. | Primary: Minimize Validation Brier score and ECE without altering ranking discrimination. |
| **Phase 5** | Single Locked Final Test Evaluation | Execute one locked test evaluation on primary 525-record test cohort with frozen parameters and fixed threshold 0.50. | Final test metrics, bootstrap CIs, subgroup analysis, error taxonomy, historical locked comparison. |

---

## 7. Governance Protocols & Test Quarantine Rules

1. **Test Quarantine:** The held-out test split is strictly locked and quarantined during Phases 0–4. No test images, masks, metadata, or labels shall be read.
2. **Deterministic Validation:** Validation evaluations are performed deterministically with `torch.inference_mode()` and fixed seed `42`.
3. **No Synthetic Augmentation on Validation/Test:** Augmentations are applied strictly to the training split.
4. **Validation-Only Model Selection:** Checkpoint and architecture selection are guided strictly by Validation Macro-F1.

---

**V9 Phase 0 Status: PASSED & COMPLETED**
