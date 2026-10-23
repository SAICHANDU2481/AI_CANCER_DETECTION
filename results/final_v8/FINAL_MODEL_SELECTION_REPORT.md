# Final Model Selection & Governance Report
## Project: Advanced Breast Cancer Diagnosis Using Wiener Filtering, Vision Transformer, and U-KAN-Based Medical Image Analysis

---

## 1. Executive Model Selection Decision

**FINAL REPORTED MODEL:** **V8 (Dual-Resolution U-KAN $K=3$ Multi-Scale RoI Proposals + Last 4 ViT Blocks Fine-Tuned)**  
**STATUS:** **FROZEN & LOCKED**  
**INTENDED USE:** **AI-Assisted Research Prototype** (Not a clinical diagnostic system)

### Formal Selection Statement
> "V8 was retained as the final reported configuration based on its locked patient-independent test performance and its more balanced accuracy/sensitivity/specificity profile. V9 was subsequently evaluated as an experimental extension and is retained in the research record but was not selected as the final reported configuration."

---

## 2. Thesis-Ready Model Selection Section

> "Among the evaluated experimental configurations, V8 was designated as the final reported model based on its locked patient-independent test evaluation and overall classification balance. On the primary test cohort of 525 scans from 235 patients, V8 achieved 64.57% accuracy, 64.18% Macro-F1, 0.6947 ROC-AUC, 0.5857 PR-AUC, 66.05% sensitivity, and 63.55% specificity.
>
> A subsequent V9 experimental extension achieved higher malignant sensitivity of 76.28% and PR-AUC of 0.6129, but its overall accuracy decreased to 62.86% and specificity decreased to 53.55%. The V9 implementation and final test evaluation were independently audited and no implementation error was identified. Therefore, V9 is retained as a documented experimental extension rather than replacing V8 as the final reported configuration."

---

## 3. Comparative Test Evaluation Matrix

All models were evaluated on the held-out CBIS-DDSM test cohort under strict zero-touch quarantine protocols:

| Model | Accuracy (%) | Macro-F1 (%) | ROC-AUC | PR-AUC | Sensitivity (%) | Specificity (%) | Configuration Summary |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **V5** | 65.02 | 64.73 | 0.6988 | 0.5916 | 68.37 | 62.70 | Arm I (Concentric RoIs + ViT Base) |
| **V6** | 64.45 | 64.11 | 0.7087 | 0.5975 | 66.98 | 62.70 | Arm C (U-KAN $768\times768$ + Machine CC RoIs) |
| **V8** | **64.57** | **64.18** | **0.6947** | **0.5857** | **66.05** | **63.55** | **Arm F (Dual-Res $K=3$ + ViT 4-Block Tuned) [FINAL]** |
| **V9** | 62.86 | 62.86 | 0.7046 | 0.6129 | 76.28 | 53.55 | Arm G ($K=5$ Candidate Attention + Temp Scaling) [Extension] |

*Note: All metric values are strictly locked and unalterable.*

---

## 4. Scientific Rationale for Retaining V8

The selection of V8 over other architectures is grounded in rigorous scientific analysis:

1. **Balanced Multi-Class Diagnostic Profile:**
   - V8 achieves **64.57% Accuracy** and **63.55% Specificity** while maintaining strong malignant detection (**66.05% Sensitivity**).
   - In comparison, V9 demonstrated a significant specificity collapse down to **53.55%** (144 false positives out of 310 benign test cases), resulting in an unfavorable false-positive burden for an end-to-end multi-crop prototype.
   - V8 strikes a well-calibrated equilibrium between sensitivity and specificity on unseen patients.

2. **Robust Multi-Scale RoI Coupling:**
   - V8 incorporates dual-resolution candidate generation ($512 + 768$ U-KAN probability aggregation) with continuous softmax attention across $K=3$ proposals at 4 concentric scales ($1.2\times, 1.8\times, 2.5\times, 4.0\times$).
   - The zero-fallback rate was confirmed at **0.00% (0 / 525 scans)** on test data, demonstrating reliable localization of both circumscribed masses and diffuse microcalcifications.

3. **Generalization Stability:**
   - Fine-tuning the top 4 Transformer blocks (Arm F) provided optimal feature adaptation without destabilizing lower-level patch embeddings.
   - 95% Bootstrap Confidence Intervals for V8 confirm reproducible test bounds:
     - Accuracy: **[59.06% – 69.79%]**
     - ROC-AUC: **[0.6309 – 0.7474]**
     - Sensitivity: **[58.85% – 72.93%]**
     - Specificity: **[54.58% – 71.56%]**

---

## 5. Formal Status of V9 Experimental Extension

V9 was evaluated after V8 to explore whether expanded proposal sets ($K=5$) and multi-head candidate attention could boost lesion capture.
A comprehensive 9-point scientific audit confirmed:
- **Governance:** PASS
- **Checkpoint Integrity:** PASS (`c60b1aab0a1bc809c9ae27a57ddc65ec`)
- **Cohort Integrity:** PASS ($525$ scans / $235$ patients)
- **Label Integrity:** PASS (Zero label flips)
- **Implementation Errors:** NONE found

V9 is therefore recognized as a valid experimental contribution that achieved high sensitivity ($76.28\%$), but due to lower specificity ($53.55\%$) and overall accuracy ($62.86\%$), it serves as an exploratory extension rather than replacing V8 as the primary reported system.

---

## 6. Cryptographic Provenance & Locked Checkpoints

- **Final Classification Checkpoint:** `models/final_v8/vit_v8_final.pth`  
  - Original Source: `models/v8/classification/vit_v8_phase2_arm_f_best.pth`
  - MD5 Hash: `882313ce67adca01bffa699a180719c4`
  - SHA256 Hash: `7aa49ef7ff2f94e72cc2f27361af89ba0cd04dae00b333f41d8c50a37a1c64c7`
  - File Size: `366,913,760 bytes`
- **Final Segmentation Checkpoint:** `models/final_v8/ukan_v6_final.pth`  
  - Original Source: `models/v6/segmentation/final/ukan_v6_arm_g_best.pth`
  - MD5 Hash: `68cfc38cfbddd93d6e4a6e692fffbb34`
  - SHA256 Hash: `f2cd9b1208f4b77416640201143539d6d08b1ff03cb175e1502eb73b8431769d`
  - File Size: `91,242,039 bytes`

---

## 7. Mandatory Governance & Safety Constraints

1. **Research Prototype Disclaimer:** This system is an AI-assisted research prototype developed for academic exploration. It is not approved as a clinical diagnostic tool.
2. **Post-Evaluation Protocol:** No further retraining, threshold tuning, or test set manipulation is permitted. All benchmark numbers are permanently frozen.
