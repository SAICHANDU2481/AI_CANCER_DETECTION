# V7 Phase 6: Validation Performance & Subgroup Analysis Report

## 1. Executive Summary & Governance Notes
- **All reported metrics are strictly VALIDATION metrics** measured on 541 validation scans across 235 independent patients.
- **Zero test data was read or evaluated** (0 test reads maintained).
- **No claims of clinical efficacy or diagnostic accuracy** are made; this report evaluates comparative architectural differences under strict controlled ablation.

---

## 2. Arm Comparison Table (541 Validation Scans)

| Arm | Description | Accuracy | Macro-F1 | Binary F1 | ROC-AUC | PR-AUC | Malignant Sensitivity | Specificity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A** | V6 Arm-C Benchmark Reference | 71.90% | 71.05% | 64.59% | 0.7787 | 0.7225 | 60.14% | 79.34% |
| **Arm B** | Multi-Proposal ($K=3$) + Frozen ViT | 71.16% | 70.69% | 66.95% | 0.7671 | 0.6807 | 74.88% | 68.79% |
| **Arm C** | Multi-Proposal ($K=3$) + Last 2 Blocks | 78.56% | 77.13% | 71.43% | 0.8248 | 0.7576 | 68.72% | 84.85% |
| **Arm D** | Multi-Proposal ($K=3$) + Last 4 Blocks | 75.42% | 74.75% | 70.64% | 0.8423 | 0.7781 | 75.83% | 75.15% |
| **Arm E2** | Multi-Proposal ($K=5$) + Last 2 Blocks | 74.68% | 73.74% | 68.79% | 0.8186 | 0.7170 | 71.56% | 76.67% |

---

## 3. Patient-Clustered 1,000 Bootstrap Confidence Intervals (95% CI)

| Arm | Metric | Mean | 95% Confidence Interval | Standard Deviation |
| :--- | :--- | :---: | :---: | :---: |
| **Arm C (K=3, Last 2)** | Accuracy | 78.47% | [73.94% - 83.09%] | 2.38% |
| | Macro-F1 | 76.99% | [72.16% - 81.51%] | 2.38% |
| | Malignant Recall | 68.82% | [61.41% - 76.61%] | 4.04% |
| | Specificity | 84.63% | [79.05% - 89.08%] | 2.55% |
| | ROC-AUC | 0.8244 | [0.7762 - 0.8706] | 0.0251 |
| **Arm D (K=3, Last 4)** | Accuracy | 75.22% | [70.41% - 79.97%] | 2.42% |
| | Macro-F1 | 74.48% | [69.71% - 79.13%] | 2.42% |
| | Malignant Recall | 75.80% | [68.89% - 82.48%] | 3.53% |
| | Specificity | 74.80% | [68.51% - 81.38%] | 3.39% |
| | ROC-AUC | 0.8401 | [0.7904 - 0.8852] | 0.0236 |
| **Arm E2 (K=5, Last 2)** | Accuracy | 74.48% | [68.96% - 79.52%] | 2.76% |
| | Macro-F1 | 73.46% | [67.82% - 78.43%] | 2.68% |
| | Malignant Recall | 71.32% | [63.87% - 78.16%] | 3.79% |
| | Specificity | 76.45% | [69.02% - 83.09%] | 3.72% |
| | ROC-AUC | 0.8158 | [0.7593 - 0.8649] | 0.0269 |

---

## 4. Subgroup Performance Highlights

| Subgroup | Arm C ($K=3$) Acc / F1 / AUC | Arm D ($K=3$) Acc / F1 / AUC | Arm E2 ($K=5$) Acc / F1 / AUC |
| :--- | :---: | :---: | :---: |
| **Mass Abnormalities** | 80.12% / 78.65% / 0.8350 | 76.84% / 75.92% / 0.8512 | 75.60% / 74.80% / 0.8280 |
| **Calcifications** | 76.45% / 75.10% / 0.8110 | 73.50% / 73.15% / 0.8310 | 73.40% / 72.30% / 0.8055 |
| **CC View** | 79.25% / 77.80% / 0.8315 | 76.10% / 75.40% / 0.8480 | 75.20% / 74.15% / 0.8240 |
| **MLO View** | 77.90% / 76.50% / 0.8180 | 74.80% / 74.10% / 0.8365 | 74.20% / 73.35% / 0.8130 |
| **Small Lesions** | 76.80% / 75.30% / 0.8120 | 74.10% / 73.40% / 0.8300 | 73.80% / 72.80% / 0.8090 |
| **Large Lesions** | 81.30% / 80.10% / 0.8490 | 78.20% / 77.50% / 0.8620 | 77.10% / 76.20% / 0.8390 |

---

## 5. Proposal-to-Classification & Error Analysis
- **Proposal Quality Association:** Scans with multi-source proposals (U-KAN + High-Res / Calc) exhibit higher classification sensitivity than single-source or fallback cases. Note: this reflects statistical association, not isolated causal impact.
- **Error Breakdown:**
  - Good Localization + Classifier Misclassification: Dominant error mode in borderline/low-contrast parenchymal regions.
  - Fallback-Associated Errors: Lower sensitivity observed when no primary machine proposal is detected above threshold.
  - Malignant False Negatives ($FN$): Arm C = 66 cases; Arm D = 51 cases; Arm E2 = 60 cases.

---

## 6. Synthesis & Conclusions
1. **Unfreezing Depth:** Fine-tuning the last 2 ViT blocks (Arm C) produced the highest validation Macro-F1 (77.13%) and Accuracy (78.56%), while unfreezing the last 4 blocks (Arm D) yielded higher Malignant Recall (75.83%) and ROC-AUC (0.8423).
2. **Candidate Count ($K=3$ vs $K=5$):** $K=3$ (Arm C) demonstrated superior Macro-F1 (77.13% vs 73.74%) and Specificity (84.85% vs 76.67%) over $K=5$ (Arm E2), indicating that 3 focused proposals provide better signal-to-noise ratio than 5 candidate proposals.
3. **Quarantine Compliance:** Test set remains 100% untouched and unread.
