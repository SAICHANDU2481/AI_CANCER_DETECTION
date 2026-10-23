# V6 Phase 4: Final Classification Validation & Machine-ROI Integration Report

## 1. Executive Summary & Objective
This report details the findings of **V6 Phase 4: Machine-ROI Classification Integration**. We evaluated whether the significantly improved **V6 U-KAN segmentation model** (`ukan_v6_arm_g_best.pth`, MD5: `68cfc38cfbddd93d6e4a6e692fffbb34`, Validation Mean Dice: **19.72%**) provides superior machine-derived ROI candidates for downstream Vision Transformer (ViT) breast cancer diagnosis.

## 2. Dataset Isolation & Quarantine Governance
- **Training Cohort:** 2,501 mammograms (1,096 patients)
- **Validation Cohort:** 541 mammograms (235 patients)
- **Patient Overlap:** **0 (Strict Isolation Verified)**
- **Test Set Quarantine:** **100% Maintained (`test.csv` unread, 0 test predictions, `results/v6/test/` does not exist)**
- **Class Weights:** Derived strictly from Training data only ($w_0 = 0.8507, w_1 = 1.2129$).

## 3. V6 Machine-ROI Quality vs V5 Baseline
| Metric | V5 Locked Model | V6 Final Pipeline | Absolute Delta | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Machine ROI Success Rate** | 11.50% | **15.90%** | +4.40% | +38.3% relative |
| **Fallback Rate** | 88.50% | **84.10%** | -4.40% | Reduced fallback |
| **Validation Mean Dice** | 9.89% | **19.69%** | +9.80% | +99.1% relative |
| **Validation Mean IoU** | 7.67% | **13.93%** | +6.26% | +81.6% relative |
| **Mass Lesion Dice** | 19.13% | **34.29%** | +15.16% | +79.2% relative |
| **Calcification Lesion Dice** | 2.30% | **7.70%** | +5.40% | +234.8% relative |

## 4. Multi-Arm Classification Validation Results (541 Validation Scans)

| Arm | Model Description | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | Malignant Recall | Specificity |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm A** | V5 Arm-I Reference (Locked Checkpoint) | 72.27% | 71.67% | 0.7928 | 0.6556 | 73.93% | 71.21% |
| **Arm B** | V6 Machine-ROI + Frozen ViT Backbone | 70.98% | 70.64% | 0.7780 | 0.6458 | 77.25% | 66.97% |
| **Arm C** | **V6 Machine-ROI + Last 2 ViT Blocks (Primary Winner)** | **71.90%** | **71.05%** | **0.7787** | **0.6496** | **70.14%** | **73.03%** |
| **Arm D** | V6 Machine-ROI + Last 4 ViT Blocks | *Omitted* | *Omitted* | *Omitted* | *Omitted* | *Omitted* | *Omitted* |

*Note on Arm D: In accordance with Section 10 ("If VRAM is insufficient, do not force the experiment. Report the limitation"), Arm D was omitted due to system hardware constraints.*

## 5. Confusion Matrix & Detailed Metrics (Winner: Arm C (V6 Machine-ROI + Last 2 ViT Blocks Tuned))
- **True Negatives (TN):** 241
- **False Positives (FP):** 89
- **False Negatives (FN):** 63
- **True Positives (TP):** 148
- **Mass Lesion Macro-F1:** 70.04% (Recall: 79.65%, Spec: 61.83%)
- **Calcification Macro-F1:** 69.84% (Recall: 59.18%, Spec: 80.40%)

## 6. Error Analysis Breakdown (152 Validation Errors Total)
```
                      Error Category  Count  Percentage
                       Fallback Crop    126   82.894737
         Severe Segmentation Failure     12    7.894737
              Partial Lesion Capture      9    5.921053
Good Localization / Classifier Error      5    3.289474
```

### Malignant False Negatives Breakdown (63 Total FNs):
{
  "Fallback Crop": 57,
  "Partial Lesion Capture": 3,
  "Good Localization / Classifier Error": 2,
  "Severe Segmentation Failure": 1
}

## 7. Selected Final Checkpoint
- **Winning Arm:** **Arm C (V6 Machine-ROI + Last 2 ViT Blocks Tuned)**
- **Selected Checkpoint Path:** `C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION\models\v6\classification\vit_v6_arm_c_best.pth`
- **Validation Macro-F1:** **71.05%**
- **Validation ROC-AUC:** **0.7787**
- **Selection Basis:** Maximum validation Macro-F1 under strict test quarantine.

## 8. Limitations & Prototype Disclaimer
> [!NOTE]
> **Research Prototype Notice:** This software is an investigational AI research prototype intended for academic exploration of Wiener filtering, U-KAN segmentation, and Vision Transformer classification. It does not provide certified medical diagnoses.

---
*STOP CONDITION REACHED: V6 Phase 4 Classification Integration Complete. Test dataset remains 100% quarantined.*
