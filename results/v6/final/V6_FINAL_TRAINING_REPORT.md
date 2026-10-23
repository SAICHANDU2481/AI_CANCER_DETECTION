# V6 Final Training & Validation Evaluation Report

## 1. Objective
Evaluate the fully trained V6 U-KAN (Arm G configuration) for high-resolution lesion segmentation, with focus on small-lesion recall, microcalcification boundary localization, and malignancy differentiation.

## 2. Dataset & Patient Isolation
- **Train Cohort:** 2,501 scans (1096 patients)
- **Validation Cohort:** 541 scans (235 patients)
- **Patient Overlap:** **0 (Strict Isolation Verified)**
- **Test Set Quarantine:** **100% Maintained (`test.csv` unread, 0 test predictions, 0 test reads)**

## 3. Preprocessing Pipeline
Raw Mammogram → Wiener Filtering (noise suppression) → CLAHE (contrast enhancement) → Zero-Padding & Normalization → 512×512 Input Tensor.

## 4. V6 Model Architecture
- **Base Encoder:** Shallow feature preservation ($1 \to 32 \to 64 \to 128 \to 256$ channels)
- **Multi-Scale Dilated Blocks:** Dilations $r=1, 2, 4$ with Squeeze-and-Excitation (SE) Gated Channel Attention
- **Dedicated Detail Branch:** Direct high-frequency shallow residual streaming for microcalcification edge retention
- **Bottleneck:** Kolmogorov-Arnold Network (KANLinear) with learnable cubic B-spline basis functions
- **Total Parameter Count:** 7,576,395 parameters (7.58M)

## 5. Loss Function
Deep-Supervised Composite Focal-Tversky-Boundary Loss:
$$\mathcal{L}_{\text{total}} = 0.60 \mathcal{L}_{\text{main}} + 0.25 \mathcal{L}_{\text{aux1}} + 0.15 \mathcal{L}_{\text{aux2}}$$
$$\mathcal{L} = 0.35 \mathcal{L}_{\text{Focal}} + 0.45 \mathcal{L}_{\text{Tversky}}(\beta=0.75) + 0.20 \mathcal{L}_{\text{Boundary}}$$

## 6. Training Configuration & Hardware
- **Hardware:** NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)
- **Batch Size:** 4 | **AMP:** FP16
- **Optimizer:** AdamW (LR: $3\times 10^-4$, Weight Decay: $1\times 10^-4$)
- **Scheduler:** Cosine Annealing (T_max=40, eta_min=1e-6)
- **Total Epochs Trained:** 40 | **Best Epoch:** 31
- **Total Training Time:** 122388.3s (2039.80 min)

## 7. Best Validation Performance Summary

- **Validation Mean Dice:** **19.72%**
- **Validation Mean IoU:** **13.96%**
- **Sensitivity / Recall:** **33.37%**
- **Precision:** **19.54%**
- **Specificity:** **99.15%**
- **Pixel Accuracy:** **98.39%**

## 8. Subgroup Stratification

- **Mass Lesion Dice:** **34.28%**
- **Calcification Lesion Dice:** **7.76%**
- **Malignant Lesion Dice:** **28.31%**
- **Benign Lesion Dice:** **14.23%**
- **Small Lesions (<500 px) Dice:** **8.93%**
- **Medium Lesions (500–2,000 px) Dice:** **23.91%**
- **Large Lesions (>2,000 px) Dice:** **23.43%**

## 9. V5 vs V6 Paired Validation Comparison

| Metric | V5 Locked Model | V6 Final (Arm G) | Absolute Improvement | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Dice** | 9.89% | **19.72%** | +9.83% | +99.4% |
| **Mean IoU** | 7.67% | **13.96%** | +6.29% | +82.0% |
| **Recall (Sensitivity)** | 11.80% | **33.37%** | +21.57% | +182.8% |
| **Calcification Dice** | 2.30% | **7.76%** | +5.46% | +237.4% |
| **Small Lesion Dice (<500 px)** | 0.78% | **8.93%** | +8.15% | +1044.9% |
| **Malignant Lesion Dice** | 16.56% | **28.31%** | +11.75% | +71.0% |
| **Mass Lesion Dice** | 19.13% | **34.28%** | +15.15% | +79.2% |

### Paired Statistical Testing
- **Total Scans Compared:** 541 paired validation scans
- **Improved Scans:** **200** / 541 (37.0%)
- **Worsened Scans:** 135 / 541
- **Unchanged Scans:** 206 / 541
- **Wilcoxon Signed-Rank Test Statistic:** 69044.0 ($p = 6.4623e-07$)

## 10. Qualitative Analysis
High-resolution validation overlays generated under `results/v6/final/qualitative/`:
- `v6_val_malignant_mass.png`
- `v6_val_benign_mass.png`
- `v6_val_malignant_calc.png`
- `v6_val_benign_calc.png`
- `v6_val_small_lesion.png`
- `v6_val_large_lesion.png`

## 11. Limitations & Research Disclaimer
> [!NOTE]
> **Research Prototype Notice:** This system is an investigational AI-assisted medical image analysis research prototype. All metrics represent validation performance on the 541-scan validation cohort under strict test isolation. It does not claim clinical diagnostic certification.

## 12. Checkpoint Details
- **Best Checkpoint Path:** `C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION\models\v6\segmentation\final\ukan_v6_arm_g_best.pth`
- **Best Checkpoint MD5:** `68cfc38cfbddd93d6e4a6e692fffbb34`
- **Best Epoch:** 31

---
*STOP CONDITION REACHED: V6 Phase 3 Training and Validation Evaluation Complete. Test set remains strictly quarantined.*
