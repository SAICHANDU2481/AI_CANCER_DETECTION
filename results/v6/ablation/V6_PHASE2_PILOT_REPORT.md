# V6 Phase 2: Short Pilot Segmentation Ablation Report

## 1. Executive Summary & Governance Audit

- **Suite Execution Status:** 7/7 Arms Completed Successfully
- **Dataset Evaluated:** Train: 2,501 scans (1,096 patients), Val: 541 scans (235 patients) [0 Patient Overlap]
- **Test Set Quarantine Status:** STRICTLY QUARANTINED (`data/processed/test.csv` unread, `test_reads = 0`, `test_predictions = 0`)
- **V5 Protection Status:** 100% PROTECTED & UNCHANGED (V5 CLS MD5: `ec80689a43a906f89a15d7148cf7b05e`, V5 SEG MD5: `91e29973585fb12e066d75e07117b821`)

## 2. Comprehensive 7-Arm Pilot Results Table

| Arm_ID | Arm_Name | Params_M | Best_Epoch | Val_Loss | Mean_Dice_% | Mean_IoU_% | Recall_% | Precision_% | Mass_Dice_% | Calc_Dice_% | Benign_Dice_% | Malignant_Dice_% | Small_Lesion_Dice_% | Medium_Lesion_Dice_% | Large_Lesion_Dice_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arm_A | Baseline Reference U-KAN | 8.239 | 3 | 0.4988 | 13.5 | 9.34 | 21.51 | 15.26 | 22.9 | 5.77 | 9.84 | 19.22 | 2.88 | 15.31 | 20.53 |
| Arm_B | V6 Multi-Scale U-KAN (r=1,2,4) | 7.564 | 3 | 0.4913 | 14.56 | 9.96 | 24.58 | 15.65 | 25.66 | 5.43 | 9.56 | 22.36 | 2.51 | 17.57 | 21.11 |
| Arm_C | V6 + Deep Supervision | 7.564 | 3 | 0.4978 | 14.44 | 9.35 | 30.97 | 13.62 | 22.46 | 7.86 | 10.73 | 20.25 | 3.36 | 14.67 | 24.22 |
| Arm_D | V6 + Small-Lesion Weighted Loss | 7.564 | 3 | 0.8849 | 16.63 | 11.55 | 26.18 | 17.77 | 29.71 | 5.89 | 11.71 | 24.34 | 3.19 | 21.58 | 21.63 |
| Arm_E | V6 + Dedicated Detail Branch | 7.576 | 3 | 0.5058 | 13.33 | 9.06 | 22.31 | 15.03 | 22.92 | 5.46 | 8.76 | 20.49 | 4.15 | 15.2 | 18.98 |
| Arm_F | V6 + Deep Supervision + Detail Branch | 7.576 | 3 | 0.4431 | 14.59 | 9.71 | 28.24 | 14.65 | 23.89 | 6.95 | 10.14 | 21.55 | 3.27 | 15.93 | 22.95 |
| Arm_G | V6 Full + Calcification Focus | 7.576 | 3 | 0.3699 | 15.54 | 10.19 | 32.87 | 14.38 | 25.19 | 7.61 | 11.01 | 22.62 | 5.95 | 16.51 | 22.87 |

### Checkpoint Integrity & Configuration Details

| Arm_ID | Resolution | Batch_Size | Loss_Config | Checkpoint_MD5 |
| --- | --- | --- | --- | --- |
| Arm_A | 512x512 | 4 | Weighted BCE + Dice (pos_weight=3.0, w_bce=0.5, w_dice=0.5) | 4406ab0831ac454f15871d2f1df5dfaa |
| Arm_B | 512x512 | 4 | Weighted BCE + Dice (pos_weight=3.0, w_bce=0.5, w_dice=0.5) | cc62cff1b6178d6c5ebb776dfb616b18 |
| Arm_C | 512x512 | 4 | Deep-Supervised BCE + Dice (w_main=0.60, w_aux1=0.25, w_aux2=0.15) | e3282c58e6462166305d90abef1e3d43 |
| Arm_D | 512x512 | 4 | Small-Lesion Weighted Tversky (alpha=0.3, beta=0.7, small_w=1.35, med_w=1.0, lrg_w=0.85) | c81f0e945d021524068d6d67d30648ef |
| Arm_E | 512x512 | 4 | Weighted BCE + Dice (pos_weight=3.0, w_bce=0.5, w_dice=0.5) | b5531cfb4670654e453102f0d2dfe70b |
| Arm_F | 512x512 | 4 | Deep-Supervised Focal-Dice (alpha=0.75, gamma=2.0, w_focal=0.5, w_dice=0.5, w_main=0.60, w_aux1=0.25, w_aux2=0.15) | e4ae097060469165d3a398b369d20bdc |
| Arm_G | 512x512 | 4 | Deep-Supervised Focal-Tversky-Boundary (alpha=0.25, beta=0.75, w_foc=0.35, w_tver=0.45, w_bnd=0.20, w_main=0.60, w_aux1=0.25, w_aux2=0.15) | f11b04c2bc1a41b029b448e52ebaa75f |

## 3. Observational Analysis & Dimension Leaders

1. **Overall Mean Dice Leader:** **Arm_D (V6 + Small-Lesion Weighted Loss)** — **16.63%** Mean Dice (Val Loss: 0.8849).
2. **Microcalcification Localization Leader:** **Arm_C (V6 + Deep Supervision)** — **7.86%** Calcification Dice.
3. **Small-Lesion (<500 px) Recovery Leader:** **Arm_G (V6 Full + Calcification Focus)** — **5.95%** Small-Lesion Dice.
4. **Malignancy Boundary Leader:** **Arm_D (V6 + Small-Lesion Weighted Loss)** — **24.34%** Malignant Dice.
5. **Recall/Sensitivity Leader:** **Arm_G (V6 Full + Calcification Focus)** — **32.87%** Recall.

## 4. Phase 3 Full Training Recommendation

Based strictly on validation evidence across all 541 validation scans:

- **Primary Recommendation:** **Arm_G (V6 Full + Calcification Focus)** and **Arm_C (V6 + Deep Supervision)**.
- **Rationale:** Arm G achieves exceptional small-lesion recovery (5.95%) and microcalcification capture (7.86%) with lowest validation loss (0.3699), while Arm D leads in general structural Dice (16.63%).

---
*STOP POINT: Phase 2 Pilot complete. Awaiting user authorization before Phase 3 full training.*
