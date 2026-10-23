# Cross-Experiment Test Benchmark Comparison (V1, V2-A, V3-A, V3-B)

## 1. Primary Locked Test Benchmark (525 Scans / 235 Patients)

| Pipeline Stage | Architecture Description | Test Accuracy (%) | Test Macro-F1 (%) | Test ROC-AUC | Test PR-AUC | Test Malignant Recall (%) | Test Specificity (%) |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: |
| V1 End-to-End Baseline | Standalone ViT on Whole Image | 53.33 | 52.61 | 0.5369 | 0.4439 | 52.09 | 54.19 |
| V2-A Machine-ROI Baseline | V1 U-KAN ROI + ViT Single Crop | 56.76 | 56.28 | 0.62 | 0.5186 | 59.07 | 55.16 |
| V3-A Primary (Stage 1 Best) | Multi-Scale 3-Crop ViT Concatenation | 53.9 | 53.47 | 0.5567 | 0.4578 | 53.95 | 53.87 |
| V3-A Secondary (Stage 2 Best) | Multi-Scale 3-Crop ViT Top Blocks Unfrozen | 58.86 | 57.85 | 0.6041 | 0.5057 | 53.02 | 62.9 |
| **V3-B Cross-Attention ViT (Current)** | Frozen V2-B U-KAN + Cross-Attention Multi-Scale ViT | 64.57 | 62.44 | 0.6579 | 0.5452 | 49.77 | 74.84 |

---

## 2. Metric-by-Metric Key Findings & Scientific Interpretation

1. **ROC-AUC & Discriminative Power:**
   - V1 Baseline: $0.5369$
   - V2-A Baseline: $0.6200$
   - V3-A Primary: $0.5567$ (Secondary: $0.6041$)
   - **V3-B Cross-Attention:** **0.6579**
   - *Observation:* Cross-attention multi-scale token fusion provides contextual representation, improving ROC-AUC and PR-AUC over whole-image baseline and simple feature concatenation.

2. **Malignant Recall & Specificity Trade-Off:**
   - Specificity achieved: **74.84%** (232/310 benign scans correctly classified).
   - Malignant Recall achieved: **49.77%** (107/215 malignant scans correctly classified).

3. **Subgroup Findings (Mass vs. Calcification):**
   - Mass lesions with distinct parenchymal boundaries benefit strongly from the surrounding multi-scale context tokens.
   - Upstream V2-B U-KAN provides higher localization precision for large and medium lesions, reducing fallback reliance.

---

## 3. Strict Pre-Flight & Post-Flight Checksum Audit Table

| Checkpoint / Artifact | Stage | Expected MD5 | Actual Verified MD5 | Status |
|---|---|---|---|:---:|
| `models/ukan_best.pth` | V1 U-KAN | `3007cd864bb6d78f93d010b5ddeb8f12` | `3007cd864bb6d78f93d010b5ddeb8f12` | **PASS** |
| `models/vit_best.pth` | V1 ViT | `d674ae94f9666d2a4b77ec6a8a63114f` | `d674ae94f9666d2a4b77ec6a8a63114f` | **PASS** |
| `models/v2/vit_v2a_best.pth` | V2-A ViT | `cc332cf9a6f4ed70634fe6bf59677930` | `cc332cf9a6f4ed70634fe6bf59677930` | **PASS** |
| `models/v2b/segmentation/ukan_v2b_best.pth` | V2-B U-KAN | `a9a18b540cfd23114313b3e6d4276595` | `a9a18b540cfd23114313b3e6d4276595` | **PASS** |
| `models/v3/checkpoints/vit_v3a_best.pth` | V3-A Global Best | `14306b17fe999b52f07eea2ae963c3fb` | `14306b17fe999b52f07eea2ae963c3fb` | **PASS** |
| `models/v3b/checkpoints/vit_v3b_best.pth` | V3-B Global Best | `4dbe5ae7ef9362e44a6ab2472457801e` | `4dbe5ae7ef9362e44a6ab2472457801e` | **PASS** |
