# Arm I Checkpoint Audit

## Checkpoint
- **Path:** `models/v5/classification/vit_v5_arm_i_best.pth`
- **Exists:** `True`
- **Size:** `366,927,754 bytes` (349.93 MB)
- **MD5:** `ec80689a43a906f89a15d7148cf7b05e`
- **Load Status:** `SUCCESS` (PyTorch `torch.load` verified, complete state dict & metadata present)

## Architecture
- **Model:** `MultiScaleViTV5` (Vision Transformer Base Backbone + Multi-Scale RoI Head + Global Canvas + Microcalcification Patch)
- **Parameters:** `91,712,902` total parameters
- **Stage 1 Trainable Parameters:** `5,914,246` parameters
- **Fusion Dimension:** `4608`
- **K Proposals:** `3`
- **Scales:** `('local', 'medium', 'large', 'context')` (4 Concentric Scales: 1.2x, 1.8x, 2.5x, 4.0x)
- **Full Mammogram:** `True` (224x224 global whole-breast canvas)
- **Calcification Patch:** `True` (224x224 high-frequency microcalcification patch with deterministic `cand1_local` fallback)

## Weight Integrity
- **NaN Count:** `0`
- **Inf Count:** `0`
- **Min Parameter Value:** `-25.347422`
- **Max Parameter Value:** `18.775558`
- **Mean Parameter Magnitude:** `0.018473`

## Metadata Verification
- **Best Stage & Epoch:** `Stage 1 Epoch 04/08`
- **Validation Metrics:**
  - **Accuracy:** `72.27%` ($391 / 541$ correct)
  - **Macro-F1:** `71.67%`
  - **Binary-F1:** `67.53%`
  - **ROC-AUC:** `0.7928`
  - **PR-AUC:** `0.6540`
  - **Malignant Recall / Sensitivity:** `73.93%` ($156 / 211$ true malignant detected)
  - **Specificity:** `71.21%` ($235 / 330$ true benign ruled out)
  - **Precision:** `62.15%` ($156 / 251$ positive predictions confirmed malignant)
- **Confusion Matrix:** `TP=156, FP=95, TN=235, FN=55`

## Inference Smoke Test
- **Batch Shapes (Validation $B=32$):**
  - `full_image`: `[32, 3, 224, 224]`
  - `local`: `[32, 3, 3, 224, 224]`
  - `medium`: `[32, 3, 3, 224, 224]`
  - `large`: `[32, 3, 3, 224, 224]`
  - `context`: `[32, 3, 3, 224, 224]`
  - `calc_enhanced`: `[32, 3, 224, 224]`
  - `scores`: `[32, 3]`
- **Logits:** `[32, 2]`
- **Probabilities:** `[32, 2]`
- **Finite Status:** `True` (All logits and softmax probabilities finite; 0 NaN/Inf)

## A-H Integrity
All 8 historical ablation checkpoints verified on disk with 100% matching MD5 hashes:
- `vit_v5_arm_a_best.pth`: `86e6503624c0f26c3e284544f44cd13f` (`MATCH: True`)
- `vit_v5_arm_b_best.pth`: `e5037e7ed114fc54491f95d5fa276085` (`MATCH: True`)
- `vit_v5_arm_c_best.pth`: `c19c5c0d73b47af5e7379bd1d8fb911b` (`MATCH: True`)
- `vit_v5_arm_d_best.pth`: `ce783fa58f7695fdc40ac45f908a7913` (`MATCH: True`)
- `vit_v5_arm_e_best.pth`: `961076e875e6c40f5ec67984ba1d8689` (`MATCH: True`)
- `vit_v5_arm_f_best.pth`: `967a4ac99b69eaf2b5c1a87b5cb4ab7c` (`MATCH: True`)
- `vit_v5_arm_g_best.pth`: `6d84b056ca812160fcba73017a3121b0` (`MATCH: True`)
- `vit_v5_arm_h_best.pth`: `e5592bd3e32d062e9f2c4963eef0ffd7` (`MATCH: True`)

## Cache Integrity
- **Train Cache Count:** `2,501` NPZ files on `F:\BreastCancer_V5_Cache\train\` (`MATCH: True`)
- **Validation Cache Count:** `541` NPZ files on `F:\BreastCancer_V5_Cache\val\` (`MATCH: True`)
- **Total Files:** `3,042` NPZ files
- **Modification Status:** `0` modified / `0` regenerated / `0` deleted

## Test Quarantine
- **Test Reads:** `0` (`data/processed/test.csv` untouched, Last Modified: Sep 17, 2026)
- **Test Files:** `0` (`results/v5/test/` directory contains 0 files)
- **Test Predictions:** `0` generated
- **Test Masks:** `0` generated

---

## FINAL VERDICT

```
CHECKPOINT_AUDIT_PASSED
```
