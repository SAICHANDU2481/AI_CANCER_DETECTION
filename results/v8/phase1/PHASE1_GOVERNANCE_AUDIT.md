# V8 Phase 1: Data & Integrity Governance Audit Report

**Date & Time:** October 1, 2026 | 23:55 IST  
**Environment:** Windows 11 | PyTorch 2.6.0+cu124 | NVIDIA GeForce RTX 2050 (4.0 GB VRAM)  
**Governance Scope:** Pre-Training Diagnostic & Multi-Source Proposal Localization Benchmark  

---

## 1. Executive Governance Summary

This audit certifies that **V8 Phase 1 (Localization Diagnostic & Multi-Source Proposal Benchmark)** has strictly complied with all data quarantine, cryptographic immutability, and evaluation integrity rules set forth by the project charter.

```
+==============================================================================+
|                      PHASE 1 GOVERNANCE AUDIT SUMMARY                        |
+==============================================================================+
| Split Quarantined:     TEST SET (526 records / 235 patients) -> ZERO ACCESS  |
| Active Cohort:         VALIDATION (541 scans / 235 patients)                 |
| Checkpoints Verified:  3 / 3 (V6 U-KAN, V6 ViT, V5 ViT)                      |
| Test Set File Reads:   0 (Zero reads of test.csv, test images, test masks)   |
| Test Set Cache Writes: 0 (Zero test directories created)                     |
| Ranking Protocol:      STRICTLY LABEL-FREE (No GT IoU / labels during rank)  |
| Decision Gate Status:  PASS — Multi-Source / High-Res Coverage Confirmed     |
+==============================================================================+
```

---

## 2. Checkpoint Cryptographic Integrity Verification

All protected baseline model checkpoints were verified against their authoritative MD5 hashes prior to the benchmark. No weights were modified or overwritten.

| Model / Checkpoint Role | File Path | Expected MD5 | Actual MD5 | Status |
|---|---|---|---|---|
| **V6 U-KAN (Arm G Best)** | `models/v6/segmentation/final/ukan_v6_arm_g_best.pth` | `68cfc38cfbddd93d6e4a6e692fffbb34` | `68cfc38cfbddd93d6e4a6e692fffbb34` | **MATCH (LOCKED)** |
| **V6 ViT Classifier (Arm C)** | `models/v6/classification/vit_v6_arm_c_best.pth` | `508250cfc1c07af1ea7d81cc2e5a7418` | `508250cfc1c07af1ea7d81cc2e5a7418` | **MATCH (LOCKED)** |
| **V5 ViT Classifier (Arm I)** | `models/v5/classification/vit_v5_arm_i_best.pth` | `ec80689a43a906f89a15d7148cf7b05e` | `ec80689a43a906f89a15d7148cf7b05e` | **MATCH (LOCKED)** |

---

## 3. Dataset Quarantine & Read Access Verification

Phase 1 benchmark evaluated solely the **Validation** development cohort. The held-out **Test** split remained strictly quarantined.

| Split | Scans (N) | Patients (N) | Patient Overlap with Other Splits | Total Reads in Phase 1 | Quarantine Status |
|---|---|---|---|---|---|
| **Train** | 2,501 | 1,096 | 0 | 0 (Phase 1 diagnostic was validation-focused) | Quarantined |
| **Validation** | 541 | 235 | 0 | 541 (Evaluated across 14 proposal arms) | Active (Dev) |
| **Test** | 526 | 235 | 0 | **0 (ZERO ACCESS)** | **STRICTLY LOCKED** |

### Test Access Verification Checks:
1. `test.csv` reads = **0**
2. Test image file reads = **0**
3. Test ground-truth mask reads = **0**
4. `results/v8/test/` directory existence = **Does NOT exist**
5. `data/cache/v8/test/` directory existence = **Does NOT exist**

---

## 4. Label-Free Ranking Protocol Verification

To eliminate data leakage and evaluation bias during proposal ranking:
1. **Allowed Ranking Signals:** U-KAN probability mass, morphological detector response amplitude, local standard deviation, and bounding-box compactness.
2. **Prohibited Signals:** Ground-truth lesion mask, ground-truth bounding box coordinates, ground-truth pathology (benign vs malignant), and ground-truth IoU overlap.
3. **Audit Confirmation:** Candidate proposals across Arms A through I were ranked purely by detector confidence before any IoU or containment metrics were computed against ground-truth evaluation masks.

---

## 5. Auditor Sign-Off

**Phase 1 Governance Gate:** **PASSED**  
All operations were deterministic, reproducible (Seed = 42), and free of test-set leakage.
