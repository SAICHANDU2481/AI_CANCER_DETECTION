# V4-A ROI Cache & Validation Freeze Report

**Project:** Advanced Breast Cancer Diagnosis Using Wiener Filtering, Vision Transformer, and U-KAN-Based Medical Image Analysis  
**Phase:** V4-A Deterministic ROI Caching, Verification, & Validation Freeze  
**Execution Date/Time:** 2026-09-21 21:08:59  
**Status:** **CACHE CREATION & INTEGRITY VERIFIED (100% TEST QUARANTINED)**

---

## 1. Executive Summary & Verification Outcome

In accordance with the V4-A Validation Freeze directives, a fully deterministic, self-contained multi-scale ROI cache was constructed for the **training split ($2,501$ scans / $1,061$ patients)** and **validation split ($541$ scans / $235$ patients)**.

### Core Governance & Integrity Verification:
- **Winning Checkpoint Locked:** `models/v4/vit_v4_best.pth` (MD5: `d016fdc756758658e03d43f02fd4482c`, Validation Accuracy = $71.90\%$, Macro-F1 = $70.27\%$).
- **Deterministic Reproducibility:** Maximum absolute pixel difference across all crops = **$0.00000000$** ($0$ difference).
- **Exact ROI Source Match:** Validation Machine ROI ($60.63\%$), Expanded ROI ($6.65\%$), Context ROI ($0.00\%$), Fallback ($32.72\%$) match the completed audit benchmarks **EXACTLY**.
- **100% Test Set Quarantine:** `data/processed/test.csv` was **NEVER loaded, cached, or referenced**. Zero test ROIs were generated.

---

## 2. Cache Structure & Directory Layout

```
data/cache/v4/
├── train/                            # 2,501 deterministic .npz crop files
├── val/                              # 541 deterministic .npz crop files
├── metadata_train.csv                # 2,501 rows with full bbox & crop coordinates + 16 features
├── metadata_val.csv                  # 541 rows with full bbox & crop coordinates + 16 features
├── quality_stats_train.json          # Quality metric & source distribution summaries for train
├── quality_stats_val.json            # Quality metric & source distribution summaries for val
└── V4_ROI_CACHE_MANIFEST.json        # Formal cryptographic & parameter cache manifest
```

Each per-case `.npz` file contains:
- `local`: float32 `[224, 224]` array ($1.0\times$ local / expanded lesion crop)
- `medium`: float32 `[224, 224]` array ($1.5\times$ intermediate context crop)
- `large`: float32 `[224, 224]` array ($2.0\times$ extended margin crop)
- `context`: float32 `[224, 224]` array ($2.5\times - 3.0\times$ parenchymal reference crop)
- `quality_vector`: float32 `[16]` array (inference-only continuous quality descriptors)
- `roi_quality_score`: float32 scalar ($S_{\text{quality}} \in [0, 1]$)

---

## 3. Dataset Split & ROI Source Statistics

### 3.1 Cohort Sample Counts
| Split | Total Scans | Patients | Benign Cases | Malignant Cases | Class Ratio (B:M) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Training Split** | $2,501$ | $1,061$ | $1,470$ | $1,031$ | $1.43 : 1$ |
| **Validation Split** | $541$ | $235$ | $330$ | $211$ | $1.56 : 1$ |
| **Test Split** | $525$ | $235$ | $310$ | $215$ | *QUARANTINED (UNTOUCHED)* |

### 3.2 ROI Proposal Source Distribution
| Proposal Source | Training Count | Training Rate (%) | Validation Count | Validation Rate (%) | Expected Benchmark | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Machine ROI (`machine_roi`)** | $1,556$ | $62.22\%$ | $328$ | **$60.63\%$** | $60.63\%$ | **EXACT MATCH** |
| **Expanded ROI (`expanded_machine_roi`)** | $173$ | $6.92\%$ | $36$ | **$6.65\%$** | $6.65\%$ | **EXACT MATCH** |
| **Context ROI (`context_roi`)** | $0$ | $0.00\%$ | $0$ | **$0.00\%$** | $0.00\%$ | **EXACT MATCH** |
| **Tissue Centroid Fallback (`fallback_context`)**| $772$ | $30.87\%$ | $177$ | **$32.72\%$** | $32.72\%$ | **EXACT MATCH** |
| **Total** | **$2,501$** | **$100.00\%$** | **$541$** | **$100.00\%$** | **$100.00\%$** | **PASS** |

---

## 4. Quality Metric & Segmentation Distributions

### 4.1 ROI Quality Score ($S_{\text{quality}}$) Summary
| Split | Mean $\pm$ Std | Median | Min | 25th % | 75th % | IQR | Max |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train Cache** | $0.5471 \pm 0.3782$ | $0.7634$ | $0.0000$ | $0.0000$ | $0.8267$ | $[0.0000, 0.8267]$ | $1.0000$ |
| **Val Cache** | **$0.5332 \pm 0.3799$** | **$0.7551$** | $0.0000$ | $0.0000$ | $0.8237$ | **$[0.0000, 0.8237]$** | $1.0000$ |

### 4.2 Segmentation Dice Summary (Evaluation Metadata Only — Never Input Feature)
| Split | Mean $\pm$ Std | Median | Min | 25th % | 75th % | Max |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train Cache** | $0.2014 \pm 0.2981$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.4678$ | $0.9412$ |
| **Val Cache** | $0.1915 \pm 0.2958$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.4411$ | $0.9281$ |

---

## 5. Coordinate Validation & Bounding Box Checks

All bounding box and multi-scale crop coordinates were rigorously validated across all $3,042$ cached scans:
1. **Coordinate Boundaries:** All `ymin, xmin, ymax, xmax` coordinates are bounded within $[0, 512]$.
2. **Dimension Validity:** $ymax > ymin$ and $xmax > xmin$ for all valid extraction boxes.
3. **Hierarchical Expansion:** For all cases:
   $$\text{Local Box Size} \le \text{Medium Box Size} \le \text{Large Box Size} \le \text{Context Box Size}$$
4. **Fallback Coordinate Separation:** For all $177$ validation fallback cases, `bbox_ymin = bbox_xmin = bbox_ymax = bbox_xmax = 0`, while crop coordinates reflect deterministic tissue-centroid placement (`cy, cx`).

---

## 6. Deterministic Reproducibility Test

A reproducibility verification was executed by re-running end-to-end U-KAN inference and crop generation on 10 randomly sampled validation cases:
- **Maximum Absolute Pixel Difference:** **$0.00000000$**
- **Quality Score Difference:** **$0.00000000$**
- **Metadata Attributes Identical:** **$100\%$ Identical**
- **Outcome:** **PASS** (Zero non-deterministic drift).

---

## 7. Baseline MD5 Checksum Governance

| Model / Artifact | Path | Expected MD5 | Actual MD5 | Governance Status |
|---|---|:---:|:---:|:---:|
| **V1 U-KAN** | `models/ukan_best.pth` | `3007cd864bb6d78f93d010b5ddeb8f12` | `3007cd864bb6d78f93d010b5ddeb8f12` | **LOCKED & VERIFIED** |
| **V1 ViT** | `models/vit_best.pth` | `d674ae94f9666d2a4b77ec6a8a63114f` | `d674ae94f9666d2a4b77ec6a8a63114f` | **LOCKED & VERIFIED** |
| **V2-A ViT** | `models/v2/vit_v2a_best.pth` | `cc332cf9a6f4ed70634fe6bf59677930` | `cc332cf9a6f4ed70634fe6bf59677930` | **LOCKED & VERIFIED** |
| **V2-B U-KAN** | `models/v2b/segmentation/ukan_v2b_best.pth` | `a9a18b540cfd23114313b3e6d4276595` | `a9a18b540cfd23114313b3e6d4276595` | **LOCKED & VERIFIED** |
| **V3-A ViT** | `models/v3/checkpoints/vit_v3a_best.pth` | `14306b17fe999b52f07eea2ae963c3fb` | `14306b17fe999b52f07eea2ae963c3fb` | **LOCKED & VERIFIED** |
| **V3-B ViT** | `models/v3b/checkpoints/vit_v3b_best.pth` | `4dbe5ae7ef9362e44a6ab2472457801e` | `4dbe5ae7ef9362e44a6ab2472457801e` | **LOCKED & VERIFIED** |
| **V4-A Winning ViT**| `models/v4/vit_v4_best.pth` | `d016fdc756758658e03d43f02fd4482c` | `d016fdc756758658e03d43f02fd4482c` | **FROZEN WINNER** |

---

## 8. Test Quarantine Verification

- **Test Cohort Status:** `data/processed/test.csv` (525 scans, 235 patients) remained **100% QUARANTINED**.
- **Zero Access:** No test images were preprocessed, no test ROIs were generated, no test labels were loaded, and no test predictions were computed.
- **Cache Isolation:** `data/cache/v4/` contains exclusively `train/` and `val/` subdirectories.

---

## 9. Final Cache Manifest Overview

The complete machine-readable manifest is stored at [V4_ROI_CACHE_MANIFEST.json](file:///c:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/data/cache/v4/V4_ROI_CACHE_MANIFEST.json).

```json
{
  "cache_version": "4.0.0",
  "pipeline": "V4-A Deterministic Multi-Scale ROI Cache",
  "source_segmentation_model": {
    "architecture": "MultiScaleUKAN",
    "checkpoint_path": "models/v2b/segmentation/ukan_v2b_best.pth",
    "md5_checksum": "a9a18b540cfd23114313b3e6d4276595"
  },
  "classifier_model": {
    "architecture": "MultiScaleViTV4",
    "selected_arm": "Arm E / F (Multi-Scale Visual Fusion)",
    "checkpoint_path": "models/v4/vit_v4_best.pth",
    "md5_checksum": "d016fdc756758658e03d43f02fd4482c",
    "validation_macro_f1_pct": 70.27,
    "validation_accuracy_pct": 71.90
  },
  "reproducibility_verification": {
    "max_absolute_pixel_difference": 0.0,
    "metadata_identical": true,
    "status": "PASS"
  }
}
```

---

## 10. FINAL STOP CONDITION

All ROI caching, coordinate logging, statistical validation, reproducibility testing, and baseline checksum verifications are **COMPLETE**.

The system is now **STOPPED** and awaiting explicit user authorization for final V4-A test evaluation.
