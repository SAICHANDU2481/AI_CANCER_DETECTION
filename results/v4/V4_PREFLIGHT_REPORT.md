# V4-A Pre-Flight Audit Report

## 1. Executive Summary & Verification Outcome
- **Experiment Stage:** V4-A Multi-Scale ROI Quality & Fusion Pre-Flight Audit
- **Audit Date/Time:** 2026-09-21 16:36:31
- **Target Hardware:** NVIDIA RTX 2050 (4GB VRAM) / PyTorch 2.6.0+cu124
- **Audit Outcome:** **ALL 22 / 22 CHECKS PASSED**

---

## 2. 22-Point Pre-Flight Verification Results

| Index & Verification Check | Status |
|---|:---:|
| **1. Baseline Checksums** | **PASS** |
| **2. Patient Isolation (0 Overlap)** | **PASS** |
| **3. Programmatic Test Quarantine Guard** | **PASS** |
| **4. Exact 16 Quality Features** | **PASS** |
| **5. Deterministic Scalar Quality Score** | **PASS** |
| **6. No GT Mask Access in Scoring** | **PASS** |
| **7. No Pathology Access in Scoring** | **PASS** |
| **8. Explicit ROI Source Labels** | **PASS** |
| **9. Primary Machine ROI Generation** | **PASS** |
| **10. Expanded Machine ROI Generation** | **PASS** |
| **11. Context ROI Generation** | **PASS** |
| **12. Fallback Context Generation** | **PASS** |
| **13. Correct Tensor Dimensions [B, 3088] -> [B, 2]** | **PASS** |
| **14. Model Forward Pass** | **PASS** |
| **15. Model Backward Pass** | **PASS** |
| **16. Non-Zero Gradients Across Trainable Parameters** | **PASS** |
| **17. Zero NaN Values** | **PASS** |
| **18. Zero Inf Values** | **PASS** |
| **19. RTX 2050 VRAM Benchmark (Batch 8 Safe)** | **PASS** |
| **20. AMP FP16 Numerical Stability** | **PASS** |
| **21. Artifact Directory Integrity** | **PASS** |
| **22. Reproducibility with Seed 42** | **PASS** |

---

## 3. Detailed Verification Notes

1. **Baseline MD5 Checksum Governance:**
   - V1 U-KAN: `3007cd864bb6d78f93d010b5ddeb8f12` (**PASS**)
   - V1 ViT: `d674ae94f9666d2a4b77ec6a8a63114f` (**PASS**)
   - V2-A ViT: `cc332cf9a6f4ed70634fe6bf59677930` (**PASS**)
   - V2-B U-KAN: `a9a18b540cfd23114313b3e6d4276595` (**PASS**)
   - V3-A Best: `14306b17fe999b52f07eea2ae963c3fb` (**PASS**)
   - V3-B Best: `4dbe5ae7ef9362e44a6ab2472457801e` (**PASS**)

2. **Patient Isolation & Zero Leakage:**
   - Train Cohort: $2,501$ scans ($1,061$ patients)
   - Validation Cohort: $541$ scans ($235$ patients)
   - Test Cohort: $525$ scans ($235$ patients)
   - **Overlap:** Train $\cap$ Val = $0$, Train $\cap$ Test = $0$, Val $\cap$ Test = $0$.

3. **Exact 16 Quality Feature Vector:**
   - 16 features extracted strictly during inference:
     `predicted_mask_area, normalized_mask_area, bbox_width, bbox_height, bbox_aspect_ratio, bbox_area, mask_confidence, mask_mean_probability, mask_max_probability, mask_probability_std, mask_entropy, connected_component_count, component_compactness, normalized_center_distance, boundary_touch_ratio, mask_fill_ratio`
   - Assertion `quality_vector.shape[-1] == 16` verified.
   - Deterministic scalar $S_{\text{quality}} \in [0, 1]$ computed without ground-truth masks or pathology labels.

4. **Hardware VRAM & AMP Benchmark:**
   - Batch size: $8$ (4 scales: Local, Medium, Large, Context $[8, 3, 224, 224]$ + Quality $[8, 16]$)
   - Peak GPU VRAM: **527.6 MB** (Well within the $4,096$ MB RTX 2050 envelope)
   - AMP FP16 forward and backward passes executed with zero NaN / Inf.

---

## 4. STOP & Governance Protocol
> [!IMPORTANT]
> **Pre-Flight Complete:** In accordance with the governance protocol, training is **NOT started**. Execution has halted. All baseline checkpoints and quarantined test data remain protected.
