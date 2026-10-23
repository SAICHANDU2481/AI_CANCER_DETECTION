# Research Paper Figures Documentation & Provenance Guide

**Project Title:** Advanced Breast Cancer Diagnosis Using Wiener Filtering, Vision Transformer, and U-KAN-Based Medical Image Analysis  
**Formally Frozen Model:** V8 (Phase 2 Arm F: Dual-Resolution U-KAN K=3 Proposals + Fine-Tuned ViT Backbone)  
**Document Status:** Complete Scientific Figure Suite (IEEE / Springer / Elsevier / B.Tech Thesis Ready)  
**Primary Locked Test Cohort:** 525 scans / 235 quarantined patients (Zero Patient-Level Leakage)  

---

## 1. Executive Overview & Governance Compliance

This directory contains the complete set of scientific visualizations and publication-grade figures generated from the **formally frozen V8 research prototype**.

### Mandatory Scientific Governance Adherence:
1. **Strictly No Retraining or Re-tuning:** Checkpoints, evaluation thresholds (tau=0.50 classification, tau=0.15 segmentation), and locked predictions were preserved without modification.
2. **Zero Test Set Re-inference:** Quantitative metrics, ROC curves, PR curves, and confusion matrices utilize the authentic saved test predictions (`results/v8/phase3/v8_arm_f_test_predictions.csv`).
3. **Zero Synthetic / Fabricated Data:** All mammograms, masks, and contours originate from verified CBIS-DDSM records.
4. **Format Availability:** Every primary figure is provided in three distinct publication formats:
   - **PNG (600 DPI):** High-resolution raster for print and camera-ready submission.
   - **PDF (Vector):** Fully vectorized PDF with embedded typography for academic publisher insertion.
   - **SVG (Vector):** Scalable vector graphics for editable presentations and web dashboards.
5. **Clear Separation of V9:** V9 results are explicitly documented as a subsequent experimental extension and are never conflated with the frozen V8 final reported system.
6. **Academic Prototype Status:** Explicit disclaimers confirm that all visualizations reflect machine learning research outputs and do not constitute clinical diagnostic claims.

---

## 2. Master Figure Catalog & Paper Placement Guide

| Figure # | Descriptive Title | Recommended Manuscript Placement | Formats Available | Primary Source Artifact |
|---|---|---|---|---|
| **Figure 1** | End-to-End System Architecture of the Frozen V8 Prototype | Section III: Methodology / Proposed System Architecture | PNG (600 DPI), PDF, SVG | `Source code: app/inference.py, models/final_v8/vit_v8_final.pth` |
| **Figure 10A** | Composite Preprocessing Pipeline (2x3 Publication Layout) | Appendix / Supplementary Materials / Fast-track Manuscript | PNG (600 DPI), PDF, SVG | `Derived from Figure 3 artifacts` |
| **Figure 10B** | Composite Segmentation Comparison (2x3 Publication Layout) | Appendix / Supplementary Materials | PNG (600 DPI), PDF, SVG | `Derived from Figure 4 artifacts` |
| **Figure 10C** | Composite Classification Diagnostic Triptych (4-Panel Linear Layout) | Main Manuscript: Section IV-C | PNG (600 DPI), PDF, SVG | `Derived from Figure 5 artifacts` |
| **Figure 10D** | Composite Cross-Generation Model Comparison (3-Panel Layout) | Main Manuscript: Section IV-D | PNG (600 DPI), PDF, SVG | `Derived from Figure 6 artifacts` |
| **Figure 10E** | Composite System Architecture (Full Banner Layout) | Main Manuscript: Section III | PNG (600 DPI), PDF, SVG | `Derived from Figure 1 artifacts` |
| **Figure 10F** | Composite Qualitative Success and Failure Gallery | Main Manuscript: Section IV-E | PNG (600 DPI), PDF, SVG | `Derived from Figure 8 artifacts` |
| **Figure 2** | CBIS-DDSM Dataset Cohort Breakdown and Patient-Level Partitioning | Section IV-A: Dataset and Evaluation Setup | PNG (600 DPI), PDF, SVG | `data/processed/metadata.csv, results/final_v8/V8_FORMAL_FREEZE_REPORT.md` |
| **Figure 3** | Adaptive Spatial Wiener Filtering, CLAHE Enhancement, and Dynamic Range Expansion | Section III-B: Image Denoising and Preprocessing | PNG (600 DPI), PDF, SVG | `preprocessing/enhancement.py, preprocessing/wiener_filter.py` |
| **Figure 4** | Dual-Resolution U-KAN Lesion Segmentation and Pixel Error Decomposition | Section IV-B: Lesion Localization and Segmentation Performance | PNG (600 DPI), PDF, SVG | `models/final_v8/ukan_v6_final.pth, segmentation/v6/ukan_v6.py` |
| **Figure 5** | Primary Diagnostic Performance and Probability Calibration of Frozen V8 Arm F | Section IV-C: Classification Performance of the Final V8 Model | PNG (600 DPI), PDF, SVG | `results/v8/phase3/v8_arm_f_test_predictions.csv, results/v8/phase3/v8_arm_f_test_calibration.json` |
| **Figure 6** | Cross-Generational Model Evolution (V5 -> V6 -> V8 -> V9) | Section IV-D: Ablation Studies and Cross-Generational Comparison | PNG (600 DPI), PDF, SVG | `results/final_v8/V5_V6_V8_V9_COMPARISON.csv, results/final_v8/V8_FORMAL_FREEZE_REPORT.md` |
| **Figure 7** | Subgroup Error Characterization and Failure Mode Decomposition | Section V: Discussion and Error Analysis | PNG (600 DPI), PDF, SVG | `results/v8/phase3/v8_arm_f_test_predictions.csv` |
| **Figure 8** | Qualitative Case Gallery of Verified Test Diagnoses | Section IV-E: Qualitative Findings and Clinical Interpretability | PNG (600 DPI), PDF, SVG | `results/v8/phase3/v8_arm_f_test_predictions.csv, data/processed/metadata.csv` |
| **Figure 9** | Vision Transformer 12-Layer Attention Rollout and Interpretability Saliency | Section IV-F: Explainable AI and Attention Visualization | PNG (600 DPI), PDF, SVG | `models/final_v8/vit_v8_final.pth, explainability/attention.py` |

---

## 3. Comprehensive Figure Profiles & Publication Captions

### Figure 1: End-to-End System Architecture of the Frozen V8 Prototype

- **Target Placement:** Section III: Methodology / Proposed System Architecture
- **Cohort Used:** N/A (Architectural Specification)
- **Sample Size:** 0
- **Underlying Source Data:** `Source code: app/inference.py, models/final_v8/vit_v8_final.pth`
- **Governance Protocol:** Frozen V8 Arm F Architecture; V9 excluded from primary pipeline.

#### Paper-Ready Publication Caption:
> Figure 1: Complete architectural schematic of the formally frozen V8 breast cancer diagnostic research pipeline. The pipeline ingests raw 16-bit/8-bit FFDM mammograms, applies 5x5 adaptive spatial Wiener filtering for quantum noise mottle attenuation, followed by CLAHE contrast enhancement and percentile normalization. Multi-scale candidate extraction utilizes dual-resolution U-KAN (512x512 and 768x768 streams) with deterministic top-K=3 proposal generation. Multi-scale concentric RoI crops (Local 1.2x, Medium 1.8x, Large 2.5x, Context 4.0x) alongside high-pass calcification and global patches form a 4,608-dimensional joint latent representation fed to a candidate-softmax aggregated Vision Transformer with deep MLP classification head.

#### Associated Output Files in Repository:
- [fig01_system_architecture.pdf](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_01_system_architecture/fig01_system_architecture.pdf) (PDF, 56.6 KB)
- [fig01_system_architecture.png](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_01_system_architecture/fig01_system_architecture.png) (PNG, 1.78 MB)
- [fig01_system_architecture.svg](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_01_system_architecture/fig01_system_architecture.svg) (SVG, 244.3 KB)

---

### Figure 2: CBIS-DDSM Dataset Cohort Breakdown and Patient-Level Partitioning

- **Target Placement:** Section IV-A: Dataset and Evaluation Setup
- **Cohort Used:** CBIS-DDSM Full Cohort (Train: 2,501, Val: 541, Test: 525 scans)
- **Sample Size:** 3568
- **Underlying Source Data:** `data/processed/metadata.csv, results/final_v8/V8_FORMAL_FREEZE_REPORT.md`
- **Governance Protocol:** Zero patient overlap across train (1,096 pts), val (235 pts), and test (235 pts) splits.

#### Paper-Ready Publication Caption:
> Figure 2: Distribution and demographic characteristics of the curated CBIS-DDSM cohort across 3,568 mammographic scans from 1,566 patients. (a) Pathology balance: 59.16% benign vs 40.84% malignant. (b) Abnormality distribution: 52.45% calcifications vs 47.55% masses. (c) Scan partitioning: 2,501 training (70.1%), 541 validation (15.2%), and 525 locked test (14.7%). (d) Quarantined patient counts per split with strictly zero patient-level leakage. (e) Projection view split: 53.13% MLO vs 46.87% CC. (f) Pathology proportions preserved across splits.

#### Associated Output Files in Repository:
- [fig02_dataset_distribution.pdf](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_02_dataset_analysis/fig02_dataset_distribution.pdf) (PDF, 36.0 KB)
- [fig02_dataset_distribution.png](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_02_dataset_analysis/fig02_dataset_distribution.png) (PNG, 822.7 KB)
- [fig02_dataset_distribution.svg](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_02_dataset_analysis/fig02_dataset_distribution.svg) (SVG, 140.7 KB)

---

### Figure 3: Adaptive Spatial Wiener Filtering, CLAHE Enhancement, and Dynamic Range Expansion

- **Target Placement:** Section III-B: Image Denoising and Preprocessing
- **Cohort Used:** CBIS-DDSM Verified FFDM Mammogram (MASS_P_00001_LEFT_CC)
- **Sample Size:** 1
- **Underlying Source Data:** `preprocessing/enhancement.py, preprocessing/wiener_filter.py`
- **Governance Protocol:** Verified genuine mammogram; zero synthetic phantom substitution.

#### Paper-Ready Publication Caption:
> Figure 3: Multi-stage mammographic image preprocessing pipeline on a representative malignant case (MASS_P_00001_LEFT_CC). (a) Raw normalized FFDM canvas showing low glandular contrast and quantum noise. (b) 5x5 adaptive 2D spatial Wiener denoising attenuating local variance fluctuations. (c) CLAHE contrast enhancement (clip limit 2.5, 8x8 tile grid) accentuating microcalcifications and parenchymal tissue boundaries. (d) Absolute residual difference map (|Enhanced - Raw|) illustrating noise mottle removal and edge enhancement. (e1-e3) High-magnification RoI insets tracing boundary enhancement. (f) Quantitative metric gains (+3.62 dB SNR, +0.67 bits entropy). (g) Cumulative intensity histograms showing full dynamic range expansion. (h) Spatial line profile transect confirming margin sharpness.

#### Associated Output Files in Repository:
- [fig03_preprocessing_pipeline.pdf](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_03_preprocessing/fig03_preprocessing_pipeline.pdf) (PDF, 773.6 KB)
- [fig03_preprocessing_pipeline.png](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_03_preprocessing/fig03_preprocessing_pipeline.png) (PNG, 3.63 MB)
- [fig03_preprocessing_pipeline.svg](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_03_preprocessing/fig03_preprocessing_pipeline.svg) (SVG, 1.29 MB)

---

### Figure 4: Dual-Resolution U-KAN Lesion Segmentation and Pixel Error Decomposition

- **Target Placement:** Section IV-B: Lesion Localization and Segmentation Performance
- **Cohort Used:** Validation Benchmark Scans (Malignant Mass, Benign Mass, Calcification, Failure Case)
- **Sample Size:** 4
- **Underlying Source Data:** `models/final_v8/ukan_v6_final.pth, segmentation/v6/ukan_v6.py`
- **Governance Protocol:** Evaluated with frozen checkpoint weights; threshold tau=0.15 strictly enforced.

#### Paper-Ready Publication Caption:
> Figure 4: Qualitative lesion segmentation evaluation of the frozen U-KAN model across four representative clinical cases. Columns: (a) Raw mammogram canvas, (b) Preprocessed input, (c) Expert radiologist ground-truth annotation mask, (d) Continuous U-KAN predicted posterior probability map, (e) Contour overlay comparing ground truth (green) with model prediction (magenta), (f) Pixel-level error decomposition highlighting True Positives (green), False Positives (red), and False Negatives (blue). Rows represent: (1) Malignant spiculated mass (Dice=0.365, IoU=0.224), (2) Benign well-circumscribed mass (Dice=0.537, IoU=0.367), (3) Clustered pleomorphic calcification (Dice=0.429, IoU=0.273), and (4) Diffuse benign microcalcification failure case (Dice=0.000, IoU=0.000).

#### Associated Output Files in Repository:
- [fig04_ukan_segmentation_evaluation.pdf](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_04_segmentation/fig04_ukan_segmentation_evaluation.pdf) (PDF, 2.50 MB)
- [fig04_ukan_segmentation_evaluation.png](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_04_segmentation/fig04_ukan_segmentation_evaluation.png) (PNG, 9.98 MB)
- [fig04_ukan_segmentation_evaluation.svg](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_04_segmentation/fig04_ukan_segmentation_evaluation.svg) (SVG, 3.65 MB)

---

### Figure 5: Primary Diagnostic Performance and Probability Calibration of Frozen V8 Arm F

- **Target Placement:** Section IV-C: Classification Performance of the Final V8 Model
- **Cohort Used:** Locked Held-Out Test Cohort (525 scans / 235 quarantined patients)
- **Sample Size:** 525
- **Underlying Source Data:** `results/v8/phase3/v8_arm_f_test_predictions.csv, results/v8/phase3/v8_arm_f_test_calibration.json`
- **Governance Protocol:** Strictly frozen evaluation results. 0 post-hoc threshold tuning, 0 test reruns.

#### Paper-Ready Publication Caption:
> Figure 5: Comprehensive held-out test evaluation of the final reported V8 model (Arm F) on 525 independent scans from 235 quarantined patients. (a) Absolute confusion matrix (TN=197, FP=113, FN=73, TP=142). (b) Row-normalized confusion matrix (Specificity: 63.55%, Sensitivity: 66.05%). (c) Empirical ROC curve achieving verified AUC = 0.6947 [95% CI: 0.6309 - 0.7474]. (d) Precision-Recall curve achieving verified PR-AUC = 0.5857 [95% CI: 0.5059 - 0.6723] against a 40.95% malignant baseline prevalence. (e) Diagnostic metric profile with 95% patient-clustered bootstrap confidence intervals (Accuracy: 64.57%, Macro-F1: 64.18%, Binary F1: 60.43%). (f) Class-disaggregated precision, recall, and F1 comparisons. (g) Posterior probability distributions. (h) 10-bin reliability diagram yielding Brier score = 0.3143 and ECE = 29.87%. (i) Calibration audit summary confirming raw logit evaluation protocol.

#### Associated Output Files in Repository:
- [fig05_classification_evaluation.pdf](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_05_classification/fig05_classification_evaluation.pdf) (PDF, 78.1 KB)
- [fig05_classification_evaluation.png](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_05_classification/fig05_classification_evaluation.png) (PNG, 2.02 MB)
- [fig05_classification_evaluation.svg](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_05_classification/fig05_classification_evaluation.svg) (SVG, 343.9 KB)

---

### Figure 6: Cross-Generational Model Evolution (V5 -> V6 -> V8 -> V9)

- **Target Placement:** Section IV-D: Ablation Studies and Cross-Generational Comparison
- **Cohort Used:** Locked Test Cohorts across Model Iterations (N=525 scans)
- **Sample Size:** 525
- **Underlying Source Data:** `results/final_v8/V5_V6_V8_V9_COMPARISON.csv, results/final_v8/V8_FORMAL_FREEZE_REPORT.md`
- **Governance Protocol:** V8 is permanently frozen as the final reported model; V9 documented as an experimental extension.

#### Paper-Ready Publication Caption:
> Figure 6: Systematic benchmark comparison across model generations on the locked test cohort. (a) Grouped bar chart comparing classification Accuracy and Macro-F1 across V5 (65.02% / 64.73%), V6 (64.45% / 64.11%), V8 (64.57% / 64.18%, formally frozen final reported model), and V9 (62.86% / 62.86%, experimental extension). (b) Discrimination metrics comparison: ROC-AUC (V5: 0.6988, V6: 0.7087, V8: 0.6947, V9: 0.7046) and PR-AUC (V5: 0.5916, V6: 0.5975, V8: 0.5857, V9: 0.6129). (c) Sensitivity versus Specificity trade-off illustrating V9's sensitivity elevation (+10.2%) accompanied by a marked drop in specificity (-10.0%). (d) Accuracy vs Sensitivity operating space. (e) Multi-metric radar comparison. (f) Historical evolution matrix documenting feature dimensionality and architectural changes.

#### Associated Output Files in Repository:
- [fig06_generation_comparison.pdf](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_06_model_comparison/fig06_generation_comparison.pdf) (PDF, 55.1 KB)
- [fig06_generation_comparison.png](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_06_model_comparison/fig06_generation_comparison.png) (PNG, 1.91 MB)
- [fig06_generation_comparison.svg](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_06_model_comparison/fig06_generation_comparison.svg) (SVG, 250.2 KB)

---

### Figure 7: Subgroup Error Characterization and Failure Mode Decomposition

- **Target Placement:** Section V: Discussion and Error Analysis
- **Cohort Used:** Locked Test Scans (N=525 scans, 186 errors)
- **Sample Size:** 525
- **Underlying Source Data:** `results/v8/phase3/v8_arm_f_test_predictions.csv`
- **Governance Protocol:** Descriptive subgroup analysis; zero causal clinical inference asserted.

#### Paper-Ready Publication Caption:
> Figure 7: Stratified error analysis of the locked V8 model across 525 test cases (186 errors: 113 False Positives, 73 False Negatives). (a) Error taxonomy breakdown (FP: 60.8% of errors vs FN: 39.2% of errors). (b) Diagnostic errors categorized by lesion type: Mass (65 FP, 34 FN) vs Calcification (48 FP, 39 FN). (c) Errors by acquisition projection: CC View (47 FP, 38 FN) vs MLO View (66 FP, 35 FN). (d) Classification error rate as a function of U-KAN candidate IoU overlap strata. (e) Model decision confidence distributions showing substantial posterior overlap between correct and erroneous predictions. (f) Subgroup accuracy breakdown across anatomical and pathological subsets. (g) Historical false positive and false negative count progression from V5 through V9. (h) Methodological audit detailing sample sizes and governance constraints.

#### Associated Output Files in Repository:
- [fig07_error_analysis.pdf](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_07_error_analysis/fig07_error_analysis.pdf) (PDF, 69.2 KB)
- [fig07_error_analysis.png](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_07_error_analysis/fig07_error_analysis.png) (PNG, 1.36 MB)
- [fig07_error_analysis.svg](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_07_error_analysis/fig07_error_analysis.svg) (SVG, 282.7 KB)

---

### Figure 8: Qualitative Case Gallery of Verified Test Diagnoses

- **Target Placement:** Section IV-E: Qualitative Findings and Clinical Interpretability
- **Cohort Used:** Locked Test Cohort (6 verified representative scans)
- **Sample Size:** 6
- **Underlying Source Data:** `results/v8/phase3/v8_arm_f_test_predictions.csv, data/processed/metadata.csv`
- **Governance Protocol:** Authentic test images and masks; fully anonymized case identifiers.

#### Paper-Ready Publication Caption:
> Figure 8: Representative qualitative case gallery illustrating the spectrum of V8 model diagnostic predictions on verified held-out test scans. Panels depict: (a) Raw mammogram canvas, (b) Wiener + CLAHE enhanced input, (c) Expert ground-truth lesion annotation, (d) Top proposal candidate detection box, (e) Complete diagnostic decision profile card detailing ground truth, model prediction, posterior probabilities, localization IoU/Dice, and neutral radiological observations. Cases 1-4 illustrate successful classifications: Case 1 (True Positive Malignant Mass, P(M)=0.748), Case 2 (True Positive Malignant Calcification, P(M)=0.978), Case 3 (True Negative Benign Mass, P(M)=0.029, IoU=0.613), Case 4 (True Negative Benign Calcification, P(M)=0.026). Cases 5-6 illustrate model failure modes: Case 5 (False Positive dense tissue overcall, P(M)=0.584), Case 6 (False Negative MLO projection miss, P(M)=0.215).

#### Associated Output Files in Repository:
- [fig08_qualitative_case_gallery.pdf](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_08_qualitative_results/fig08_qualitative_case_gallery.pdf) (PDF, 3.97 MB)
- [fig08_qualitative_case_gallery.png](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_08_qualitative_results/fig08_qualitative_case_gallery.png) (PNG, 17.04 MB)
- [fig08_qualitative_case_gallery.svg](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_08_qualitative_results/fig08_qualitative_case_gallery.svg) (SVG, 5.87 MB)

---

### Figure 9: Vision Transformer 12-Layer Attention Rollout and Interpretability Saliency

- **Target Placement:** Section IV-F: Explainable AI and Attention Visualization
- **Cohort Used:** Development Benchmark Cases (MASS_P_00001 and MASS_P_00004)
- **Sample Size:** 2
- **Underlying Source Data:** `models/final_v8/vit_v8_final.pth, explainability/attention.py`
- **Governance Protocol:** Extracted via authentic forward hooks on the 12-block ViT backbone; test quarantine preserved.

#### Paper-Ready Publication Caption:
> Figure 9: Visual explanation of Vision Transformer feature representation using authentic 12-layer Attention Rollout. Rows: (1) Biopsy-proven malignant spiculated mass (MASS_P_00001_LEFT_CC), (2) Biopsy-proven benign circumscribed fibroadenoma (MASS_P_00004_LEFT_CC). Columns: (a) Full FFDM mammogram indicating localized candidate RoI box, (b) High-resolution candidate RoI crop (224x224), (c) Continuous 12-layer Attention Rollout saliency map, (d) Blended saliency overlay illustrating model focus on radiating spicular margins in the malignant case versus central homogeneous attenuation in the benign case, (e) Multi-scale context (4.0x) attention distribution, (f) Mathematical formulation detailing layer-wise residual attention propagation and mandatory scientific disclaimer.

#### Associated Output Files in Repository:
- [fig09_attention_explainability.pdf](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_09_explainability/fig09_attention_explainability.pdf) (PDF, 1.87 MB)
- [fig09_attention_explainability.png](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_09_explainability/fig09_attention_explainability.png) (PNG, 4.15 MB)
- [fig09_attention_explainability.svg](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_09_explainability/fig09_attention_explainability.svg) (SVG, 2.85 MB)

---

### Figure 10A: Composite Preprocessing Pipeline (2x3 Publication Layout)

- **Target Placement:** Appendix / Supplementary Materials / Fast-track Manuscript
- **Cohort Used:** CBIS-DDSM Mammogram (MASS_P_00001_LEFT_CC)
- **Sample Size:** 1
- **Underlying Source Data:** `Derived from Figure 3 artifacts`
- **Governance Protocol:** IEEE single-column / double-column ready composite format.

#### Paper-Ready Publication Caption:
> Figure 10A: Ready-to-insert 2x3 composite visualization of mammographic preprocessing: Raw, Wiener denoised, CLAHE enhanced, residual difference, histogram, and spatial margin transect.

#### Associated Output Files in Repository:
- [fig10a_composite_preprocessing_2x3.pdf](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10a_composite_preprocessing_2x3.pdf) (PDF, 1.14 MB)
- [fig10a_composite_preprocessing_2x3.png](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10a_composite_preprocessing_2x3.png) (PNG, 4.65 MB)
- [fig10a_composite_preprocessing_2x3.svg](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10a_composite_preprocessing_2x3.svg) (SVG, 1.74 MB)

---

### Figure 10B: Composite Segmentation Comparison (2x3 Publication Layout)

- **Target Placement:** Appendix / Supplementary Materials
- **Cohort Used:** Representative Validation Cases
- **Sample Size:** 2
- **Underlying Source Data:** `Derived from Figure 4 artifacts`
- **Governance Protocol:** Ready-to-insert 2x3 layout showing mass and calcification segmentation.

#### Paper-Ready Publication Caption:
> Figure 10B: 2x3 composite layout illustrating U-KAN segmentation and pixel error decomposition for mass and calcification lesions.

#### Associated Output Files in Repository:
- [fig10b_composite_segmentation_2x3.pdf](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10b_composite_segmentation_2x3.pdf) (PDF, 1.07 MB)
- [fig10b_composite_segmentation_2x3.png](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10b_composite_segmentation_2x3.png) (PNG, 3.71 MB)
- [fig10b_composite_segmentation_2x3.svg](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10b_composite_segmentation_2x3.svg) (SVG, 1.55 MB)

---

### Figure 10C: Composite Classification Diagnostic Triptych (4-Panel Linear Layout)

- **Target Placement:** Main Manuscript: Section IV-C
- **Cohort Used:** Locked Test Cohort (525 scans)
- **Sample Size:** 525
- **Underlying Source Data:** `Derived from Figure 5 artifacts`
- **Governance Protocol:** Locked test results: Confusion Matrix, ROC, PR curve, and bootstrap metric bars.

#### Paper-Ready Publication Caption:
> Figure 10C: Compact classification performance composite combining the locked confusion matrix, empirical ROC curve (AUC=0.6947), PR curve (AUC=0.5857), and diagnostic metric scores with 95% CIs.

#### Associated Output Files in Repository:
- [fig10c_composite_classification_triptych.pdf](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10c_composite_classification_triptych.pdf) (PDF, 40.2 KB)
- [fig10c_composite_classification_triptych.png](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10c_composite_classification_triptych.png) (PNG, 707.7 KB)
- [fig10c_composite_classification_triptych.svg](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10c_composite_classification_triptych.svg) (SVG, 123.0 KB)

---

### Figure 10D: Composite Cross-Generation Model Comparison (3-Panel Layout)

- **Target Placement:** Main Manuscript: Section IV-D
- **Cohort Used:** Historical Comparison Cohorts (N=525 scans)
- **Sample Size:** 525
- **Underlying Source Data:** `Derived from Figure 6 artifacts`
- **Governance Protocol:** V8 clearly highlighted as the final reported model.

#### Paper-Ready Publication Caption:
> Figure 10D: 3-panel composite layout contrasting Accuracy/Macro-F1, ROC/PR AUC, and Sensitivity-Specificity trade-offs across V5, V6, V8, and V9.

#### Associated Output Files in Repository:
- [fig10d_composite_model_comparison.pdf](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10d_composite_model_comparison.pdf) (PDF, 33.2 KB)
- [fig10d_composite_model_comparison.png](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10d_composite_model_comparison.png) (PNG, 421.0 KB)
- [fig10d_composite_model_comparison.svg](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10d_composite_model_comparison.svg) (SVG, 90.4 KB)

---

### Figure 10E: Composite System Architecture (Full Banner Layout)

- **Target Placement:** Main Manuscript: Section III
- **Cohort Used:** System Design Specification
- **Sample Size:** 0
- **Underlying Source Data:** `Derived from Figure 1 artifacts`
- **Governance Protocol:** Vector format suitable for two-column spanning banner.

#### Paper-Ready Publication Caption:
> Figure 10E: Two-column banner diagram of the complete V8 system architecture.

#### Associated Output Files in Repository:
- [fig10e_composite_architecture.pdf](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10e_composite_architecture.pdf) (PDF, 56.6 KB)
- [fig10e_composite_architecture.png](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10e_composite_architecture.png) (PNG, 1.78 MB)
- [fig10e_composite_architecture.svg](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10e_composite_architecture.svg) (SVG, 244.3 KB)

---

### Figure 10F: Composite Qualitative Success and Failure Gallery

- **Target Placement:** Main Manuscript: Section IV-E
- **Cohort Used:** Locked Test Cohort (6 scans)
- **Sample Size:** 6
- **Underlying Source Data:** `Derived from Figure 8 artifacts`
- **Governance Protocol:** Representative clinical case matrix with complete diagnostic profiles.

#### Paper-Ready Publication Caption:
> Figure 10F: Comprehensive multi-case qualitative gallery demonstrating clinical successes (TP, TN) and failure modes (FP, FN) on the locked test cohort.

#### Associated Output Files in Repository:
- [fig10f_composite_qualitative_gallery.pdf](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10f_composite_qualitative_gallery.pdf) (PDF, 3.97 MB)
- [fig10f_composite_qualitative_gallery.png](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10f_composite_qualitative_gallery.png) (PNG, 17.04 MB)
- [fig10f_composite_qualitative_gallery.svg](file:///C:/Users/sys/OneDrive/Desktop/AI_CANCER_DETECTION/results/research_paper_figures/figure_10_publication_layouts/fig10f_composite_qualitative_gallery.svg) (SVG, 5.87 MB)

---

## 4. Reproducibility Scripts Directory

All figures can be deterministically reproduced using the standalone scripts located in `scripts/research_figures/`:

| Script Name | Target Output Figure | Primary Dependencies |
|---|---|---|
| `generate_fig01_architecture.py` | Figure 1 & Figure 10E | Matplotlib Patches, Vector Drawing |
| `generate_fig02_dataset.py` | Figure 2 | Pandas, NumPy, CBIS-DDSM Metadata |
| `generate_fig03_preprocessing.py` | Figure 3 & Figure 10A | OpenCV, SciPy Signal, NumPy |
| `generate_fig04_segmentation.py` | Figure 4 & Figure 10B | PyTorch, U-KAN Checkpoint, OpenCV |
| `generate_fig05_classification.py` | Figure 5 & Figure 10C | Scikit-learn, Seaborn, Matplotlib |
| `generate_fig06_comparison.py` | Figure 6 & Figure 10D | Pandas, Matplotlib Polar/Radar |
| `generate_fig07_error_analysis.py` | Figure 7 | Pandas, Subgroup Stratification |
| `generate_fig08_qualitative.py` | Figure 8 & Figure 10F | OpenCV, Matplotlib, Verified Scans |
| `generate_fig09_explainability.py` | Figure 9 | PyTorch ViT Backbone, Attention Rollout |
| `generate_fig10_publication_layouts.py` | Figure 10 (A-F) | Matplotlib Composite Builders |
| `generate_all_figures.py` | Complete Suite + Manifest | Master Pipeline Orchestrator |

### How to Regenerate All Figures:
```bash
# Run master suite
python scripts/research_figures/generate_all_figures.py
```

---

## 5. Verified Locked Model Results (Reference Standard)

| Diagnostic Metric | Verified Locked V8 Value | 95% Bootstrap Confidence Interval |
|---|---|---|
| **Overall Accuracy** | **64.57%** | [59.06% – 69.79%] |
| **Macro-Average F1** | **64.18%** | [58.80% – 69.01%] |
| **Binary F1 (Malignant)** | **60.43%** | [54.03% – 66.09%] |
| **ROC-AUC Score** | **0.6947** | [0.6309 – 0.7474] |
| **PR-AUC Score (Average Precision)** | **0.5857** | [0.5059 – 0.6723] |
| **Sensitivity (Recall on Malignant)** | **66.05%** | [58.85% – 72.93%] |
| **Specificity (Benign Retention)** | **63.55%** | [54.58% – 71.56%] |
| **Precision (PPV)** | **55.69%** | [47.88% – 63.27%] |
| **Brier Score (Probability Error)** | **0.3143** | Reference: 10 Uniform Bins |
| **Expected Calibration Error (ECE)** | **29.87%** | Raw ViT Logit Evaluation |

**Confusion Matrix:**
- True Negatives (TN): **197**
- False Positives (FP): **113**
- False Negatives (FN): **73**
- True Positives (TP): **142**
- Total Primary Cohort: **525 Scans / 235 Quarantined Patients**

---
*Document generated automatically as part of the formal research documentation protocol.*