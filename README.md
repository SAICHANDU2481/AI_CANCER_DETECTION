# Advanced Breast Cancer Diagnosis Using Wiener Filtering, Vision Transformer, and U-KAN-Based Medical Image Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)
[![License: Academic Research](https://img.shields.io/badge/License-Academic_Research-green.svg)]()
[![Model Status: V8 Frozen](https://img.shields.io/badge/Model_Status-V8_Frozen-success.svg)]()

> **IMPORTANT RESEARCH & SAFETY NOTICE:**  
> This software is an **AI-assisted academic research prototype** developed exclusively for scientific exploration and peer-reviewed medical AI investigation on the CBIS-DDSM dataset. It is **NOT** an FDA/CE-cleared clinical diagnostic device. Predictions, segmentation contours, and attention heatmaps are research artifacts and must never replace qualified clinical radiological interpretation or primary clinical decision-making.

---

## 📌 Project Status & Final Model Designation

- **FINAL REPORTED MODEL:** **V8 (Phase 2 Arm F: Dual-Resolution U-KAN $K=3$ Proposals + Last 4 ViT Blocks Fine-Tuned)**
- **MODEL STATUS:** **FROZEN**
- **TEST BENCHMARK:** **LOCKED** (Quarantined Held-Out Test Split)
- **V9 STATUS:** **DOCUMENTED EXPERIMENTAL EXTENSION** (Retained in research record)
- **FURTHER TEST OPTIMIZATION:** **PROHIBITED**
- **FURTHER MODEL TRAINING:** **NOT REQUIRED FOR FINAL REPORT**

---

## 📖 Project Overview & Core Methodology

This academic research project develops an end-to-end computer-aided breast cancer diagnosis framework for full-field digital mammography (FFDM) from the **CBIS-DDSM** dataset:

1. **Adaptive Wiener Spatial Denoising:** Suppresses sensor quantum mottle and digitization noise while preserving high-frequency microcalcification boundaries.
2. **Contrast-Limited Adaptive Histogram Equalization (CLAHE):** Enhances local tissue density gradients and glandular tissue contrast.
3. **Dual-Resolution U-KAN Lesion Localization ($512\times 512 + 768\times 768$):** Employs Kolmogorov-Arnold Network (KAN) B-spline activation functions to accurately segment suspicious mass margins and diffuse calcification clusters.
4. **Deterministic Machine Connected-Component Proposals:** Extracts, ranks, and spatial-suppresses candidate lesion regions ($K=3$).
5. **Multi-Scale Concentric RoI Extraction:** Extracts concentric receptive fields:
   - **Local ($1.2\times$):** Lesion interior & microcalcification architecture
   - **Medium ($1.8\times$):** Perilesional boundary margins
   - **Large ($2.5\times$):** Radiating spiculation & architectural distortion
   - **Context ($4.0\times$):** Background parenchymal density
   - **Calcification Patch:** High-frequency spatial filter
   - **Full-Mammogram Canvas ($224\times 224$):** Global context
6. **MultiScaleViTV8 Classification Network (Arm F):** Fuses concentric RoIs across candidates via continuous softmax attention with global context into a $4,608$-dimensional representation for Benign vs Malignant classification.
7. **Authentic Vision Transformer Attention Rollout:** Produces interpretable self-attention saliency maps across all 12 transformer layers.

---

## 🏗️ End-to-End System Pipeline Flow

```
CBIS-DDSM Mammogram (FFDM)
            ↓
  Adaptive Wiener Filtering (5x5)
            ↓
  CLAHE + Normalization [0, 1]
            ↓
  V6 U-KAN Segmentation (512x512 + 768x768)
            ↓
  V8 Proposal Generation (Deterministic Top K=3)
            ↓
  Multi-Scale RoI Extraction (4 Concentric Scales + Calc + Full)
            ↓
  V8 Candidate-Fusion Vision Transformer (Arm F, 4608-d)
            ↓
  AI-Assisted Benign / Malignant Research Prediction + Attention Rollout
```

---

## 🔒 Final Locked Test Evaluation (V8 Primary Benchmark)

Evaluated once on the **quarantined held-out test cohort** ($525$ scans / $235$ unique patients) with **zero patient leakage** and a fixed threshold ($\tau=0.50$):

| Metric | Locked Test Value | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Accuracy** | **64.57%** (339 / 525) | [59.06% – 69.79%] |
| **Macro-F1 Score** | **64.18%** | [58.80% – 69.01%] |
| **Binary F1 (Malignant)** | **60.43%** | [54.03% – 66.09%] |
| **ROC-AUC** | **0.6947** | [0.6309 – 0.7474] |
| **PR-AUC** | **0.5857** | [0.5059 – 0.6723] |
| **Sensitivity (Recall)** | **66.05%** (142 / 215) | [58.85% – 72.93%] |
| **Specificity** | **63.55%** (197 / 310) | [54.58% – 71.56%] |
| **Precision (PPV)** | **55.69%** (142 / 255) | [47.88% – 63.27%] |
| **Brier Score** | **0.3143** | — |
| **Expected Calibration Error** | **29.87%** | — |

### Locked Test Confusion Matrix ($N=525$)
- **True Negatives (TN):** $197$
- **False Positives (FP):** $113$
- **False Negatives (FN):** $73$
- **True Positives (TP):** $142$

---

## 📊 Comprehensive Experimental Comparison (V5 → V6 → V8 → V9)

| Model | Accuracy (%) | Macro-F1 (%) | ROC-AUC | PR-AUC | Sensitivity (%) | Specificity (%) | Role in Research Record |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **V5 (Arm I)** | 65.02 | 64.73 | 0.6988 | 0.5916 | 68.37 | 62.70 | Baseline Concentric RoI Extraction |
| **V6 (Arm C)** | 64.45 | 64.11 | 0.7087 | 0.5975 | 66.98 | 62.70 | High-Res U-KAN ($768\times768$) + ViT Head |
| **V8 (Arm F)** | **64.57** | **64.18** | **0.6947** | **0.5857** | **66.05** | **63.55** | **FINAL REPORTED MODEL (Frozen)** |
| **V9 (Arm G)** | 62.86 | 62.86 | 0.7046 | 0.6129 | 76.28 | 53.55 | Documented Experimental Extension |

*Note: All values represent locked held-out test evaluations under strict zero-tuning governance.*

---

## 🎓 Formal Thesis & Academic Selection Statement

> "Among the evaluated experimental configurations, V8 was designated as the final reported model based on its locked patient-independent test evaluation and overall classification balance. On the primary test cohort of 525 scans from 235 patients, V8 achieved 64.57% accuracy, 64.18% Macro-F1, 0.6947 ROC-AUC, 0.5857 PR-AUC, 66.05% sensitivity, and 63.55% specificity.
>
> A subsequent V9 experimental extension achieved higher malignant sensitivity of 76.28% and PR-AUC of 0.6129, but its overall accuracy decreased to 62.86% and specificity decreased to 53.55%. The V9 implementation and final test evaluation were independently audited and no implementation error was identified. Therefore, V9 is retained as a documented experimental extension rather than replacing V8 as the final reported configuration."

---

## 🔐 Model Checkpoints & Cryptographic Verification

All frozen models are cryptographically validated to guarantee reproducibility and prevent weight drift:

- **Final Classification Checkpoint:** `models/final_v8/vit_v8_final.pth`  
  - Source: `models/v8/classification/vit_v8_phase2_arm_f_best.pth`  
  - **MD5:** `882313ce67adca01bffa699a180719c4`  
  - **SHA256:** `7aa49ef7ff2f94e72cc2f27361af89ba0cd04dae00b333f41d8c50a37a1c64c7`
- **Final Segmentation Checkpoint:** `models/final_v8/ukan_v6_final.pth`  
  - Source: `models/v6/segmentation/final/ukan_v6_arm_g_best.pth`  
  - **MD5:** `68cfc38cfbddd93d6e4a6e692fffbb34`  
  - **SHA256:** `f2cd9b1208f4b77416640201143539d6d08b1ff03cb175e1502eb73b8431769d`

---

## 🚀 Installation & Research Dashboard Launch

### 1. Environment Setup
```bash
# Navigate to the project directory
cd AI_CANCER_DETECTION

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run Smoke Test & Checkpoint Verification
```bash
python test_app_smoke.py
```

### 3. Launch the Streamlit Research Dashboard
```bash
streamlit run app/app.py
```

Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
AI_CANCER_DETECTION/
├── app/
│   ├── app.py              # Streamlit interactive multi-page dashboard
│   ├── components.py       # Styling, risk cards, hero headers, flowchart
│   ├── config.py           # Constants, checkpoint paths, MD5s, metrics
│   ├── inference.py        # Checkpoint loader, U-KAN & ViT execution engine
│   ├── preprocessing.py    # Multi-format image loader, Wiener filter, CLAHE
│   └── visualization.py    # Overlays, heatmaps, RoI panels, confusion matrix
├── classification/
│   └── v6/                 # MultiScaleViTV6 model, Top-K proposals, RoI generator
├── explainability/
│   └── attention.py        # Authentic ViT attention rollout extraction
├── models/
│   ├── final_v8/           # Cryptographically locked final V8 checkpoints
│   ├── v8/                 # V8 experimental classification checkpoints
│   └── v9/                 # V9 experimental extension checkpoints
├── results/
│   ├── final_v8/           # Formal final V8 model report, metrics, and audits
│   ├── v8/                 # V8 Phase 1, Phase 2, and Phase 3 reports
│   └── v9/                 # V9 Phase 0-5 and final audit reports
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation and final freeze report
└── test_app_smoke.py       # End-to-end verification smoke test
```

---

## ⚖️ Academic Safety & Ethical Statement

This platform is an academic research prototype. Predictions should never be utilized as autonomous clinical decisions. Clinical diagnosis must always be established by certified medical specialists.
