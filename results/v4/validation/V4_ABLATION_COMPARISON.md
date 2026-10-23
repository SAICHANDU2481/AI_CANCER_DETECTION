# V4-A Controlled Validation Ablation Comparison

## 1. Executive Summary & Core Findings

This document presents the complete empirical results from the **V4-A Controlled Validation Ablation Suite** across the $541$ validation scans ($235$ patients).

All experiments were conducted with strict data governance:
- **100% Test Set Quarantine:** The $525$-scan / $235$-patient test set remained locked and untouched. Zero test evaluation or test metric calculation was performed.
- **Controlled Validation Comparison:** Model architectures and training strategies were evaluated exclusively on the validation cohort using **Macro-F1** as the primary selection criterion, supported by **ROC-AUC**, **PR-AUC**, **Accuracy**, **Malignant Recall**, and **Specificity**.

---

## 2. Complete Validation Ablation Summary Table (Arms A – H)

| Arm | Configuration | Scales Used | Quality Vector | Dynamic Gating ($\tau=0.40$) | Stage 1 Best Epoch | Stage 2 Best Epoch | Val Acc (%) | Val Macro-F1 (%) | Val Binary-F1 (%) | Val ROC-AUC | Val PR-AUC | Malignant Recall (%) | Specificity (%) | Precision (%) | Confusion Matrix (TP, TN, FP, FN) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | **V3-B Locked Reference** | Local + Med + Lrg | No | No | Ep 7 | N/A | 69.32 | 66.65 | 57.22 | 0.6933 | 0.5814 | 52.61 | 80.00 | 62.71 | TP=111, TN=264, FP=66, FN=100 |
| **B** | **Local ROI Only** | Local | No | No | Ep 2 | Ep 2 | 64.51 | 63.38 | 55.60 | 0.6826 | 0.5653 | 63.51 | 58.48 | 49.45 | TP=134, TN=193, FP=137, FN=77 |
| **C** | **Local + Medium** | Local + Med | No | No | Ep 9 | Ep 4 | 67.65 | 65.86 | 56.60 | 0.7165 | 0.5994 | 56.87 | 71.82 | 56.34 | TP=120, TN=237, FP=93, FN=91 |
| **D** | **Local + Med + Large** | Local + Med + Lrg | No | No | Ep 5 | Ep 7 | 67.47 | 67.10 | 56.95 | 0.7282 | 0.5536 | 60.19 | 67.27 | 54.04 | TP=127, TN=222, FP=108, FN=84 |
| **E / F** | **Multi-Scale Visual (WINNER)** | Local + Med + Lrg + Ctx | No | No | **Ep 5** | Ep 3 | **71.90** | **70.27** | **63.29** | **0.7612** | **0.6506** | 62.09 | **78.18** | **64.53** | **TP=131, TN=258, FP=72, FN=80** |
| **G** | **Multi-Scale + 16-D Quality Vec** | Local + Med + Lrg + Ctx | **Yes (16-D)** | No | Ep 3 | **Ep 1** | 66.91 | 66.88 | **65.90** | **0.7568** | **0.6573** | **81.99** | 57.27 | 55.10 | TP=173, TN=189, FP=141, FN=38 |
| **H** | **Quality-Gated Routing** | Local + Med + Lrg + Ctx | **Yes (16-D)** | **Yes ($\tau=0.40$)** | **Ep 6** | Ep 15 | 67.65 | 66.20 | 59.21 | 0.7082 | 0.5831 | 60.19 | 72.42 | 58.26 | TP=127, TN=239, FP=91, FN=84 |

---

## 3. In-Depth Scientific Comparisons & Ablation Analysis

### 3.1 Context Progression (Arms B $\rightarrow$ C $\rightarrow$ D $\rightarrow$ E)
- **Local ROI Only (Arm B):** Validation Macro-F1 = $63.38\%$, ROC-AUC = $0.6826$. Without tissue context, the network struggles with boundary ambiguity and calcification clusters.
- **Local + Medium Context (Arm C):** Macro-F1 rises to $65.86\%$ ($+2.48\%$), ROC-AUC rises to $0.7165$ ($+0.0339$).
- **Local + Medium + Large Context (Arm D):** Macro-F1 rises to $67.10\%$ ($+1.24\%$), ROC-AUC rises to $0.7282$ ($+0.0117$).
- **Local + Medium + Large + Surrounding Context (Arm E / F):** Macro-F1 surges to **$70.27\%$** ($+3.17\%$ over Arm D, **$+3.62\%$ over V3-B**), Accuracy reaches **$71.90\%$**, and ROC-AUC peaks at **$0.7612$** ($+0.0679$ over V3-B).
- **Scientific Takeaway:** Multi-scale spatial hierarchy is the primary performance driver. Mammographic lesion interpretation strongly benefits from whole-breast parenchymal reference context.

---

### 3.2 Comparison F vs. G: Impact of 16-Dimensional Quality Vector
The explicit research question was whether the 16-dimensional ROI quality descriptor adds useful signal beyond multi-scale visual tokens.

| Metric | Arm F (Pure Multi-Scale Visual) | Arm G (Multi-Scale + 16-Dim Quality Vector) | $\Delta$ (G vs F) | Scientific Interpretation |
|---|:---:|:---:|:---:|---|
| **Validation Macro-F1** | **70.27%** | 66.88% | $-3.39\%$ | Pure visual fusion achieves higher balanced F1 |
| **Validation Accuracy** | **71.90%** | 66.91% | $-4.99\%$ | Visual fusion has fewer false alarms |
| **Validation ROC-AUC** | **0.7612** | 0.7568 | $-0.0044$ | Comparable ranking discrimination |
| **Validation PR-AUC** | 0.6506 | **0.6573** | $+0.0067$ | Slightly higher precision-recall area with quality vector |
| **Malignant Recall** | 62.09% ($131/211$) | **81.99%** ($173/211$) | **$+19.90\%$** | **Massive cancer sensitivity boost ($+42$ detected cancers)** |
| **Specificity** | **78.18%** ($258/330$) | 57.27% ($189/330$) | $-20.91\%$ | Quality features shift prior toward conservative cancer detection |
| **Malignant FN** | 80 cases | **38 cases** | **$-42$ FN (52.5% FN reduction)** | Crucial clinical benefit for cancer screening |

**Core Conclusion on F vs. G:**
1. If the optimization target is pure **balanced Macro-F1 / Accuracy**, **Arm F (Multi-Scale Visual)** is superior ($70.27\%$ vs $66.88\%$).
2. If the clinical target is **Malignant Sensitivity / FN reduction**, **Arm G (Quality-Aware ViT)** captures **$81.99\%$** of all validation malignancies, slashing false negatives from $80$ down to $38$.

---

### 3.3 Comparison G vs. H: Impact of Quality-Gated Dynamic Routing
Arm H routes cases dynamically:
- When $S_{\text{quality}} \ge 0.40 \rightarrow$ Path A (Full 4-scale fusion + quality vector)
- When $S_{\text{quality}} < 0.40 \rightarrow$ Path B (Expanded ROI + Context + quality vector)

- **Validation Accuracy:** $67.65\%$ (Stage 1 Ep 6)
- **Validation Macro-F1:** $66.20\%$ (Stage 1 Ep 6)
- **Validation Specificity:** $72.42\%$ ($239/330$) vs $57.27\%$ in Arm G ($+15.15\%$)
- **Validation Malignant Recall:** $60.19\%$ ($127/211$)
- **Takeaway:** Gated dynamic routing restores benign specificity when segmentation is unreliable, but does not surpass the holistic 4-scale visual representation of Arm E/F.

---

## 4. ROI Generation & Quality Distribution Statistics

Analysis across all $541$ validation mammograms:

### 4.1 Proposal Tier Rates
- **Primary Machine ROI Rate:** $328 / 541$ ($60.63\%$)
- **Expanded ROI Rate:** $36 / 541$ ($6.65\%$)
- **Context ROI Rate:** $0 / 541$ ($0.00\%$)
- **Deterministic Fallback Rate:** $177 / 541$ ($32.72\%$)

### 4.2 Quality Metric Distributions

| Metric | Mean $\pm$ Std | Median | Min | 25th % | 75th % | Max |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ROI Quality Score ($S_{\text{quality}}$)** | $0.5332 \pm 0.3799$ | $0.7551$ | $0.0000$ | $0.0000$ | $0.8237$ | $1.0000$ |
| **Segmentation Dice** | $0.1915 \pm 0.2958$ | $0.0000$ | $0.0000$ | $0.0000$ | $0.4411$ | $0.9281$ |

---

## 5. Winning Model Checkpoint Identification

- **Selected Best Validation Architecture:** **Arm E / F (Multi-Scale Visual Fusion: Local + Medium + Large + Context)**
- **Selected Checkpoint:** `models/v4/vit_v4_best.pth`
- **MD5 Checksum:** `d016fdc756758658e03d43f02fd4482c`
- **Optimal Training Phase:** Stage 1 Warmup (Epoch 5)
- **Primary Selection Metric:** **Macro-F1 = 70.27%** (Highest across all arms)
- **Secondary Metrics:** **ROC-AUC = 0.7612**, **Accuracy = 71.90%**, **Specificity = 78.18%**, **PR-AUC = 0.6506**
