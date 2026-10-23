# V6 Classification Validation Comparison

## Controlled Multi-Arm Validation Suite (541 Validation Scans)

| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | Malignant Recall | Specificity | Mass Macro-F1 | Calc Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5 Arm-I Reference (Locked)** | 72.27% | 71.67% | 0.7928 | 0.6556 | 73.93% | 71.21% | 68.43% | 73.28% |
| **V6 Arm-B (Frozen ViT)** | 70.98% | 70.64% | 0.7780 | 0.6458 | 77.25% | 66.97% | 69.94% | 69.79% |
| **V6 Arm-C (Last 2 Blocks Tuned - Primary)** | **71.90%** | **71.05%** | **0.7787** | **0.6496** | **70.14%** | **73.03%** | **70.04%** | **69.84%** |
| **V6 Arm-D (Last 4 Blocks Tuned)** | *N/A (Omitted due to hardware limit)* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* |

*Evaluated strictly on the 541-scan validation cohort under complete test data quarantine.*
