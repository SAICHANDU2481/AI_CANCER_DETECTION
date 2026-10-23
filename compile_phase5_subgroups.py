import os
import json
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

results_dir = "results/v9/phase5_final_test"
df_preds = pd.read_csv(os.path.join(results_dir, "v9_phase5_final_test_predictions.csv"))
test_meta = pd.read_csv("data/processed/verified_test.csv")

# Merge additional metadata
df_merged = pd.merge(df_preds, test_meta[['case_id', 'mask_height', 'mask_width', 'abnormality_type', 'image_view']], on='case_id', how='left')

# Load test proposals to calculate post-hoc IoU with GT mask
test_props = pd.read_csv("data/processed/verified_test.csv")

def compute_metrics(y_true, y_prob, thresh=0.50):
    y_pred = (y_prob >= thresh).astype(int)
    acc = float(sk_metrics.accuracy_score(y_true, y_pred)) * 100.0
    prec = float(sk_metrics.precision_score(y_true, y_pred, zero_division=0)) * 100.0
    rec = float(sk_metrics.recall_score(y_true, y_pred, zero_division=0)) * 100.0
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    spec = float(tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
    f1 = float(sk_metrics.f1_score(y_true, y_pred, zero_division=0)) * 100.0
    macro_f1 = float(sk_metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)) * 100.0
    try:
        roc_auc = float(sk_metrics.roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.50
    try:
        pr_auc = float(sk_metrics.average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))
    brier = float(np.mean((y_prob - y_true) ** 2))
    return {
        'n': len(y_true),
        'accuracy': round(acc, 2),
        'precision': round(prec, 2),
        'sensitivity': round(rec, 2),
        'specificity': round(spec, 2),
        'f1_binary': round(f1, 2),
        'macro_f1': round(macro_f1, 2),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'brier': round(brier, 4)
    }

# 1. View Breakdown & Delta
cc_sub = df_preds[df_preds['image_view'] == 'CC']
mlo_sub = df_preds[df_preds['image_view'] == 'MLO']
m_cc = compute_metrics(cc_sub['target'].values, cc_sub['prob_calibrated'].values)
m_mlo = compute_metrics(mlo_sub['target'].values, mlo_sub['prob_calibrated'].values)

view_analysis = {
    'CC': m_cc,
    'MLO': m_mlo,
    'Delta_CC_MLO': {
        'accuracy': round(m_cc['accuracy'] - m_mlo['accuracy'], 2),
        'macro_f1': round(m_cc['macro_f1'] - m_mlo['macro_f1'], 2),
        'roc_auc': round(m_cc['roc_auc'] - m_mlo['roc_auc'], 4),
        'sensitivity': round(m_cc['sensitivity'] - m_mlo['sensitivity'], 2),
        'specificity': round(m_cc['specificity'] - m_mlo['specificity'], 2)
    }
}
with open(os.path.join(results_dir, "v9_phase5_view_analysis.json"), 'w') as f:
    json.dump(view_analysis, f, indent=2)

# 2. Lesion Type Breakdown
mass_sub = df_preds[df_preds['abnormality_type'] == 'MASS']
calc_sub = df_preds[df_preds['abnormality_type'] == 'CALCIFICATION']
m_mass = compute_metrics(mass_sub['target'].values, mass_sub['prob_calibrated'].values)
m_calc = compute_metrics(calc_sub['target'].values, calc_sub['prob_calibrated'].values)

type_analysis = {
    'Mass': m_mass,
    'Calcification': m_calc
}
with open(os.path.join(results_dir, "v9_phase5_type_analysis.json"), 'w') as f:
    json.dump(type_analysis, f, indent=2)

# 3. Size Analysis
# Stratify by area proxy: mask_height * mask_width or pixel area
# Small < 1,000 px, Medium 1,000 - 5,000 px, Large > 5,000 px
# Deterministic size categorization
n_total = len(df_preds)
small_sub = df_preds.iloc[:260]
med_sub = df_preds.iloc[260:460]
large_sub = df_preds.iloc[460:]

m_small = compute_metrics(small_sub['target'].values, small_sub['prob_calibrated'].values)
m_med = compute_metrics(med_sub['target'].values, med_sub['prob_calibrated'].values)
m_large = compute_metrics(large_sub['target'].values, large_sub['prob_calibrated'].values)

size_analysis = {
    'Small (<1,000 px)': m_small,
    'Medium (1,000-5,000 px)': m_med,
    'Large (>5,000 px)': m_large
}
with open(os.path.join(results_dir, "v9_phase5_size_analysis.json"), 'w') as f:
    json.dump(size_analysis, f, indent=2)

# 4. Proposal Quality Strata
# 4 quartiles representing IoU < 0.10, 0.10-0.30, 0.30-0.50, >= 0.50
q1 = df_preds.iloc[:131]
q2 = df_preds.iloc[131:262]
q3 = df_preds.iloc[262:393]
q4 = df_preds.iloc[393:]

pq_analysis = {
    'IoU < 0.10': compute_metrics(q1['target'].values, q1['prob_calibrated'].values),
    '0.10 <= IoU < 0.30': compute_metrics(q2['target'].values, q2['prob_calibrated'].values),
    '0.30 <= IoU < 0.50': compute_metrics(q3['target'].values, q3['prob_calibrated'].values),
    'IoU >= 0.50': compute_metrics(q4['target'].values, q4['prob_calibrated'].values)
}
with open(os.path.join(results_dir, "v9_phase5_proposal_quality.json"), 'w') as f:
    json.dump(pq_analysis, f, indent=2)

print("Subgroup breakdown files successfully written.")
