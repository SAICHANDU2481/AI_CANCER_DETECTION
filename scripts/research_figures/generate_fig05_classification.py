"""
================================================================================
Figure 5: Classification Performance of the Final Frozen V8 Model (Arm F)
================================================================================
Generates publication-quality 9-panel comprehensive evaluation using genuine
locked test results (525 scans / 235 quarantined patients):
- Panel A: Absolute Count Confusion Matrix (TN=197, FP=113, FN=73, TP=142)
- Panel B: Row-Normalized Confusion Matrix (True Class Sensitivities)
- Panel C: Empirical ROC Curve (Verified AUC = 0.6947 [95% CI: 0.6309 - 0.7474])
- Panel D: Empirical Precision-Recall Curve (Verified AP/PR-AUC = 0.5857)
- Panel E: Primary Diagnostic Metrics Bar Chart with 95% Bootstrap CIs
- Panel F: Class-Disaggregated Performance Comparison (Benign vs Malignant)
- Panel G: Predicted Posterior Confidence Distributions (Target 0 vs Target 1)
- Panel H: 10-Bin Reliability Calibration Diagram (ECE = 29.87%, Brier = 0.3143)
- Panel I: Calibration Error Decomposition & Bin Sample Histogram
- Exports: PDF vector, SVG vector, 600 DPI PNG
================================================================================
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)

def generate_fig05_classification():
    pred_path = 'results/v8/phase3/v8_arm_f_test_predictions.csv'
    calib_path = 'results/v8/phase3/v8_arm_f_test_calibration.json'
    
    if not os.path.exists(pred_path) or not os.path.exists(calib_path):
        raise FileNotFoundError("Missing locked V8 test predictions or calibration json.")
        
    df = pd.read_csv(pred_path)
    with open(calib_path, 'r') as f:
        calib_data = json.load(f)
        
    y_true = df['target'].values
    y_pred = df['predicted_class'].values
    y_prob = df['prob_malignant'].values
    
    # Calculate exact confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Styling configuration
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 8.5
    plt.rcParams['axes.linewidth'] = 0.9
    
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 14.5), facecolor='white', dpi=300)
    plt.subplots_adjust(wspace=0.32, hspace=0.38, left=0.06, right=0.96, top=0.95, bottom=0.05)
    
    # Color palette
    c_benign = '#2B7A78'
    c_malignant = '#D9534F'
    c_navy = '#1B2A4A'
    c_accent = '#3498DB'
    
    # -------------------------------------------------------------------------
    # Panel A: Raw Count Confusion Matrix
    # -------------------------------------------------------------------------
    ax_a = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_a,
                annot_kws={'size': 13, 'weight': 'bold'},
                xticklabels=['Pred Benign', 'Pred Malignant'],
                yticklabels=['Actual Benign', 'Actual Malignant'])
    ax_a.set_title('(a) Confusion Matrix (N=525)', fontsize=10, fontweight='bold', pad=10)
    ax_a.text(0.5, -0.20, f"TN={tn} | FP={fp} | FN={fn} | TP={tp}",
              transform=ax_a.transAxes, ha='center', fontsize=8.5, fontweight='bold', color=c_navy)

    # -------------------------------------------------------------------------
    # Panel B: Row-Normalized Confusion Matrix
    # -------------------------------------------------------------------------
    ax_b = axes[0, 1]
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', cbar=False, ax=ax_b,
                annot_kws={'size': 12, 'weight': 'bold'},
                xticklabels=['Pred Benign', 'Pred Malignant'],
                yticklabels=['Actual Benign', 'Actual Malignant'])
    ax_b.set_title('(b) Normalized Confusion Matrix (%)', fontsize=10, fontweight='bold', pad=10)
    ax_b.text(0.5, -0.20, f"Specificity: {cm_norm[0,0]:.1f}% | Sensitivity: {cm_norm[1,1]:.1f}%",
              transform=ax_b.transAxes, ha='center', fontsize=8.5, fontweight='bold', color='#137333')

    # -------------------------------------------------------------------------
    # Panel C: Empirical ROC Curve (AUC = 0.6947)
    # -------------------------------------------------------------------------
    ax_c = axes[0, 2]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    
    ax_c.plot(fpr, tpr, color='#1B2A4A', lw=2.2,
              label=f"V8 Arm F (AUC = {auc_val:.4f})\n95% CI: [0.6309 - 0.7474]")
    ax_c.plot([0, 1], [0, 1], color='#7F8C8D', lw=1.2, ls='--', label='Random Chance (AUC = 0.50)')
    # Operating point (Threshold = 0.50)
    op_fpr = fp / (tn + fp)
    op_tpr = tp / (tp + fn)
    ax_c.scatter([op_fpr], [op_tpr], color='#E74C3C', s=60, zorder=5,
                 label=f"Locked Operating Pt (tau=0.50)\nFPR={op_fpr:.3f}, TPR={op_tpr:.3f}")
    
    ax_c.set_title('(c) Receiver Operating Characteristic (ROC)', fontsize=10, fontweight='bold', pad=10)
    ax_c.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=9)
    ax_c.set_ylabel('True Positive Rate (Sensitivity)', fontsize=9)
    ax_c.set_xlim(-0.02, 1.02)
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.grid(True, ls=':', alpha=0.5)
    ax_c.legend(loc='lower right', frameon=True, fontsize=7.8)

    # -------------------------------------------------------------------------
    # Panel D: Empirical Precision-Recall Curve (PR-AUC = 0.5857)
    # -------------------------------------------------------------------------
    ax_d = axes[1, 0]
    prec_pts, rec_pts, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = average_precision_score(y_true, y_prob)
    prevalence = np.mean(y_true)
    
    ax_d.plot(rec_pts, prec_pts, color='#8E44AD', lw=2.2,
              label=f"V8 Arm F (PR-AUC = {pr_auc_val:.4f})\n95% CI: [0.5059 - 0.6723]")
    ax_d.axhline(prevalence, color='#7F8C8D', lw=1.2, ls='--',
                 label=f"Malignant Prevalence ({prevalence*100:.1f}%)")
    # Operating point
    op_prec = tp / (tp + fp)
    op_rec = tp / (tp + fn)
    ax_d.scatter([op_rec], [op_prec], color='#E74C3C', s=60, zorder=5,
                 label=f"Operating Pt (Prec={op_prec:.3f}, Rec={op_rec:.3f})")
    
    ax_d.set_title('(d) Precision-Recall (PR) Curve', fontsize=10, fontweight='bold', pad=10)
    ax_d.set_xlabel('Recall (Sensitivity)', fontsize=9)
    ax_d.set_ylabel('Precision (Positive Predictive Value)', fontsize=9)
    ax_d.set_xlim(-0.02, 1.02)
    ax_d.set_ylim(-0.02, 1.02)
    ax_d.grid(True, ls=':', alpha=0.5)
    ax_d.legend(loc='upper right', frameon=True, fontsize=7.8)

    # -------------------------------------------------------------------------
    # Panel E: Primary Performance Metrics with 95% Bootstrap CIs
    # -------------------------------------------------------------------------
    ax_e = axes[1, 1]
    metrics_names = ['Accuracy', 'Macro-F1', 'Binary F1', 'Sensitivity', 'Specificity', 'Precision']
    metrics_vals = [64.57, 64.18, 60.43, 66.05, 63.55, 55.69]
    ci_err_low = [64.57 - 59.06, 64.18 - 58.80, 60.43 - 54.03, 66.05 - 58.85, 63.55 - 54.58, 55.69 - 47.88]
    ci_err_high = [69.79 - 64.57, 69.01 - 64.18, 66.09 - 60.43, 72.93 - 66.05, 71.56 - 63.55, 63.27 - 55.69]
    
    y_pos = np.arange(len(metrics_names))
    colors_e = ['#2C3E50', '#2980B9', '#16A085', '#27AE60', '#D35400', '#8E44AD']
    
    ax_e.barh(y_pos, metrics_vals, xerr=[ci_err_low, ci_err_high], color=colors_e,
              edgecolor='#1B2A4A', height=0.58, capsize=4.0, alpha=0.9)
    ax_e.set_yticks(y_pos)
    ax_e.set_yticklabels(metrics_names, fontsize=8.8, fontweight='bold')
    ax_e.set_xlim(35, 80)
    ax_e.set_xlabel('Score (%) [with 95% Bootstrap CI]', fontsize=9)
    ax_e.set_title('(e) Locked Evaluation Metrics (N=525)', fontsize=10, fontweight='bold', pad=10)
    ax_e.grid(axis='x', ls=':', alpha=0.5)
    
    for i, v in enumerate(metrics_vals):
        ax_e.text(v + 1.2, i, f"{v:.2f}%", va='center', fontsize=8.2, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel F: Class-Disaggregated Performance
    # -------------------------------------------------------------------------
    ax_f = axes[1, 2]
    # Class metrics
    prec_b = tn / (tn + fn) * 100
    rec_b = tn / (tn + fp) * 100
    f1_b = 2 * (prec_b * rec_b) / (prec_b + rec_b)
    
    prec_m = tp / (tp + fp) * 100
    rec_m = tp / (tp + fn) * 100
    f1_m = 2 * (prec_m * rec_m) / (prec_m + rec_m)
    
    categories = ['Precision', 'Recall', 'F1-Score']
    x_c = np.arange(len(categories))
    w = 0.35
    
    b_vals = [prec_b, rec_b, f1_b]
    m_vals = [prec_m, rec_m, f1_m]
    
    bars_fb = ax_f.bar(x_c - w/2, b_vals, w, label='Benign (N=310)', color=c_benign, edgecolor='#1B2A4A')
    bars_fm = ax_f.bar(x_c + w/2, m_vals, w, label='Malignant (N=215)', color=c_malignant, edgecolor='#1B2A4A')
    
    ax_f.set_title('(f) Class-Disaggregated Performance', fontsize=10, fontweight='bold', pad=10)
    ax_f.set_xticks(x_c)
    ax_f.set_xticklabels(categories, fontsize=9, fontweight='semibold')
    ax_f.set_ylabel('Metric (%)', fontsize=9)
    ax_f.set_ylim(0, 95)
    ax_f.grid(axis='y', ls=':', alpha=0.5)
    ax_f.legend(loc='lower right', frameon=True, fontsize=8.2)
    
    for bar in bars_fb:
        h = bar.get_height()
        ax_f.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%",
                  ha='center', va='bottom', fontsize=7.8, fontweight='bold')
    for bar in bars_fm:
        h = bar.get_height()
        ax_f.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%",
                  ha='center', va='bottom', fontsize=7.8, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel G: Prediction Confidence Distribution (Separated by True Class)
    # -------------------------------------------------------------------------
    ax_g = axes[2, 0]
    prob_benign_cases = df[df['target'] == 0]['prob_malignant'].values
    prob_mal_cases = df[df['target'] == 1]['prob_malignant'].values
    
    bins_g = np.linspace(0, 1, 21)
    ax_g.hist(prob_benign_cases, bins=bins_g, alpha=0.6, color=c_benign, label='Ground Truth: Benign', density=True)
    ax_g.hist(prob_mal_cases, bins=bins_g, alpha=0.6, color=c_malignant, label='Ground Truth: Malignant', density=True)
    ax_g.axvline(0.50, color='#2C3E50', lw=1.6, ls='--', label='Decision Threshold (0.50)')
    
    ax_g.set_title('(g) Posterior Confidence Distribution', fontsize=10, fontweight='bold', pad=10)
    ax_g.set_xlabel('Predicted Probability P(Malignant)', fontsize=9)
    ax_g.set_ylabel('Density', fontsize=9)
    ax_g.set_xlim(0, 1.0)
    ax_g.grid(True, ls=':', alpha=0.5)
    ax_g.legend(loc='upper right', frameon=True, fontsize=7.8)

    # -------------------------------------------------------------------------
    # Panel H: Reliability Calibration Diagram (10 Bins)
    # -------------------------------------------------------------------------
    ax_h = axes[2, 1]
    rel_bins = calib_data['reliability_bins']
    
    mean_preds = [b['mean_predicted_prob'] for b in rel_bins]
    emp_accs = [b['empirical_accuracy'] for b in rel_bins]
    sample_counts = [b['sample_count'] for b in rel_bins]
    
    ax_h.plot([0, 1], [0, 1], color='#7F8C8D', lw=1.2, ls='--', label='Perfect Calibration')
    ax_h.plot(mean_preds, emp_accs, marker='o', markersize=6, lw=2.0, color='#D35400',
              label=f"V8 Arm F (ECE = {calib_data['ece_percent']:.2f}%)")
    
    ax_h.set_title('(h) Reliability Calibration Curve', fontsize=10, fontweight='bold', pad=10)
    ax_h.set_xlabel('Mean Predicted Probability', fontsize=9)
    ax_h.set_ylabel('Empirical Accuracy (Fraction of Positives)', fontsize=9)
    ax_h.set_xlim(-0.02, 1.02)
    ax_h.set_ylim(-0.02, 1.02)
    ax_h.grid(True, ls=':', alpha=0.5)
    ax_h.legend(loc='upper left', frameon=True, fontsize=7.8)
    
    # Inset histogram for bin sample counts
    ax_h_inset = ax_h.inset_axes([0.60, 0.12, 0.36, 0.30])
    ax_h_inset.bar(range(len(sample_counts)), sample_counts, color='#F39C12', edgecolor='#2C3E50', width=0.7)
    ax_h_inset.set_title('Bin Sample Counts', fontsize=7.2, fontweight='bold')
    ax_h_inset.tick_params(labelsize=6.5)
    ax_h_inset.set_ylabel('N', fontsize=6.8)

    # -------------------------------------------------------------------------
    # Panel I: Calibration Error Summary & Governance Protocol
    # -------------------------------------------------------------------------
    ax_i = axes[2, 2]
    ax_i.axis('off')
    
    brier = calib_data['brier_score']
    ece = calib_data['ece_percent']
    
    summary_text = (
        "PROBABILITY CALIBRATION SUMMARY\n"
        "====================================\n\n"
        f"• Final Brier Score:       {brier:.4f}\n"
        f"• Expected Calibration Error: {ece:.2f}%\n"
        f"• Optimal Bin Count:       10 Uniform Bins\n"
        f"• Calibration Cohort:      525 Locked Test Scans\n"
        f"• Quarantined Patients:    235 Patients (0 Leakage)\n\n"
        "SCIENTIFIC GOVERNANCE NOTE:\n"
        "------------------------------------\n"
        "1. Probability calibration evaluates predictive confidence\n"
        "   alignment, not classification discrimination.\n"
        "2. ECE value is strictly dependent upon the 10-bin uniform\n"
        "   partitioning convention.\n"
        "3. High ECE reflects raw ViT logit overconfidence;\n"
        "   temperature scaling (T=1.21) was explored in V9 but\n"
        "   is NOT applied to frozen V8 to preserve raw audit integrity."
    )
    
    ax_i.text(0.04, 0.95, summary_text, transform=ax_i.transAxes,
              va='top', ha='left', fontsize=8.2, family='monospace',
              bbox=dict(boxstyle='round,pad=0.6', facecolor='#F4F6F9', edgecolor='#90A4AE', lw=1.2))
    ax_i.set_title('(i) Calibration Audit & Scientific Protocol', fontsize=10, fontweight='bold', pad=10)

    out_dir = 'results/research_paper_figures/figure_05_classification'
    os.makedirs(out_dir, exist_ok=True)
    
    png_path = os.path.join(out_dir, 'fig05_classification_evaluation.png')
    pdf_path = os.path.join(out_dir, 'fig05_classification_evaluation.pdf')
    svg_path = os.path.join(out_dir, 'fig05_classification_evaluation.svg')
    
    print(f"Exporting Figure 5 to {out_dir}...")
    fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[OK] Figure 5 successfully generated: PNG (600 DPI), PDF, SVG.")

if __name__ == '__main__':
    generate_fig05_classification()
