"""
================================================================================
Figure 7: Comprehensive Diagnostic Error Analysis of the Frozen V8 Model
================================================================================
Generates publication-quality scientific error characterization on locked test
results (N = 525 scans, 235 quarantined patients):
- Panel A: Primary Error Taxonomy: False Positives (113) vs False Negatives (73)
- Panel B: Diagnostic Error Breakdown by Lesion Abnormality Type (Mass vs Calc)
- Panel C: Diagnostic Error Breakdown by Mammographic Projection View (CC vs MLO)
- Panel D: Classification Error Rate Stratified by U-KAN Localization Quality (IoU)
- Panel E: Model Prediction Confidence Distributions for Correct vs Error Scans
- Panel F: Multi-Subgroup Accuracy, Sensitivity & Specificity Profile
- Panel G: Historical Evolution of False Positives & False Negatives (V5 -> V9)
- Panel H: Methodological Governance and Subgroup Sample Size Audit
- Exports: PDF vector, SVG vector, 600 DPI PNG
================================================================================
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_fig07_error_analysis():
    pred_path = 'results/v8/phase3/v8_arm_f_test_predictions.csv'
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing test predictions at {pred_path}")
        
    df = pd.read_csv(pred_path)
    print(f"Loaded {len(df)} locked test predictions.")
    
    # Classify error types:
    # TP: target=1, pred=1
    # TN: target=0, pred=0
    # FP: target=0, pred=1 (Type I error)
    # FN: target=1, pred=0 (Type II error)
    df['error_type'] = 'Correct'
    df.loc[(df['target'] == 0) & (df['predicted_class'] == 1), 'error_type'] = 'False Positive (FP)'
    df.loc[(df['target'] == 1) & (df['predicted_class'] == 0), 'error_type'] = 'False Negative (FN)'
    
    total_errors = len(df[df['error_type'] != 'Correct']) # 186
    fp_cnt = len(df[df['error_type'] == 'False Positive (FP)']) # 113
    fn_cnt = len(df[df['error_type'] == 'False Negative (FN)']) # 73
    
    # Styling configuration
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 8.5
    plt.rcParams['axes.linewidth'] = 0.9
    
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 9.5), facecolor='white', dpi=300)
    plt.subplots_adjust(wspace=0.32, hspace=0.38, left=0.05, right=0.96, top=0.93, bottom=0.07)
    
    # Colors
    c_fp = '#E74C3C'     # Red for False Positive
    c_fn = '#3498DB'     # Blue for False Negative
    c_corr = '#2ECC71'   # Green for Correct
    c_navy = '#1B2A4A'
    
    # -------------------------------------------------------------------------
    # Panel A: FP vs FN Overview
    # -------------------------------------------------------------------------
    ax_a = axes[0, 0]
    bars_a = ax_a.bar(['False Positives\n(Benign as Malignant)', 'False Negatives\n(Malignant as Benign)'],
                      [fp_cnt, fn_cnt], color=[c_fp, c_fn], width=0.55, edgecolor='#1B2A4A', linewidth=1.2)
    ax_a.set_title('(a) Primary Error Taxonomy (N=186)', fontsize=9.8, fontweight='bold', pad=10)
    ax_a.set_ylabel('Error Count', fontsize=9)
    ax_a.set_ylim(0, max(fp_cnt, fn_cnt) * 1.25)
    ax_a.grid(axis='y', ls=':', alpha=0.5)
    
    for bar in bars_a:
        h = bar.get_height()
        pct = (h / total_errors) * 100
        cohort_pct = (h / len(df)) * 100
        ax_a.text(bar.get_x() + bar.get_width()/2., h + 3,
                  f"{h}\n({pct:.1f}% errors)\n[{cohort_pct:.1f}% cohort]",
                  ha='center', va='bottom', fontsize=7.6, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel B: Error Distribution by Abnormality Type
    # -------------------------------------------------------------------------
    ax_b = axes[0, 1]
    # For Mass & Calc: FP, FN
    types = ['MASS', 'CALCIFICATION']
    fp_by_type = [len(df[(df['abnormality_type'] == t) & (df['error_type'] == 'False Positive (FP)')]) for t in types]
    fn_by_type = [len(df[(df['abnormality_type'] == t) & (df['error_type'] == 'False Negative (FN)')]) for t in types]
    
    x = np.arange(len(types))
    w = 0.35
    b_fp = ax_b.bar(x - w/2, fp_by_type, w, label='False Positive', color=c_fp, edgecolor='#1B2A4A')
    b_fn = ax_b.bar(x + w/2, fn_by_type, w, label='False Negative', color=c_fn, edgecolor='#1B2A4A')
    
    ax_b.set_title('(b) Error Counts by Lesion Type', fontsize=9.8, fontweight='bold', pad=10)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(['Mass Lesions\n(N=267)', 'Calcifications\n(N=258)'], fontsize=8.2, fontweight='semibold')
    ax_b.set_ylabel('Error Count', fontsize=9)
    ax_b.set_ylim(0, max(fp_by_type + fn_by_type) * 1.25)
    ax_b.grid(axis='y', ls=':', alpha=0.5)
    ax_b.legend(loc='upper right', frameon=True, fontsize=7.8)
    
    for bar in b_fp:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h}", ha='center', va='bottom', fontsize=7.6, fontweight='bold')
    for bar in b_fn:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h}", ha='center', va='bottom', fontsize=7.6, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel C: Error Distribution by Mammographic View
    # -------------------------------------------------------------------------
    ax_c = axes[0, 2]
    views = ['CC', 'MLO']
    fp_by_view = [len(df[(df['image_view'] == v) & (df['error_type'] == 'False Positive (FP)')]) for v in views]
    fn_by_view = [len(df[(df['image_view'] == v) & (df['error_type'] == 'False Negative (FN)')]) for v in views]
    
    x_v = np.arange(len(views))
    c_fp_b = ax_c.bar(x_v - w/2, fp_by_view, w, label='False Positive', color=c_fp, edgecolor='#1B2A4A')
    c_fn_b = ax_c.bar(x_v + w/2, fn_by_view, w, label='False Negative', color=c_fn, edgecolor='#1B2A4A')
    
    ax_c.set_title('(c) Error Counts by Projection View', fontsize=9.8, fontweight='bold', pad=10)
    ax_c.set_xticks(x_v)
    ax_c.set_xticklabels(['CC View\n(N=250)', 'MLO View\n(N=275)'], fontsize=8.2, fontweight='semibold')
    ax_c.set_ylabel('Error Count', fontsize=9)
    ax_c.set_ylim(0, max(fp_by_view + fn_by_view) * 1.25)
    ax_c.grid(axis='y', ls=':', alpha=0.5)
    ax_c.legend(loc='upper right', frameon=True, fontsize=7.8)
    
    for bar in c_fp_b:
        h = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h}", ha='center', va='bottom', fontsize=7.6, fontweight='bold')
    for bar in c_fn_b:
        h = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h}", ha='center', va='bottom', fontsize=7.6, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel D: Error Rate vs U-KAN Localization Quality (IoU Strata)
    # -------------------------------------------------------------------------
    ax_d = axes[0, 3]
    df['iou_strata'] = pd.cut(df['best_iou'], bins=[-1, 0.01, 0.1, 0.3, 0.5, 1.0],
                              labels=['0.0\n(Miss)', '0.01-0.1\n(Weak)', '0.1-0.3\n(Partial)', '0.3-0.5\n(Good)', '>0.5\n(Strong)'])
    
    iou_grouped = df.groupby('iou_strata', observed=False).agg(
        total=('is_correct', 'count'),
        errors=('is_correct', lambda s: (~s).sum())
    )
    iou_grouped['error_rate'] = (iou_grouped['errors'] / iou_grouped['total']) * 100
    
    x_iou = np.arange(len(iou_grouped))
    bars_d = ax_d.bar(x_iou, iou_grouped['error_rate'], color='#E67E22', edgecolor='#1B2A4A', width=0.58)
    
    ax_d.set_title('(d) Error Rate by Proposal IoU', fontsize=9.8, fontweight='bold', pad=10)
    ax_d.set_xticks(x_iou)
    ax_d.set_xticklabels(iou_grouped.index, fontsize=7.6)
    ax_d.set_ylabel('Error Rate (%)', fontsize=9)
    ax_d.set_ylim(0, 60)
    ax_d.grid(axis='y', ls=':', alpha=0.5)
    
    for bar, (_, row_data) in zip(bars_d, iou_grouped.iterrows()):
        h = bar.get_height()
        ax_d.text(bar.get_x() + bar.get_width()/2., h + 1.5,
                  f"{h:.1f}%\n({int(row_data['errors'])}/{int(row_data['total'])})",
                  ha='center', va='bottom', fontsize=7.2, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel E: Confidence Distribution for Correct vs Error Scans
    # -------------------------------------------------------------------------
    ax_e = axes[1, 0]
    # Model confidence = max(prob_benign, prob_malignant)
    df['confidence'] = df[['prob_benign', 'prob_malignant']].max(axis=1)
    
    conf_correct = df[df['is_correct']]['confidence'].values
    conf_error = df[~df['is_correct']]['confidence'].values
    
    bins_e = np.linspace(0.5, 1.0, 16)
    ax_e.hist(conf_correct, bins=bins_e, alpha=0.55, color=c_corr, label=f"Correct (Mean={conf_correct.mean():.2f})", density=True)
    ax_e.hist(conf_error, bins=bins_e, alpha=0.55, color='#E74C3C', label=f"Error (Mean={conf_error.mean():.2f})", density=True)
    
    ax_e.set_title('(e) Decision Confidence (Correct vs Error)', fontsize=9.8, fontweight='bold', pad=10)
    ax_e.set_xlabel('Max Posterior Likelihood max(P)', fontsize=9)
    ax_e.set_ylabel('Density', fontsize=9)
    ax_e.set_xlim(0.48, 1.02)
    ax_e.grid(True, ls=':', alpha=0.5)
    ax_e.legend(loc='upper right', frameon=True, fontsize=7.8)

    # -------------------------------------------------------------------------
    # Panel F: Subgroup Performance Profile (Acc, Sens, Spec)
    # -------------------------------------------------------------------------
    ax_f = axes[1, 1]
    
    # Compute accuracy across subgroups
    subgroups = ['Overall', 'Mass', 'Calc', 'CC View', 'MLO View']
    subgroup_accs = [
        df['is_correct'].mean() * 100,
        df[df['abnormality_type'] == 'MASS']['is_correct'].mean() * 100,
        df[df['abnormality_type'] == 'CALCIFICATION']['is_correct'].mean() * 100,
        df[df['image_view'] == 'CC']['is_correct'].mean() * 100,
        df[df['image_view'] == 'MLO']['is_correct'].mean() * 100
    ]
    
    y_s = np.arange(len(subgroups))
    ax_f.barh(y_s, subgroup_accs, color=['#2C3E50', '#3498DB', '#E67E22', '#16A085', '#8E44AD'],
              edgecolor='#1B2A4A', height=0.58)
    ax_f.set_yticks(y_s)
    ax_f.set_yticklabels(subgroups, fontsize=8.5, fontweight='bold')
    ax_f.set_xlabel('Subgroup Accuracy (%)', fontsize=9)
    ax_f.set_xlim(50, 75)
    ax_f.set_title('(f) Subgroup Accuracy Profile', fontsize=9.8, fontweight='bold', pad=10)
    ax_f.grid(axis='x', ls=':', alpha=0.5)
    
    for i, v in enumerate(subgroup_accs):
        ax_f.text(v + 0.8, i, f"{v:.2f}%", va='center', fontsize=7.8, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel G: Historical Cross-Generational FP and FN Evolution
    # -------------------------------------------------------------------------
    ax_g = axes[1, 2]
    # Confusion matrix values for V5, V6, V8, V9
    # V5 (N=525): TN=194, FP=116, FN=68, TP=147 (Sens=68.37%, Spec=62.70%, Acc=65.02%)
    # V6 (N=526): TN=195, FP=116, FN=71, TP=144 (Sens=66.98%, Spec=62.70%, Acc=64.45%)
    # V8 (N=525): TN=197, FP=113, FN=73, TP=142 (Sens=66.05%, Spec=63.55%, Acc=64.57%)
    # V9 (N=525): TN=166, FP=144, FN=51, TP=164 (Sens=76.28%, Spec=53.55%, Acc=62.86%)
    
    hist_models = ['V5', 'V6', 'V8 (Final)', 'V9 (Ext.)']
    fp_hist = [116, 116, 113, 144]
    fn_hist = [68, 71, 73, 51]
    
    x_h = np.arange(len(hist_models))
    w_h = 0.35
    ax_g.bar(x_h - w_h/2, fp_hist, w_h, label='False Positives (FP)', color=c_fp, edgecolor='#1B2A4A')
    ax_g.bar(x_h + w_h/2, fn_hist, w_h, label='False Negatives (FN)', color=c_fn, edgecolor='#1B2A4A')
    
    ax_g.set_title('(g) Historical Error Trade-off Evolution', fontsize=9.8, fontweight='bold', pad=10)
    ax_g.set_xticks(x_h)
    ax_g.set_xticklabels(hist_models, fontsize=7.8, fontweight='semibold')
    ax_g.set_ylabel('Absolute Error Count', fontsize=9)
    ax_g.set_ylim(0, 175)
    ax_g.grid(axis='y', ls=':', alpha=0.5)
    ax_g.legend(loc='upper right', frameon=True, fontsize=7.5)
    
    for i in range(len(hist_models)):
        ax_g.text(x_h[i] - w_h/2, fp_hist[i] + 3, f"{fp_hist[i]}", ha='center', fontsize=7.4, fontweight='bold')
        ax_g.text(x_h[i] + w_h/2, fn_hist[i] + 3, f"{fn_hist[i]}", ha='center', fontsize=7.4, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel H: Subgroup Sample Size & Governance Audit Box
    # -------------------------------------------------------------------------
    ax_h = axes[1, 3]
    ax_h.axis('off')
    
    audit_text = (
        "SUBGROUP AUDIT & GOVERNANCE\n"
        "====================================\n\n"
        "SAMPLE SIZES (LOCKED TEST):\n"
        "• Total Cohort:      N = 525 (100%)\n"
        "• Mass Abnormalities: N = 267 (50.9%)\n"
        "• Calcifications:    N = 258 (49.1%)\n"
        "• CC Views:          N = 250 (47.6%)\n"
        "• MLO Views:         N = 275 (52.4%)\n"
        "• True Benign:       N = 310 (59.0%)\n"
        "• True Malignant:    N = 215 (41.0%)\n\n"
        "METHODOLOGICAL CAVEATS:\n"
        "------------------------------------\n"
        "1. Subgroup analyses are purely descriptive\n"
        "   and observational.\n"
        "2. No causal inference should be deduced\n"
        "   solely from stratified error variances.\n"
        "3. Zero post-hoc threshold tuning or subgroup\n"
        "   re-calibration was permitted."
    )
    
    ax_h.text(0.04, 0.95, audit_text, transform=ax_h.transAxes,
              va='top', ha='left', fontsize=8.0, family='monospace',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#F4F6F9', edgecolor='#B0BEC5', lw=1.2))
    ax_h.set_title('(h) Scientific Governance Audit', fontsize=9.8, fontweight='bold', pad=10)

    out_dir = 'results/research_paper_figures/figure_07_error_analysis'
    os.makedirs(out_dir, exist_ok=True)
    
    png_path = os.path.join(out_dir, 'fig07_error_analysis.png')
    pdf_path = os.path.join(out_dir, 'fig07_error_analysis.pdf')
    svg_path = os.path.join(out_dir, 'fig07_error_analysis.svg')
    
    print(f"Exporting Figure 7 to {out_dir}...")
    fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[OK] Figure 7 successfully generated: PNG (600 DPI), PDF, SVG.")

if __name__ == '__main__':
    generate_fig07_error_analysis()
