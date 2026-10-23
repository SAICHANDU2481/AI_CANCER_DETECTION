"""
================================================================================
Figure 11: Comparative Benchmark Against Literature & 92% RoI Paradigm Analysis
================================================================================
Generates publication-quality comparative evaluation charts:
- Panel A: Accuracy Across Task Paradigms & Evaluation Scopes (%)
- Panel B: ROC-AUC Discrimination Capability Across Paradigms
- Panel C: Data Partitioning Impact: Patient-Quarantined vs Leakage Gap
- Panel D: Structured Literature Comparison Matrix (Task, Split, Acc, AUC)
Exports: PDF vector, SVG vector, 300 DPI PNG
================================================================================
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_fig11_comparison():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'mathtext.fontset': 'cm',
        'axes.edgecolor': '#2C3E50',
        'axes.linewidth': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.color': '#2C3E50',
        'ytick.color': '#2C3E50',
        'text.color': '#2C3E50',
        'axes.labelcolor': '#2C3E50',
    })

    fig = plt.figure(figsize=(10.5, 7.8), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28, left=0.07, right=0.96, top=0.90, bottom=0.08)

    # -------------------------------------------------------------------------
    # Panel A: Accuracy Comparison
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    
    methods_patch = ['Shen et al.\n(2019) [5]', 'Patch-CNN\nBaseline', 'Ours: ViT\n(RoI-Only)']
    acc_patch = [91.40, 92.20, 91.80]
    
    methods_full = ['ResNet-50\n(Full-Scan)', 'U-Net+ViT\nBaseline', 'Ours: V8\n(Final Arm F)']
    acc_full = [59.24, 61.85, 64.57]
    
    x_patch = np.array([0, 1, 2])
    x_full = np.array([3.7, 4.7, 5.7])
    
    c_patch_lit = '#5DADE2'
    c_patch_ours = '#16A085'
    c_full_base = '#BDC3C7'
    c_full_ours = '#27AE60'
    
    bars1 = ax_a.bar(x_patch[:2], acc_patch[:2], color=c_patch_lit, edgecolor='#1B4F72', width=0.65)
    bar_our_patch = ax_a.bar(x_patch[2], acc_patch[2], color=c_patch_ours, edgecolor='#0E6251', width=0.65, hatch='//')
    
    bars2 = ax_a.bar(x_full[:2], acc_full[:2], color=c_full_base, edgecolor='#7F8C8D', width=0.65)
    bar_our_full = ax_a.bar(x_full[2], acc_full[2], color=c_full_ours, edgecolor='#145A32', width=0.65, hatch='\\\\')
    
    for b in list(bars1) + [bar_our_patch[0]] + list(bars2) + [bar_our_full[0]]:
        h = b.get_height()
        ax_a.text(b.get_x() + b.get_width()/2., h + 1.2, f'{h:.1f}%', ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#2C3E50')
        
    ax_a.axvline(2.85, color='#BDC3C7', linestyle='--', linewidth=1.2)
    ax_a.text(1.0, 99.5, 'Pre-Cropped RoI Tasks\n(Radiologist Pre-Localized)', ha='center', va='center', fontsize=7.8, fontweight='bold', color='#1B4F72', bbox=dict(boxstyle='round,pad=0.25', facecolor='#EBF5FB', edgecolor='#AED6F1', lw=0.8))
    ax_a.text(4.7, 99.5, 'Whole-Scan Screening\n(Autonomous Discovery)', ha='center', va='center', fontsize=7.8, fontweight='bold', color='#145A32', bbox=dict(boxstyle='round,pad=0.25', facecolor='#EAFAF1', edgecolor='#A9DFBF', lw=0.8))
    
    all_x = list(x_patch) + list(x_full)
    all_labels = methods_patch + methods_full
    ax_a.set_xticks(all_x)
    ax_a.set_xticklabels(all_labels, fontsize=7.4)
    ax_a.set_ylabel('Classification Accuracy (%)', fontsize=9.0, fontweight='bold')
    ax_a.set_ylim(45, 108)
    ax_a.set_title('(a) Diagnostic Accuracy Across Task Paradigms', fontsize=9.6, fontweight='bold', pad=12)
    ax_a.grid(axis='y', linestyle=':', alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel B: ROC-AUC Comparison
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    
    auc_patch = [0.9320, 0.9150, 0.9240]
    auc_full = [0.6120, 0.6435, 0.6947]
    
    bars_auc1 = ax_b.bar(x_patch[:2], auc_patch[:2], color='#85C1E9', edgecolor='#1B4F72', width=0.65)
    bar_our_auc1 = ax_b.bar(x_patch[2], auc_patch[2], color='#48C9B0', edgecolor='#0E6251', width=0.65, hatch='//')
    
    bars_auc2 = ax_b.bar(x_full[:2], auc_full[:2], color='#D5D8DC', edgecolor='#7F8C8D', width=0.65)
    bar_our_auc2 = ax_b.bar(x_full[2], auc_full[2], color='#52BE80', edgecolor='#145A32', width=0.65, hatch='\\\\')
    
    for b in list(bars_auc1) + [bar_our_auc1[0]] + list(bars_auc2) + [bar_our_auc2[0]]:
        h = b.get_height()
        ax_b.text(b.get_x() + b.get_width()/2., h + 0.015, f'{h:.4f}', ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='#2C3E50')
        
    ax_b.axvline(2.85, color='#BDC3C7', linestyle='--', linewidth=1.2)
    ax_b.set_xticks(all_x)
    ax_b.set_xticklabels(all_labels, fontsize=7.4)
    ax_b.set_ylabel('Area Under ROC Curve (ROC-AUC)', fontsize=9.0, fontweight='bold')
    ax_b.set_ylim(0.48, 1.05)
    ax_b.set_title('(b) Discrimination Capability (ROC-AUC)', fontsize=9.6, fontweight='bold', pad=12)
    ax_b.grid(axis='y', linestyle=':', alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel C: Data Partitioning & Leakage Impact
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    
    categories = ['Scan-Level Split\n(Patient Data Leakage)', 'Patient-Quarantined\n(Strict Zero Leakage)']
    acc_with_leakage = [91.80, 64.57]
    
    ax_c.bar([0], [acc_with_leakage[0]], color='#E74C3C', edgecolor='#78281F', width=0.52, yerr=[[1.5], [1.8]], capsize=5)
    ax_c.bar([1], [acc_with_leakage[1]], color='#2ECC71', edgecolor='#145A32', width=0.52, yerr=[[5.51], [5.22]], capsize=5)
    
    ax_c.text(0, acc_with_leakage[0] + 3.8, f'{acc_with_leakage[0]:.1f}%\n(Inflated Test Acc)', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#922B21')
    ax_c.text(1, acc_with_leakage[1] + 7.0, f'{acc_with_leakage[1]:.2f}%\n[59.1% - 69.8% CI]\n(Clinical Ground Truth)', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#145A32')
    
    # Delta annotation arrow
    ax_c.annotate('', xy=(1, 74), xytext=(0, 90),
                 arrowprops=dict(arrowstyle="->", color='#C0392B', lw=1.8, ls='--'))
    ax_c.text(0.5, 83.5, 'Leakage Inflation Gap\n$\\Delta \\approx -27.2\\%$', ha='center', va='center', fontsize=8.2, fontweight='bold', color='#C0392B', bbox=dict(boxstyle='round,pad=0.25', facecolor='#FDEDEC', edgecolor='#F5B7B1', lw=0.8))
    
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(categories, fontsize=8.4, fontweight='bold')
    ax_c.set_ylabel('Observed Test Accuracy (%)', fontsize=9.0, fontweight='bold')
    ax_c.set_ylim(40, 108)
    ax_c.set_title('(c) Data Leakage Audit: Why Literature Reports >90%', fontsize=9.6, fontweight='bold', pad=12)
    ax_c.grid(axis='y', linestyle=':', alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel D: Benchmark Matrix Table Graphic
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis('off')
    
    table_data = [
        ['Paradigm / Model', 'Task Scope', 'Patient Isolation', 'Acc (%)', 'ROC-AUC'],
        ['Shen et al. [5]', 'RoI Crop', 'Scan Split (Leakage)', '91.40', '0.9320'],
        ['Patch-CNN Base', 'RoI Crop', 'Scan Split (Leakage)', '92.20', '0.9150'],
        ['Ours: ViT RoI', 'RoI Crop', 'Patient-Quarantined', '91.80', '0.9240'],
        ['ResNet-50 Base', 'Full Scan', 'Patient-Quarantined', '59.24', '0.6120'],
        ['U-Net + ViT Base', 'Full Scan', 'Patient-Quarantined', '61.85', '0.6435'],
        ['Ours: V8 Arm F', 'Full Scan', 'Strict Isolation', '64.57', '0.6947']
    ]
    
    tbl = ax_d.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.27, 0.20, 0.31, 0.12, 0.14]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.45)
    
    # Style header
    for c in range(5):
        cell = tbl[(0, c)]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')
        
    # Style rows
    for r in range(1, 7):
        if r == 3: # Our RoI
            bg = '#D4EFDF'
            for c in range(5):
                cell = tbl[(r, c)]
                cell.set_facecolor(bg)
                cell.set_text_props(fontweight='bold', color='#145A32')
        elif r == 6: # Our V8
            bg = '#A9DFBF'
            for c in range(5):
                cell = tbl[(r, c)]
                cell.set_facecolor(bg)
                cell.set_text_props(fontweight='bold', color='#0E6251')
        elif r % 2 == 1:
            bg = '#F8F9FA'
            for c in range(5):
                tbl[(r, c)].set_facecolor(bg)
        else:
            bg = '#EDF2F7'
            for c in range(5):
                tbl[(r, c)].set_facecolor(bg)
                
    ax_d.set_title('(d) Comprehensive Benchmark & Governance Matrix', fontsize=9.6, fontweight='bold', pad=12)
    
    # Main Figure Title with ample clearance
    fig.suptitle('Comparative Literature Benchmark and Task-Level Performance Disaggregation', fontsize=12.0, fontweight='bold', color='#1A252F', y=0.97)
    
    # Export paths
    out_dir_latex = 'research_paper_latex/figures'
    out_dir_results = 'results/research_paper_figures/figure_11_literature_comparison'
    os.makedirs(out_dir_latex, exist_ok=True)
    os.makedirs(out_dir_results, exist_ok=True)
    
    for d in [out_dir_latex, out_dir_results]:
        png_path = os.path.join(d, 'fig11_literature_comparison.png')
        pdf_path = os.path.join(d, 'fig11_literature_comparison.pdf')
        svg_path = os.path.join(d, 'fig11_literature_comparison.svg')
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
        fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
        print(f"Exported Figure 11 to {d}")
        
    plt.close(fig)
    print("[SUCCESS] Figure 11 regenerated cleanly.")

if __name__ == '__main__':
    generate_fig11_comparison()
