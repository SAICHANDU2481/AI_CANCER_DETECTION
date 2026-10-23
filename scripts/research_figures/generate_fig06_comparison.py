"""
================================================================================
Figure 6: Model Comparison Across Generations (V5 -> V6 -> V8 -> V9)
================================================================================
Generates publication-quality comparative evaluation charts:
- Panel A: Accuracy (%) and Macro-F1 (%) Grouped Bar Chart
- Panel B: ROC-AUC and PR-AUC Discrimination Metric Comparison
- Panel C: Sensitivity vs Specificity Trade-Off Scatter Analysis
- Panel D: Accuracy vs Sensitivity Clinical Operating Space
- Panel E: Multi-Metric Hexagonal Radar / Spider Chart
- Panel F: Formal Architectural Evolution & Locked Results Matrix
- Strict Governance Enforcement:
  * V8 is prominently distinguished as the FROZEN FINAL REPORTED MODEL.
  * V9 is clearly marked as an EXPERIMENTAL EXTENSION (not final model).
  * Exact verified locked test results used throughout.
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
from matplotlib.patches import Patch

def generate_fig06_comparison():
    comp_path = 'results/final_v8/V5_V6_V8_V9_COMPARISON.csv'
    if not os.path.exists(comp_path):
        raise FileNotFoundError(f"Missing comparison table at {comp_path}")
        
    df = pd.read_csv(comp_path)
    print(f"Loaded comparison metrics for {len(df)} generations.")
    
    # Model descriptors
    models = df['Model'].tolist() # ['V5', 'V6', 'V8', 'V9']
    model_labels = [
        'V5 (Arm I)\nHistorical',
        'V6 (Arm C)\nBenchmark',
        'V8 (Arm F)\nFINAL REPORTED',
        'V9 (Arm G)\nExperimental Ext.'
    ]
    
    # Palette
    c_v5 = '#7F8C8D'      # Neutral gray
    c_v6 = '#3498DB'      # Slate blue
    c_v8 = '#2ECC71'      # Highlighting Emerald Green (Final)
    c_v9 = '#E67E22'      # Amber Orange (Experimental)
    colors = [c_v5, c_v6, c_v8, c_v9]
    edge_colors = ['#2C3E50', '#1B4F72', '#145A32', '#7E5109']
    
    # Styling configuration
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 8.5
    plt.rcParams['axes.linewidth'] = 0.9
    
    fig = plt.figure(figsize=(15.5, 11), facecolor='white', dpi=300)
    gs = fig.add_gridspec(2, 3, wspace=0.32, hspace=0.35, left=0.06, right=0.96, top=0.93, bottom=0.07)
    
    # -------------------------------------------------------------------------
    # Panel A: Accuracy and Macro-F1 Comparison
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(models))
    w = 0.35
    
    acc_bars = ax_a.bar(x - w/2, df['Accuracy'], w, label='Accuracy (%)',
                        color=['#95A5A6', '#5DADE2', '#2ECC71', '#F39C12'],
                        edgecolor=edge_colors, linewidth=1.1)
    f1_bars = ax_a.bar(x + w/2, df['Macro_F1'], w, label='Macro-F1 (%)',
                       color=['#7F8C8D', '#2980B9', '#27AE60', '#D35400'],
                       edgecolor=edge_colors, linewidth=1.1)
    
    ax_a.set_title('(a) Accuracy & Macro-F1 across Generations', fontsize=9.8, fontweight='bold', pad=10)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(model_labels, fontsize=8.0, fontweight='semibold')
    ax_a.set_ylabel('Score (%)', fontsize=9)
    ax_a.set_ylim(55, 70)
    ax_a.grid(axis='y', ls=':', alpha=0.5)
    ax_a.legend(loc='lower left', frameon=True, fontsize=8.0)
    
    # Value annotations
    for bar in acc_bars:
        h = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width()/2., h + 0.3, f"{h:.2f}",
                  ha='center', va='bottom', fontsize=7.6, fontweight='bold')
    for bar in f1_bars:
        h = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width()/2., h + 0.3, f"{h:.2f}",
                  ha='center', va='bottom', fontsize=7.6, fontweight='bold')

    # Star badge over V8
    ax_a.text(2, 67.5, "[FINAL MODEL]", ha='center', va='center',
              fontsize=7.2, fontweight='bold', color='#145A32',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F8F5', edgecolor='#2ECC71', lw=1.2))

    # -------------------------------------------------------------------------
    # Panel B: ROC-AUC and PR-AUC Comparison
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    roc_bars = ax_b.bar(x - w/2, df['ROC_AUC'], w, label='ROC-AUC',
                        color=['#BDC3C7', '#85C1E9', '#58D68D', '#F8C471'],
                        edgecolor=edge_colors, linewidth=1.1)
    pr_bars = ax_b.bar(x + w/2, df['PR_AUC'], w, label='PR-AUC',
                       color=['#7F8C8D', '#1B4F72', '#1E8449', '#BA4A00'],
                       edgecolor=edge_colors, linewidth=1.1)
    
    ax_b.set_title('(b) Discrimination Capacity (ROC & PR AUC)', fontsize=9.8, fontweight='bold', pad=10)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(model_labels, fontsize=8.0, fontweight='semibold')
    ax_b.set_ylabel('Area Under Curve (AUC)', fontsize=9)
    ax_b.set_ylim(0.48, 0.76)
    ax_b.grid(axis='y', ls=':', alpha=0.5)
    ax_b.legend(loc='lower left', frameon=True, fontsize=8.0)
    
    for bar in roc_bars:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.007, f"{h:.4f}",
                  ha='center', va='bottom', fontsize=7.4, fontweight='bold')
    for bar in pr_bars:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.007, f"{h:.4f}",
                  ha='center', va='bottom', fontsize=7.4, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel C: Sensitivity vs Specificity Trade-Off
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    
    for i, row in df.iterrows():
        m = row['Model']
        sens = row['Sensitivity']
        spec = row['Specificity']
        col = colors[i]
        edge = edge_colors[i]
        size = 140 if m == 'V8' else 100
        marker = 's' if m == 'V8' else ('^' if m == 'V9' else 'o')
        
        ax_c.scatter(spec, sens, s=size, color=col, edgecolor=edge, linewidth=1.6,
                     marker=marker, zorder=5)
        
        offset_y = 1.0 if m in ['V5', 'V8'] else -1.5
        offset_x = 0.5 if m != 'V9' else -1.2
        lbl = f"{m} (Final)" if m == 'V8' else (f"{m} (Ext.)" if m == 'V9' else m)
        ax_c.annotate(f"{lbl}\n({spec:.1f}%, {sens:.1f}%)",
                      xy=(spec, sens), xytext=(spec + offset_x, sens + offset_y),
                      fontsize=8.0, fontweight='bold', color=edge,
                      arrowprops=dict(arrowstyle='->', color=edge, lw=0.9))
        
    ax_c.set_title('(c) Sensitivity vs Specificity Trade-off', fontsize=9.8, fontweight='bold', pad=10)
    ax_c.set_xlabel('Specificity (%) [Benign Retention]', fontsize=9)
    ax_c.set_ylabel('Sensitivity (%) [Malignant Recall]', fontsize=9)
    ax_c.set_xlim(50, 68)
    ax_c.set_ylim(60, 80)
    ax_c.grid(True, ls=':', alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel D: Accuracy vs Sensitivity Scatter Space
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 0])
    
    for i, row in df.iterrows():
        m = row['Model']
        acc = row['Accuracy']
        sens = row['Sensitivity']
        col = colors[i]
        edge = edge_colors[i]
        size = 140 if m == 'V8' else 100
        marker = 's' if m == 'V8' else ('^' if m == 'V9' else 'o')
        
        ax_d.scatter(acc, sens, s=size, color=col, edgecolor=edge, linewidth=1.6,
                     marker=marker, zorder=5)
        
        lbl = f"{m} (Final)" if m == 'V8' else (f"{m} (Ext.)" if m == 'V9' else m)
        ax_d.annotate(lbl, xy=(acc, sens), xytext=(acc + 0.12, sens + 0.6),
                      fontsize=8.2, fontweight='bold', color=edge)

    ax_d.set_title('(d) Overall Accuracy vs Sensitivity', fontsize=9.8, fontweight='bold', pad=10)
    ax_d.set_xlabel('Accuracy (%)', fontsize=9)
    ax_d.set_ylabel('Sensitivity (%)', fontsize=9)
    ax_d.set_xlim(62, 66)
    ax_d.set_ylim(63, 79)
    ax_d.grid(True, ls=':', alpha=0.5)
    
    # Highlight trade-off zone
    ax_d.text(0.05, 0.12, "V9 increases Sensitivity (+10.2%)\nat cost of -10.0% Specificity\nand -1.71% Overall Accuracy.",
              transform=ax_d.transAxes, fontsize=7.6, color='#78281F',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', edgecolor='#F5B7B1'))

    # -------------------------------------------------------------------------
    # Panel E: Hexagonal Radar / Spider Chart
    # -------------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 1], polar=True)
    
    radar_metrics = ['Accuracy', 'Macro-F1', 'ROC-AUC x100', 'PR-AUC x100', 'Sensitivity', 'Specificity']
    num_vars = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # complete the loop
    
    ax_e.set_theta_offset(np.pi / 2)
    ax_e.set_theta_direction(-1)
    ax_e.set_xticks(angles[:-1])
    ax_e.set_xticklabels(radar_metrics, fontsize=7.5, fontweight='bold')
    ax_e.set_ylim(45, 80)
    ax_e.set_yticks([50, 60, 70])
    ax_e.set_yticklabels(['50', '60', '70'], fontsize=6.8, color='#7F8C8D')
    
    for i, row in df.iterrows():
        m = row['Model']
        vals = [
            row['Accuracy'],
            row['Macro_F1'],
            row['ROC_AUC'] * 100,
            row['PR_AUC'] * 100,
            row['Sensitivity'],
            row['Specificity']
        ]
        vals += vals[:1]
        
        lw = 2.4 if m == 'V8' else 1.5
        ls = '-' if m in ['V8', 'V6'] else ('--' if m == 'V9' else ':')
        ax_e.plot(angles, vals, label=m, color=colors[i], linewidth=lw, linestyle=ls)
        if m == 'V8':
            ax_e.fill(angles, vals, color=colors[i], alpha=0.15)
            
    ax_e.set_title('(e) Multi-Metric Radar Comparison', fontsize=9.8, fontweight='bold', pad=15)
    ax_e.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=True, fontsize=7.8)

    # -------------------------------------------------------------------------
    # Panel F: Quantitative Summary Table & Status Inset
    # -------------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis('off')
    
    table_headers = ['Metric / Property', 'V5 (Arm I)', 'V6 (Arm C)', 'V8 (Arm F)', 'V9 (Arm G)']
    table_rows = [
        ['Status', 'Historical', 'Benchmark', 'FINAL REPORTED', 'Experimental'],
        ['Acc (%)', f"{df.loc[0, 'Accuracy']:.2f}%", f"{df.loc[1, 'Accuracy']:.2f}%", f"{df.loc[2, 'Accuracy']:.2f}%", f"{df.loc[3, 'Accuracy']:.2f}%"],
        ['Macro-F1 (%)', f"{df.loc[0, 'Macro_F1']:.2f}%", f"{df.loc[1, 'Macro_F1']:.2f}%", f"{df.loc[2, 'Macro_F1']:.2f}%", f"{df.loc[3, 'Macro_F1']:.2f}%"],
        ['ROC-AUC', f"{df.loc[0, 'ROC_AUC']:.4f}", f"{df.loc[1, 'ROC_AUC']:.4f}", f"{df.loc[2, 'ROC_AUC']:.4f}", f"{df.loc[3, 'ROC_AUC']:.4f}"],
        ['PR-AUC', f"{df.loc[0, 'PR_AUC']:.4f}", f"{df.loc[1, 'PR_AUC']:.4f}", f"{df.loc[2, 'PR_AUC']:.4f}", f"{df.loc[3, 'PR_AUC']:.4f}"],
        ['Sensitivity (%)', f"{df.loc[0, 'Sensitivity']:.2f}%", f"{df.loc[1, 'Sensitivity']:.2f}%", f"{df.loc[2, 'Sensitivity']:.2f}%", f"{df.loc[3, 'Sensitivity']:.2f}%"],
        ['Specificity (%)', f"{df.loc[0, 'Specificity']:.2f}%", f"{df.loc[1, 'Specificity']:.2f}%", f"{df.loc[2, 'Specificity']:.2f}%", f"{df.loc[3, 'Specificity']:.2f}%"],
        ['ViT Latent Dim', '768-d', '3,072-d', '4,608-d', '4,608-d + TTA'],
        ['Quarantined Test', '525 Scans', '526 Scans', '525 Scans', '525 Scans']
    ]
    
    tbl = ax_f.table(cellText=table_rows, colLabels=table_headers,
                     loc='center', cellLoc='center', colWidths=[0.30, 0.17, 0.17, 0.20, 0.16])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.6)
    tbl.scale(1.0, 1.45)
    
    for (r_idx, c_idx), cell in tbl.get_celld().items():
        if r_idx == 0:
            cell.set_facecolor('#1B2A4A')
            cell.set_text_props(color='white', fontweight='bold')
        elif c_idx == 3: # V8 Column highlighted
            cell.set_facecolor('#E8F8F5')
            cell.set_text_props(fontweight='bold', color='#145A32')
        elif r_idx % 2 == 1:
            cell.set_facecolor('#F8F9FA')
        else:
            cell.set_facecolor('#EDF2F7')
            
    ax_f.set_title('(f) Formal Historical Performance Matrix', fontsize=9.8, fontweight='bold', pad=12)

    out_dir = 'results/research_paper_figures/figure_06_model_comparison'
    os.makedirs(out_dir, exist_ok=True)
    
    png_path = os.path.join(out_dir, 'fig06_generation_comparison.png')
    pdf_path = os.path.join(out_dir, 'fig06_generation_comparison.pdf')
    svg_path = os.path.join(out_dir, 'fig06_generation_comparison.svg')
    
    print(f"Exporting Figure 6 to {out_dir}...")
    fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[OK] Figure 6 successfully generated: PNG (600 DPI), PDF, SVG.")

if __name__ == '__main__':
    generate_fig06_comparison()
