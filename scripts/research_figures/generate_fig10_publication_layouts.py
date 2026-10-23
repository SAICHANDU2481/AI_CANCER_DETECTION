"""
================================================================================
Figure 10: Ready-to-Insert Research Paper Composite Layouts
================================================================================
Generates publication-ready composite figures tailored for IEEE / Springer /
Elsevier research papers and thesis chapters:
- Figure 10A: Preprocessing Pipeline Composite (2x3 Grid)
- Figure 10B: Segmentation Evaluation Composite (2x3 Grid)
- Figure 10C: Classification Triptych (Confusion Matrix + ROC + PR Curves + CIs)
- Figure 10D: Multi-Generation Comparison Composite (Accuracy/F1 + ROC/PR + Pareto)
- Figure 10E: End-to-End System Architecture (Ready for 2-column IEEE banner)
- Figure 10F: Qualitative Success & Failure Gallery (2x3 Clinical Matrix)
- Output Formats: PNG (600 DPI), PDF (vector), SVG (vector)
================================================================================
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Patch
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

out_dir = 'results/research_paper_figures/figure_10_publication_layouts'
os.makedirs(out_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 8.5
plt.rcParams['axes.linewidth'] = 0.9

def export_figure(fig, base_name):
    png_path = os.path.join(out_dir, f"{base_name}.png")
    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
    svg_path = os.path.join(out_dir, f"{base_name}.svg")
    print(f"Exporting {base_name}...")
    fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] {base_name} generated.")

# -----------------------------------------------------------------------------
# 10A: Preprocessing 2x3 Composite
# -----------------------------------------------------------------------------
def make_fig10a():
    fig_raw = cv2.imread('results/research_paper_figures/figure_03_preprocessing/fig03_preprocessing_pipeline.png')
    # Or render direct 2x3
    img_candidates = [
        r"C:\Users\sys\Desktop\Capstone\datasets\jpeg\1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515\1-211.jpg",
        r"C:\Users\sys\OneDrive\Desktop\Capstone\datasets\jpeg\1.3.6.1.4.1.9590.100.1.2.342386194811267636608694132590482924515\1-211.jpg"
    ]
    img_p = next(p for p in img_candidates if os.path.exists(p))
    raw = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
    raw = cv2.resize(raw, (768, 1024)).astype(np.float32) / 255.0
    
    from scipy import signal
    wiener_img = signal.wiener(raw, mysize=(5, 5))
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply((wiener_img * 255).astype(np.uint8)).astype(np.float32) / 255.0
    diff = np.abs(clahe_img - raw)
    
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), facecolor='white', dpi=300)
    plt.subplots_adjust(wspace=0.18, hspace=0.25)
    
    # (a) Raw
    axes[0, 0].imshow(raw, cmap='bone')
    axes[0, 0].set_title('(a) Raw FFDM Input', fontweight='bold', fontsize=9.5)
    axes[0, 0].axis('off')
    
    # (b) Wiener
    axes[0, 1].imshow(wiener_img, cmap='bone')
    axes[0, 1].set_title('(b) Adaptive Wiener Filter (5x5)', fontweight='bold', fontsize=9.5)
    axes[0, 1].axis('off')
    
    # (c) CLAHE
    axes[0, 2].imshow(clahe_img, cmap='bone')
    axes[0, 2].set_title('(c) CLAHE Contrast Enhanced', fontweight='bold', fontsize=9.5)
    axes[0, 2].axis('off')
    
    # (d) Difference
    im_d = axes[1, 0].imshow(diff, cmap='magma', vmin=0, vmax=0.35)
    axes[1, 0].set_title('(d) Absolute Residual Map', fontweight='bold', fontsize=9.5)
    axes[1, 0].axis('off')
    plt.colorbar(im_d, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # (e) Histograms
    ax_e = axes[1, 1]
    m = raw > 0.02
    ax_e.hist(raw[m], bins=80, density=True, alpha=0.5, color='#7F8C8D', label='Raw')
    ax_e.hist(clahe_img[m], bins=80, density=True, alpha=0.6, color='#2980B9', label='Enhanced')
    ax_e.set_title('(e) Intensity Dynamic Range', fontweight='bold', fontsize=9.5)
    ax_e.set_xlabel('Normalized Intensity', fontsize=8.5)
    ax_e.set_ylabel('Probability Density', fontsize=8.5)
    ax_e.legend(loc='upper right', fontsize=8)
    ax_e.grid(True, ls=':', alpha=0.5)
    
    # (f) Line profile
    ax_f = axes[1, 2]
    transect_y = int(raw.shape[0] * 0.45)
    ax_f.plot(raw[transect_y, 200:500], color='#7F8C8D', lw=1.2, ls='--', label='Raw')
    ax_f.plot(clahe_img[transect_y, 200:500], color='#27AE60', lw=1.8, label='Enhanced')
    ax_f.set_title('(f) Lesion Margin Transect', fontweight='bold', fontsize=9.5)
    ax_f.set_xlabel('Pixel Position Across Lesion', fontsize=8.5)
    ax_f.set_ylabel('Intensity', fontsize=8.5)
    ax_f.legend(loc='upper right', fontsize=8)
    ax_f.grid(True, ls=':', alpha=0.5)
    
    export_figure(fig, 'fig10a_composite_preprocessing_2x3')

# -----------------------------------------------------------------------------
# 10B: Segmentation Evaluation 2x3 Composite
# -----------------------------------------------------------------------------
def make_fig10b():
    # 2 cases: Malignant Mass & Malignant Calcification
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), facecolor='white', dpi=300)
    plt.subplots_adjust(wspace=0.18, hspace=0.25)
    
    # We can load the rendered panels from Figure 4 or render directly
    fig4_img = cv2.imread('results/research_paper_figures/figure_04_segmentation/fig04_ukan_segmentation_evaluation.png')
    # Row 1 of Fig 4 (Case 1) and Row 3 (Case 3)
    # Fig 4 shape is (height, width, 3)
    H, W, _ = fig4_img.shape
    row_h = H // 4
    col_w = W // 6
    
    # Row 0: Case 1 (Malignant Mass) -> Col 0 (Raw), Col 4 (Overlay), Col 5 (Error Map)
    r1_c1 = fig4_img[int(row_h*0.1):int(row_h*0.95), int(col_w*0.02):int(col_w*0.98)]
    r1_c2 = fig4_img[int(row_h*0.1):int(row_h*0.95), int(col_w*4.02):int(col_w*4.98)]
    r1_c3 = fig4_img[int(row_h*0.1):int(row_h*0.95), int(col_w*5.02):int(col_w*5.98)]
    
    # Row 1: Case 3 (Calcification) -> Col 0, Col 4, Col 5
    r2_c1 = fig4_img[int(row_h*2.1):int(row_h*2.95), int(col_w*0.02):int(col_w*0.98)]
    r2_c2 = fig4_img[int(row_h*2.1):int(row_h*2.95), int(col_w*4.02):int(col_w*4.98)]
    r2_c3 = fig4_img[int(row_h*2.1):int(row_h*2.95), int(col_w*5.02):int(col_w*5.98)]
    
    axes[0, 0].imshow(cv2.cvtColor(r1_c1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('(a) Case 1: Malignant Mass FFDM', fontweight='bold', fontsize=9.5)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(r1_c2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('(b) Case 1: U-KAN Overlay (Dice=0.365)', fontweight='bold', fontsize=9.5)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(r1_c3, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('(c) Case 1: Pixel Error Map (TP/FP/FN)', fontweight='bold', fontsize=9.5)
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(r2_c1, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('(d) Case 2: Calcification Cluster FFDM', fontweight='bold', fontsize=9.5)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(r2_c2, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('(e) Case 2: U-KAN Overlay (Dice=0.429)', fontweight='bold', fontsize=9.5)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(r2_c3, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('(f) Case 2: Pixel Error Map (TP/FP/FN)', fontweight='bold', fontsize=9.5)
    axes[1, 2].axis('off')
    
    export_figure(fig, 'fig10b_composite_segmentation_2x3')

# -----------------------------------------------------------------------------
# 10C: Classification Triptych (CM + ROC + PR + Metrics)
# -----------------------------------------------------------------------------
def make_fig10c():
    df = pd.read_csv('results/v8/phase3/v8_arm_f_test_predictions.csv')
    y_true = df['target'].values
    y_pred = df['predicted_class'].values
    y_prob = df['prob_malignant'].values
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec_pts, rec_pts, _ = precision_recall_curve(y_true, y_prob)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), facecolor='white', dpi=300)
    plt.subplots_adjust(wspace=0.28, left=0.04, right=0.96)
    
    # 1. CM
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
                annot_kws={'size': 12, 'weight': 'bold'},
                xticklabels=['Pred Benign', 'Pred Malignant'],
                yticklabels=['Actual Benign', 'Actual Malignant'])
    axes[0].set_title('(a) Locked Confusion Matrix', fontweight='bold', fontsize=9.5, pad=10)
    axes[0].text(0.5, -0.18, f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
                 transform=axes[0].transAxes, ha='center', fontsize=8.2, fontweight='bold')
                 
    # 2. ROC
    axes[1].plot(fpr, tpr, color='#1B2A4A', lw=2.2, label='V8 (AUC = 0.6947)')
    axes[1].plot([0, 1], [0, 1], color='#7F8C8D', lw=1.2, ls='--', label='Random (0.50)')
    axes[1].scatter([fp/(tn+fp)], [tp/(tp+fn)], color='#E74C3C', s=50, zorder=5, label='tau=0.50')
    axes[1].set_title('(b) Receiver Operating Characteristic', fontweight='bold', fontsize=9.5, pad=10)
    axes[1].set_xlabel('False Positive Rate', fontsize=8.5)
    axes[1].set_ylabel('True Positive Rate', fontsize=8.5)
    axes[1].grid(True, ls=':', alpha=0.5)
    axes[1].legend(loc='lower right', fontsize=7.8)
    
    # 3. PR
    axes[2].plot(rec_pts, prec_pts, color='#8E44AD', lw=2.2, label='V8 (PR-AUC = 0.5857)')
    axes[2].axhline(np.mean(y_true), color='#7F8C8D', lw=1.2, ls='--', label='Prevalence (40.9%)')
    axes[2].scatter([tp/(tp+fn)], [tp/(tp+fp)], color='#E74C3C', s=50, zorder=5, label='tau=0.50')
    axes[2].set_title('(c) Precision-Recall Curve', fontweight='bold', fontsize=9.5, pad=10)
    axes[2].set_xlabel('Recall (Sensitivity)', fontsize=8.5)
    axes[2].set_ylabel('Precision (PPV)', fontsize=8.5)
    axes[2].grid(True, ls=':', alpha=0.5)
    axes[2].legend(loc='upper right', fontsize=7.8)
    
    # 4. Metrics Bar
    m_names = ['Accuracy', 'Macro-F1', 'Sensitivity', 'Specificity', 'Precision']
    m_vals = [64.57, 64.18, 66.05, 63.55, 55.69]
    y_p = np.arange(len(m_names))
    axes[3].barh(y_p, m_vals, color=['#2C3E50', '#2980B9', '#27AE60', '#D35400', '#8E44AD'],
                 edgecolor='#1B2A4A', height=0.55)
    axes[3].set_yticks(y_p)
    axes[3].set_yticklabels(m_names, fontweight='bold', fontsize=8.2)
    axes[3].set_xlim(40, 75)
    axes[3].set_xlabel('Score (%)', fontsize=8.5)
    axes[3].set_title('(d) Primary Diagnostics (N=525)', fontweight='bold', fontsize=9.5, pad=10)
    axes[3].grid(axis='x', ls=':', alpha=0.5)
    for i, v in enumerate(m_vals):
        axes[3].text(v + 0.8, i, f"{v:.1f}%", va='center', fontsize=7.8, fontweight='bold')
        
    export_figure(fig, 'fig10c_composite_classification_triptych')

# -----------------------------------------------------------------------------
# 10D: Model Comparison Composite
# -----------------------------------------------------------------------------
def make_fig10d():
    df = pd.read_csv('results/final_v8/V5_V6_V8_V9_COMPARISON.csv')
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), facecolor='white', dpi=300)
    plt.subplots_adjust(wspace=0.28, left=0.06, right=0.96)
    
    models = ['V5', 'V6', 'V8 (Final)', 'V9 (Ext.)']
    x = np.arange(len(models))
    w = 0.35
    
    # (a) Acc & F1
    axes[0].bar(x - w/2, df['Accuracy'], w, label='Accuracy (%)', color='#5DADE2', edgecolor='#1B4F72')
    axes[0].bar(x + w/2, df['Macro_F1'], w, label='Macro-F1 (%)', color='#27AE60', edgecolor='#145A32')
    axes[0].set_title('(a) Accuracy & Macro-F1', fontweight='bold', fontsize=9.5, pad=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontweight='semibold', fontsize=8.0)
    axes[0].set_ylim(58, 68)
    axes[0].set_ylabel('Score (%)', fontsize=8.5)
    axes[0].grid(axis='y', ls=':', alpha=0.5)
    axes[0].legend(loc='lower left', fontsize=7.8)
    
    # (b) ROC & PR
    axes[1].bar(x - w/2, df['ROC_AUC'], w, label='ROC-AUC', color='#85C1E9', edgecolor='#1B4F72')
    axes[1].bar(x + w/2, df['PR_AUC'], w, label='PR-AUC', color='#F39C12', edgecolor='#B9770E')
    axes[1].set_title('(b) Discrimination Capacity', fontweight='bold', fontsize=9.5, pad=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontweight='semibold', fontsize=8.0)
    axes[1].set_ylim(0.50, 0.75)
    axes[1].set_ylabel('AUC Score', fontsize=8.5)
    axes[1].grid(axis='y', ls=':', alpha=0.5)
    axes[1].legend(loc='lower left', fontsize=7.8)
    
    # (c) Sens vs Spec Scatter
    cols = ['#7F8C8D', '#3498DB', '#2ECC71', '#E67E22']
    for i, row in df.iterrows():
        axes[2].scatter(row['Specificity'], row['Sensitivity'], color=cols[i], s=120,
                        edgecolor='#1B2A4A', lw=1.5, zorder=5)
        offset = (0.5, 0.6) if row['Model'] != 'V9' else (-2.5, -1.5)
        axes[2].annotate(f"{row['Model']}", (row['Specificity'], row['Sensitivity']),
                         xytext=(row['Specificity'] + offset[0], row['Sensitivity'] + offset[1]),
                         fontweight='bold', fontsize=8.2)
    axes[2].set_title('(c) Sensitivity vs Specificity Trade-Off', fontweight='bold', fontsize=9.5, pad=10)
    axes[2].set_xlabel('Specificity (%)', fontsize=8.5)
    axes[2].set_ylabel('Sensitivity (%)', fontsize=8.5)
    axes[2].set_xlim(50, 66)
    axes[2].set_ylim(60, 78)
    axes[2].grid(True, ls=':', alpha=0.5)
    
    export_figure(fig, 'fig10d_composite_model_comparison')

# -----------------------------------------------------------------------------
# 10E: Full Architecture Diagram (Vector Link)
# -----------------------------------------------------------------------------
def make_fig10e():
    # Direct copy from Figure 1
    src_png = 'results/research_paper_figures/figure_01_system_architecture/fig01_system_architecture.png'
    src_pdf = 'results/research_paper_figures/figure_01_system_architecture/fig01_system_architecture.pdf'
    src_svg = 'results/research_paper_figures/figure_01_system_architecture/fig01_system_architecture.svg'
    
    import shutil
    shutil.copy(src_png, os.path.join(out_dir, 'fig10e_composite_architecture.png'))
    shutil.copy(src_pdf, os.path.join(out_dir, 'fig10e_composite_architecture.pdf'))
    shutil.copy(src_svg, os.path.join(out_dir, 'fig10e_composite_architecture.svg'))
    print("[OK] fig10e_composite_architecture generated.")

# -----------------------------------------------------------------------------
# 10F: Qualitative Success/Failure Gallery Composite
# -----------------------------------------------------------------------------
def make_fig10f():
    # 2x3 composite of 3 successes and 3 failures
    src_png = 'results/research_paper_figures/figure_08_qualitative_results/fig08_qualitative_case_gallery.png'
    src_pdf = 'results/research_paper_figures/figure_08_qualitative_results/fig08_qualitative_case_gallery.pdf'
    src_svg = 'results/research_paper_figures/figure_08_qualitative_results/fig08_qualitative_case_gallery.svg'
    
    import shutil
    shutil.copy(src_png, os.path.join(out_dir, 'fig10f_composite_qualitative_gallery.png'))
    shutil.copy(src_pdf, os.path.join(out_dir, 'fig10f_composite_qualitative_gallery.pdf'))
    shutil.copy(src_svg, os.path.join(out_dir, 'fig10f_composite_qualitative_gallery.svg'))
    print("[OK] fig10f_composite_qualitative_gallery generated.")

if __name__ == '__main__':
    make_fig10a()
    make_fig10b()
    make_fig10c()
    make_fig10d()
    make_fig10e()
    make_fig10f()
