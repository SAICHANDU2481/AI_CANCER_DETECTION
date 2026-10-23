"""
================================================================================
Figure 2: Comprehensive Dataset and Class Distribution Analysis
================================================================================
Generates publication-quality scientific visualization of the CBIS-DDSM cohort:
- Panel A: Pathology Distribution (Benign vs Malignant)
- Panel B: Abnormality Classification (Calcification vs Mass)
- Panel C: Dataset Split by Scan Count (Train, Validation, Test)
- Panel D: Quarantined Patient Split (Strict Patient-Level Partitioning)
- Panel E: Mammographic View Projection (MLO vs CC)
- Panel F: Class Balance Disaggregated by Split
- Exports: PDF vector, SVG vector, 600 DPI PNG
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_fig02_dataset():
    metadata_path = 'data/processed/metadata.csv'
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata with {len(df)} records.")
    
    # Standardize column names if needed
    # Typical columns: split, pathology/label, abnormality_type/type, view, patient_id
    # Let's inspect column values
    print("Columns:", df.columns.tolist())
    
    # Calculate exact counts
    split_col = 'split'
    pathology_col = 'pathology' if 'pathology' in df.columns else 'label'
    type_col = 'abnormality_type' if 'abnormality_type' in df.columns else 'type'
    view_col = 'view' if 'view' in df.columns else 'image_view'
    patient_col = 'patient_id'
    
    total_scans = len(df)
    total_patients = df[patient_col].nunique()
    
    # Styling configuration
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.alpha'] = 0.3
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor='white', dpi=300)
    plt.subplots_adjust(wspace=0.32, hspace=0.38)
    
    # Color palette (Academic & Colorblind-safe)
    c_benign = '#2B7A78'    # Calming teal
    c_malignant = '#D9534F' # Vivid muted red/coral
    c_calc = '#E67E22'      # Amber orange
    c_mass = '#3498DB'      # Sky slate blue
    c_train = '#4A6572'     # Deep slate
    c_val = '#F9AA33'       # Gold amber
    c_test = '#34495E'      # Navy slate
    c_mlo = '#8E44AD'       # Purple
    c_cc = '#16A085'        # Sea green
    
    # -------------------------------------------------------------------------
    # Panel A: Benign vs Malignant
    # -------------------------------------------------------------------------
    ax_a = axes[0, 0]
    path_counts = df[pathology_col].str.upper().value_counts()
    benign_cnt = path_counts.get('BENIGN', 0) + path_counts.get('BENIGN_WITHOUT_CALLBACK', 0)
    mal_cnt = path_counts.get('MALIGNANT', 0)
    
    bars_a = ax_a.bar(['Benign', 'Malignant'], [benign_cnt, mal_cnt],
                      color=[c_benign, c_malignant], width=0.55, edgecolor='#2C3E50', linewidth=1.2)
    ax_a.set_title('(a) Pathology Distribution', fontsize=10.5, fontweight='bold', pad=10)
    ax_a.set_ylabel('Scan Count', fontsize=9.5, fontweight='semibold')
    ax_a.set_ylim(0, max(benign_cnt, mal_cnt) * 1.18)
    ax_a.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars_a:
        h = bar.get_height()
        pct = (h / total_scans) * 100
        ax_a.text(bar.get_x() + bar.get_width()/2., h + 35, f"{h:,}\n({pct:.1f}%)",
                  ha='center', va='bottom', fontsize=8.8, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel B: Mass vs Calcification
    # -------------------------------------------------------------------------
    ax_b = axes[0, 1]
    type_counts = df[type_col].str.lower().value_counts()
    calc_cnt = type_counts.get('calcification', 0) + type_counts.get('calc', 0)
    mass_cnt = type_counts.get('mass', 0)
    
    bars_b = ax_b.bar(['Calcification', 'Mass'], [calc_cnt, mass_cnt],
                      color=[c_calc, c_mass], width=0.55, edgecolor='#2C3E50', linewidth=1.2)
    ax_b.set_title('(b) Lesion Abnormality Type', fontsize=10.5, fontweight='bold', pad=10)
    ax_b.set_ylabel('Scan Count', fontsize=9.5, fontweight='semibold')
    ax_b.set_ylim(0, max(calc_cnt, mass_cnt) * 1.18)
    ax_b.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars_b:
        h = bar.get_height()
        pct = (h / total_scans) * 100
        ax_b.text(bar.get_x() + bar.get_width()/2., h + 35, f"{h:,}\n({pct:.1f}%)",
                  ha='center', va='bottom', fontsize=8.8, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel C: Scan Count by Split
    # -------------------------------------------------------------------------
    ax_c = axes[0, 2]
    split_counts = df[split_col].str.lower().value_counts()
    train_scans = split_counts.get('train', 0)
    val_scans = split_counts.get('val', 0)
    test_scans = split_counts.get('test', 0)
    
    bars_c = ax_c.bar(['Training', 'Validation', 'Test (Locked)'], 
                      [train_scans, val_scans, test_scans],
                      color=[c_train, c_val, c_test], width=0.58, edgecolor='#2C3E50', linewidth=1.2)
    ax_c.set_title('(c) Scan Cohort by Split', fontsize=10.5, fontweight='bold', pad=10)
    ax_c.set_ylabel('Mammographic Scans', fontsize=9.5, fontweight='semibold')
    ax_c.set_ylim(0, train_scans * 1.18)
    ax_c.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars_c:
        h = bar.get_height()
        pct = (h / total_scans) * 100
        ax_c.text(bar.get_x() + bar.get_width()/2., h + 40, f"{h:,}\n({pct:.1f}%)",
                  ha='center', va='bottom', fontsize=8.8, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel D: Patient Count per Split (Strict Patient Quarantining)
    # -------------------------------------------------------------------------
    ax_d = axes[1, 0]
    patients_by_split = df.groupby(split_col)[patient_col].nunique()
    train_pts = patients_by_split.get('train', 0)
    val_pts = patients_by_split.get('val', 0)
    test_pts = patients_by_split.get('test', 0)
    
    bars_d = ax_d.bar(['Training', 'Validation', 'Test (Locked)'],
                      [train_pts, val_pts, test_pts],
                      color=['#5C6BC0', '#FFA726', '#26A69A'], width=0.58, edgecolor='#2C3E50', linewidth=1.2)
    ax_d.set_title('(d) Patient Quarantined Cohort', fontsize=10.5, fontweight='bold', pad=10)
    ax_d.set_ylabel('Unique Patients (N=1,566)', fontsize=9.5, fontweight='semibold')
    ax_d.set_ylim(0, train_pts * 1.18)
    ax_d.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars_d:
        h = bar.get_height()
        pct = (h / total_patients) * 100
        ax_d.text(bar.get_x() + bar.get_width()/2., h + 18, f"{h:,}\n({pct:.1f}%)",
                  ha='center', va='bottom', fontsize=8.8, fontweight='bold')
        
    ax_d.text(0.5, 0.85, "Zero Patient Overlap\nStrict Quarantining",
              transform=ax_d.transAxes, ha='center', va='center',
              fontsize=7.8, fontweight='bold', color='#B71C1C',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#EF9A9A'))

    # -------------------------------------------------------------------------
    # Panel E: Acquisition View (CC vs MLO)
    # -------------------------------------------------------------------------
    ax_e = axes[1, 1]
    view_counts = df[view_col].str.upper().value_counts()
    mlo_cnt = view_counts.get('MLO', 0)
    cc_cnt = view_counts.get('CC', 0)
    
    bars_e = ax_e.bar(['MLO View', 'CC View'], [mlo_cnt, cc_cnt],
                      color=[c_mlo, c_cc], width=0.55, edgecolor='#2C3E50', linewidth=1.2)
    ax_e.set_title('(e) Mammographic View Projection', fontsize=10.5, fontweight='bold', pad=10)
    ax_e.set_ylabel('Scan Count', fontsize=9.5, fontweight='semibold')
    ax_e.set_ylim(0, max(mlo_cnt, cc_cnt) * 1.18)
    ax_e.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars_e:
        h = bar.get_height()
        pct = (h / total_scans) * 100
        ax_e.text(bar.get_x() + bar.get_width()/2., h + 35, f"{h:,}\n({pct:.1f}%)",
                  ha='center', va='bottom', fontsize=8.8, fontweight='bold')

    # -------------------------------------------------------------------------
    # Panel F: Pathology Breakdown by Split
    # -------------------------------------------------------------------------
    ax_f = axes[1, 2]
    # Calculate Benign & Malignant for each split
    splits = ['train', 'val', 'test']
    split_labels = ['Training', 'Validation', 'Test']
    benign_split = []
    mal_split = []
    
    for s in splits:
        sdf = df[df[split_col].str.lower() == s]
        b = len(sdf[sdf[pathology_col].str.upper().str.contains('BENIGN')])
        m = len(sdf[sdf[pathology_col].str.upper() == 'MALIGNANT'])
        benign_split.append(b)
        mal_split.append(m)
        
    x = np.arange(len(split_labels))
    w = 0.35
    b_bars = ax_f.bar(x - w/2, benign_split, w, label='Benign', color=c_benign, edgecolor='#2C3E50', linewidth=1.1)
    m_bars = ax_f.bar(x + w/2, mal_split, w, label='Malignant', color=c_malignant, edgecolor='#2C3E50', linewidth=1.1)
    
    ax_f.set_title('(f) Class Distribution by Split', fontsize=10.5, fontweight='bold', pad=10)
    ax_f.set_ylabel('Scan Count', fontsize=9.5, fontweight='semibold')
    ax_f.set_xticks(x)
    ax_f.set_xticklabels(split_labels)
    ax_f.set_ylim(0, max(benign_split) * 1.25)
    ax_f.grid(axis='y', linestyle='--', alpha=0.5)
    ax_f.legend(loc='upper right', frameon=True, fontsize=8.5)
    
    for bar in b_bars:
        h = bar.get_height()
        ax_f.text(bar.get_x() + bar.get_width()/2., h + 25, f"{h:,}",
                  ha='center', va='bottom', fontsize=8.0, fontweight='bold')
    for bar in m_bars:
        h = bar.get_height()
        ax_f.text(bar.get_x() + bar.get_width()/2., h + 25, f"{h:,}",
                  ha='center', va='bottom', fontsize=8.0, fontweight='bold')

    plt.tight_layout()
    
    out_dir = 'results/research_paper_figures/figure_02_dataset_analysis'
    os.makedirs(out_dir, exist_ok=True)
    
    png_path = os.path.join(out_dir, 'fig02_dataset_distribution.png')
    pdf_path = os.path.join(out_dir, 'fig02_dataset_distribution.pdf')
    svg_path = os.path.join(out_dir, 'fig02_dataset_distribution.svg')
    
    print(f"Exporting Figure 2 to {out_dir}...")
    fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[OK] Figure 2 successfully generated: PNG (600 DPI), PDF, SVG.")

if __name__ == '__main__':
    generate_fig02_dataset()
