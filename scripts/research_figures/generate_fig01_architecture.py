"""
================================================================================
Figure 1: Complete V8 System Architecture Generator
================================================================================
Generates publication-quality vector architecture diagram for the frozen V8
breast cancer AI research prototype:
- High-level end-to-end pipeline flow
- Preprocessing (Wiener filter + CLAHE + Normalization)
- Dual-Resolution U-KAN Segmentation (512 + 768)
- Deterministic Top-K proposal generation (K=3)
- Concentric Multi-Scale RoI extraction (Local, Medium, Large, Context)
- Candidate-Fusion Vision Transformer (Arm F: 4 blocks tuned, 4608-d MLP)
- Authentic 12-Layer Attention Rollout Saliency
- Clean publication styling (PDF, SVG, 600 DPI PNG)
================================================================================
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle, PathPatch
import matplotlib.patheffects as patheffects

def draw_v8_architecture():
    # Setup high-resolution figure with clean IEEE/Nature aspect ratio
    fig = plt.figure(figsize=(16, 10.5), facecolor='white', dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette (Nature / IEEE Colorblind-safe)
    c_input     = '#F4F6F9'  # Cool gray-white
    c_prep      = '#E8F1F5'  # Soft steel blue
    c_ukan      = '#E6F4EA'  # Clinical light mint
    c_props     = '#FFF8E7'  # Warm amber light
    c_vit       = '#EDE7F6'  # Elegant lavender
    c_fusion    = '#E0F2F1'  # Soft teal
    c_output    = '#FDEED9'  # Soft coral
    
    b_border    = '#2C3E50'  # Deep slate navy
    b_prep      = '#1F4E79'  # Deep navy
    b_ukan      = '#137333'  # Deep forest green
    b_props     = '#B06000'  # Deep amber
    b_vit       = '#5E35B1'  # Deep violet
    b_fusion    = '#00695C'  # Deep teal
    b_output    = '#C0392B'  # Deep red-crimson

    def add_box(x, y, w, h, fill_col, border_col, title, subtitle=None, badge=None, lw=1.6):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.2",
            facecolor=fill_col,
            edgecolor=border_col,
            linewidth=lw,
            zorder=3
        )
        ax.add_patch(box)
        
        # Text positioning
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center',
                    fontsize=10.5, fontweight='bold', color=border_col, zorder=4)
            ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center',
                    fontsize=8.2, color='#333333', zorder=4)
        else:
            ax.text(x + w/2, y + h*0.5, title, ha='center', va='center',
                    fontsize=10.5, fontweight='bold', color=border_col, zorder=4)
            
        if badge:
            badge_box = FancyBboxPatch(
                (x + w - 4.5, y + h - 1.2), 4.2, 2.0,
                boxstyle="round,pad=0.2,rounding_size=0.6",
                facecolor=border_col, edgecolor='none', zorder=5
            )
            ax.add_patch(badge_box)
            ax.text(x + w - 2.4, y + h - 0.2, badge, ha='center', va='center',
                    fontsize=6.8, fontweight='bold', color='white', zorder=6)

    def draw_arrow(x1, y1, x2, y2, label=None, col='#2C3E50', lw=1.6, rad=0.0):
        style = "Simple,tail_width=0.8,head_width=4.5,head_length=5.5"
        kw = dict(arrowstyle=style, color=col, linewidth=lw, zorder=2)
        if rad == 0.0:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=kw)
        else:
            kw["connectionstyle"] = f"arc3,rad={rad}"
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=kw)
            
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 1.2, label, ha='center', va='bottom',
                    fontsize=7.8, fontweight='semibold', color=col, zorder=5,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))

    # =========================================================================
    # Header Banner
    # =========================================================================
    header_box = FancyBboxPatch(
        (2, 91), 96, 7.5,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        facecolor='#1B2A4A', edgecolor='none', zorder=2
    )
    ax.add_patch(header_box)
    ax.text(50, 95.8, "ADVANCED BREAST CANCER DIAGNOSTIC RESEARCH PROTOTYPE",
            ha='center', va='center', fontsize=13.5, fontweight='heavy', color='#FFFFFF')
    ax.text(50, 92.8, "Formally Frozen V8 Architecture: Adaptive Wiener Denoising + Dual-Resolution U-KAN + Multi-Scale ViT (Arm F)",
            ha='center', va='center', fontsize=9.2, color='#A9C6E2')

    # =========================================================================
    # Stage 1: Input & Patient Governance
    # =========================================================================
    add_box(3, 76, 17, 11, c_input, b_border,
            "1. CBIS-DDSM FFDM",
            "Screening / Diagnostic\n16-bit / 8-bit Gray Canvas\n235 Quarantined Patients",
            badge="INPUT")

    draw_arrow(20, 81.5, 24, 81.5, col=b_border)

    # =========================================================================
    # Stage 2: Adaptive Wiener Denoising & CLAHE
    # =========================================================================
    add_box(24, 76, 21, 11, c_prep, b_prep,
            "2. Spatial Wiener + CLAHE",
            "5x5 Adaptive Local Variance\nNoise Mottle Attenuation\nCLAHE Glandular Contrast\nMin-Max Norm [0.0, 1.0]",
            badge="DENOISE")

    draw_arrow(45, 81.5, 49, 81.5, col=b_prep)

    # =========================================================================
    # Stage 3: Dual-Resolution U-KAN Segmentation
    # =========================================================================
    add_box(49, 76, 23, 11, c_ukan, b_ukan,
            "3. Dual-Res U-KAN",
            "512x512 + 768x768 Streams\nKolmogorov-Arnold Splines\nLearnable 1D B-Splines\nFrozen Arm G Checkpoint",
            badge="U-KAN")

    draw_arrow(72, 81.5, 76, 81.5, col=b_ukan)

    # =========================================================================
    # Stage 4: Top-K Proposal Generation
    # =========================================================================
    add_box(76, 76, 21, 11, c_props, b_props,
            "4. Deterministic Proposals",
            "Machine CC (tau=0.15, A>=15)\nProb Mass + Peak Scoring\nSpatial NMS (IoU <= 0.65)\nTop K=3 Candidates",
            badge="TOP-K")

    # Connector down to Multi-scale feature generation
    draw_arrow(86.5, 76, 86.5, 68, col=b_props)

    # =========================================================================
    # Stage 5: Multi-Scale Concentric RoI Extraction (Middle Deck)
    # =========================================================================
    # Outer grouping boundary for Concentric Scales
    concentric_frame = FancyBboxPatch(
        (3, 40), 94, 27,
        boxstyle="round,pad=0.8,rounding_size=1.5",
        facecolor='#FAFBFD', edgecolor='#B0BEC5', linewidth=1.2, linestyle='--', zorder=1
    )
    ax.add_patch(concentric_frame)
    ax.text(5, 65.5, "STAGE 5: MULTI-SCALE RECEPTIVE FIELD GENERATION & CANDIDATE CROPS (K=3 PROPOSALS)",
            ha='left', va='center', fontsize=9.2, fontweight='bold', color='#455A64', zorder=2)

    # 4 Concentric Scales Boxes
    scales_info = [
        ("Local Scale (1.2x)", "Lesion Core &\nMicrocalc Interior", 5, c_props, b_props),
        ("Medium Scale (1.8x)", "Perilesional Margins\n& Micro-lobulation", 21, c_props, b_props),
        ("Large Scale (2.5x)", "Spiculation Radius\n& Parenchymal Distort", 37, c_props, b_props),
        ("Context Scale (4.0x)", "Regional Background\nParenchymal Density", 53, c_props, b_props),
        ("Calcification Patch", "High-Pass Laplacian\nFilter (224x224)", 69, '#FFF0F5', '#C2185B'),
        ("Global Mammogram", "Full Breast Canvas\nContext (224x224)", 82, '#E8EAF6', '#3949AB')
    ]

    for title, desc, x_pos, fill, border in scales_info:
        w_box = 11.5 if x_pos < 69 else (11.5 if x_pos == 69 else 13.5)
        add_box(x_pos, 49, w_box, 13.5, fill, border, title, desc, lw=1.2)
        # Arrow down into ViT Aggregator
        draw_arrow(x_pos + w_box/2, 49, x_pos + w_box/2, 41, col=border, lw=1.2)

    # =========================================================================
    # Stage 6: V8 Candidate-Fusion ViT Classifier (Lower Deck)
    # =========================================================================
    vit_frame = FancyBboxPatch(
        (3, 14), 72, 23.5,
        boxstyle="round,pad=0.8,rounding_size=1.5",
        facecolor=c_vit, edgecolor=b_vit, linewidth=1.6, zorder=2
    )
    ax.add_patch(vit_frame)
    ax.text(5, 35.8, "STAGE 6: V8 ARM F CANDIDATE-FUSION VISION TRANSFORMER (vit_base_patch16_224)",
            ha='left', va='center', fontsize=9.8, fontweight='bold', color=b_vit, zorder=4)

    # Softmax Aggregator Block
    add_box(6, 23, 21, 9.5, 'white', b_vit,
            "Candidate Attention",
            "CandidateSoftmaxAggregator\nalpha_k = Softmax(W^T e_k)\nPools K=3 -> 1x768 per scale",
            badge="4x768-d")

    # Shared Backbone Block
    add_box(29, 23, 22, 9.5, 'white', b_vit,
            "Shared ViT Backbone",
            "12 Transformer Blocks\nLast 4 Blocks Fine-Tuned (8-11)\nBlocks 0-7 Frozen",
            badge="TUNED")

    # Joint Concatenation
    add_box(53, 23, 20, 9.5, c_fusion, b_fusion,
            "Joint Latent Fusion",
            "Local + Med + Lrg + Ctx (3072)\n+ Calc (768) + Global (768)\nTotal Dimension: 4,608-d",
            badge="4608-d")

    draw_arrow(27, 27.7, 29, 27.7, col=b_vit)
    draw_arrow(51, 27.7, 53, 27.7, col=b_vit)

    # Deep MLP Head
    add_box(6, 16.5, 67, 5, 'white', b_fusion,
            "Deep MLP Head: LayerNorm(4608) -> Linear(1024) -> GELU -> Dropout(0.25) -> Linear(512) -> Linear(128) -> Linear(2)")

    # Arrow from Stage 6 to Output
    draw_arrow(75, 25.7, 78, 25.7, col=b_fusion, lw=2.0)

    # =========================================================================
    # Stage 7: Research Prediction & Attention Explainability
    # =========================================================================
    add_box(78, 14, 19, 23.5, c_output, b_output,
            "7. Research Output",
            "P(Malignant) vs P(Benign)\nLocked Acc: 64.57%\nLocked F1: 64.18%\nROC-AUC: 0.6947\nSens: 66.05% | Spec: 63.55%\n\n12-Layer Attention Rollout\nInterpretability Saliency",
            badge="LOCKED V8", lw=1.8)

    # =========================================================================
    # Bottom Governance & Scientific Integrity Strip
    # =========================================================================
    gov_box = FancyBboxPatch(
        (3, 2), 94, 9,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        facecolor='#F1F3F4', edgecolor='#78909C', linewidth=1.0, zorder=2
    )
    ax.add_patch(gov_box)
    
    ax.text(5, 8.8, "MANDATORY DATA GOVERNANCE & PROTOCOL INTEGRITY:",
            fontsize=8.5, fontweight='bold', color='#1A237E', zorder=3)
    gov_text = (
        "• Primary Test Cohort: Exactly 525 scans / 235 quarantined patients with zero patient overlap between train, validation, and test splits.\n"
        "• Checkpoint Provenance: models/final_v8/vit_v8_final.pth (MD5: 882313ce67adca01bffa699a180719c4). Zero post-hoc test retuning or retraining.\n"
        "• Scientific Status: V8 is the permanently frozen final reported model. V9 is retained as an experimental extension. Academic research prototype only."
    )
    ax.text(5, 5.0, gov_text, fontsize=7.6, color='#263238', linespacing=1.35, zorder=3)

    plt.tight_layout()
    
    # Save in required formats
    out_dir = 'results/research_paper_figures/figure_01_system_architecture'
    os.makedirs(out_dir, exist_ok=True)
    
    png_path = os.path.join(out_dir, 'fig01_system_architecture.png')
    pdf_path = os.path.join(out_dir, 'fig01_system_architecture.pdf')
    svg_path = os.path.join(out_dir, 'fig01_system_architecture.svg')

    print(f"Exporting Figure 1 to {out_dir}...")
    fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"[OK] Figure 1 successfully generated: PNG (600 DPI), PDF, SVG.")

if __name__ == '__main__':
    draw_v8_architecture()
