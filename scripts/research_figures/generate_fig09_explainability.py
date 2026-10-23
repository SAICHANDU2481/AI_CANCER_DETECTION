"""
================================================================================
Figure 9: Vision Transformer Self-Attention Rollout & Interpretability
================================================================================
Generates publication-quality interpretability visualizations using authentic
12-layer Attention Rollout extracted directly from the frozen V8 ViT backbone:
- Evaluated on verified training/benchmark cases (preserving test set quarantine):
  * Case 1: Malignant Spiculated Mass (MASS_P_00001_LEFT_CC)
  * Case 2: Benign Circumscribed Mass (MASS_P_00004_LEFT_CC)
- Visualization Components:
  (a) Global FFDM Mammogram with RoI Proposal Bounding Box
  (b) Candidate RoI Crop (224x224 input space)
  (c) 12-Layer Recursive Attention Rollout Saliency Map
  (d) Blended Saliency Overlay Highlighting Morphological Margins
  (e) Multi-Scale Attention Concentration (Local 1.2x vs Context 4.0x)
  (f) Mathematical Formulation & Scientific Governance Protocol
- Explicit Scientific Disclaimer:
  "Self-attention maps illustrate model token weighting patterns and do NOT
   imply causal biological or clinical diagnostic reasoning."
- Exports: PDF vector, SVG vector, 600 DPI PNG
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
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

from app.inference import load_v8_pipeline_models, extract_vit_attention_map, to_vit_tensor
from app.config import DEMO_SCANS, resolve_path
from app.preprocessing import preprocess_mammogram_full

def generate_fig09_explainability():
    # Load frozen V8 model
    models, err = load_v8_pipeline_models()
    if err:
        raise RuntimeError(f"Failed to load V8 models: {err}")
    device = models.device
    print(f"Loaded frozen V8 ViT backbone on {device} (MD5 Verified).")
    
    # 2 Benchmark cases from DEMO_SCANS
    cases = [
        {
            'name': 'Case 1: Malignant Spiculated Mass',
            'case_id': 'MASS_P_00001_LEFT_CC',
            'path': DEMO_SCANS['Case 1: MASS_P_00001_LEFT_CC (Malignant Spiculated Mass)']['image_path'],
            'roi_box': (500, 1500, 300, 1300), # y1, y2, x1, x2 approx
            'type': 'Malignant (Spiculated Margins)',
            'desc': 'Attention concentrates on hyper-dense core and radiating spicules.'
        },
        {
            'name': 'Case 2: Benign Circumscribed Mass',
            'case_id': 'MASS_P_00004_LEFT_CC',
            'path': DEMO_SCANS['Case 2: MASS_P_00004_LEFT_CC (Benign Circumscribed Mass)']['image_path'],
            'roi_box': (1000, 2000, 400, 1400),
            'type': 'Benign (Circumscribed Fibroadenoma)',
            'desc': 'Attention distributes smoothly across homogeneous lesion interior.'
        }
    ]
    
    # Styling configuration
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 8.5
    
    fig = plt.figure(figsize=(16, 11), facecolor='white', dpi=300)
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 1.4], wspace=0.20, hspace=0.30,
                           left=0.04, right=0.96, top=0.92, bottom=0.06)
    
    headers = [
        "(a) Global Mammogram",
        "(b) RoI Candidate Crop",
        "(c) Attention Rollout",
        "(d) Saliency Overlay",
        "(e) Context Scale (4.0x)",
        "(f) Methodology & Governance"
    ]
    for c, h in enumerate(headers):
        ax_top = fig.add_subplot(gs[0, c])
        ax_top.text(0.5, 1.15, h, transform=ax_top.transAxes,
                    ha='center', va='bottom', fontsize=9.2, fontweight='bold', color='#1B2A4A')
        ax_top.remove() # remove dummy subplot
        
    for r, case in enumerate(cases):
        img_p = resolve_path(case['path'])
        raw_bgr = cv2.imread(img_p)
        if raw_bgr is None:
            # Fallback to secondary path
            alt = img_p.replace('Desktop', 'OneDrive\\Desktop') if 'OneDrive' not in img_p else img_p.replace('OneDrive\\', '')
            raw_bgr = cv2.imread(alt)
            
        raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        h_orig, w_orig = raw_gray.shape
        
        # Preprocessing
        prep = preprocess_mammogram_full(raw_gray)
        img_512 = prep["input_512"]
        
        # Run U-KAN to find actual candidate RoI
        t_in_768 = torch.from_numpy(prep["input_768"]).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.inference_mode():
            logits_768 = models.ukan(t_in_768)
            prob_768 = torch.sigmoid(logits_768).squeeze().float().cpu().numpy()
        prob_512 = cv2.resize(prob_768, (512, 512), interpolation=cv2.INTER_LINEAR)
        proposals, _, _ = models.proposal_gen.generate_proposals(img_512, prob_512, k=1)
        top_prop = proposals[0]
        
        # Multi-scale crops
        crops, bboxes = models.roi_gen.generate_candidate_crops(img_512, top_prop)
        local_crop = crops['local']
        context_crop = crops['context']
        
        # 12-layer attention rollout extraction on local crop
        t_local = to_vit_tensor(local_crop.astype(np.float32)).unsqueeze(0).to(device)
        attn_map = extract_vit_attention_map(models.vit.backbone, t_local, device)
        
        # 12-layer attention rollout on context crop
        t_ctx = to_vit_tensor(context_crop.astype(np.float32)).unsqueeze(0).to(device)
        attn_ctx = extract_vit_attention_map(models.vit.backbone, t_ctx, device)
        
        # Color overlay
        uint8_crop = (local_crop * 255.0).astype(np.uint8)
        rgb_crop = cv2.cvtColor(uint8_crop, cv2.COLOR_GRAY2RGB)
        
        heatmap_colored = cv2.applyColorMap((attn_map * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(rgb_crop, 0.55, heatmap_colored, 0.45, 0)
        
        # Subplot 1: Full Mammogram with RoI Bounding Box
        ax1 = fig.add_subplot(gs[r, 0])
        ax1.imshow(prep["original"], cmap='bone')
        # Map proposal box back to 512 coordinate space
        bx, by, bw, bh = top_prop['bbox']
        rect = Rectangle((bx, by), bw, bh, edgecolor='#E74C3C', facecolor='none', lw=1.8, ls='-')
        ax1.add_patch(rect)
        ax1.axis('off')
        ax1.text(-0.18, 0.5, f"{case['name'].split(':')[0]}\n{case['type'].split('(')[0].strip()}",
                 transform=ax1.transAxes, ha='right', va='center', fontsize=8.2, fontweight='bold',
                 color='#1B2A4A', rotation=90)
        ax1.text(0.5, 0.04, f"{case['case_id']}", transform=ax1.transAxes, ha='center',
                 fontsize=6.8, color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
        
        # Subplot 2: Candidate RoI Crop (Local 1.2x)
        ax2 = fig.add_subplot(gs[r, 1])
        ax2.imshow(local_crop, cmap='bone')
        ax2.axis('off')
        ax2.text(0.5, 0.04, "Local RoI (1.2x Scale)", transform=ax2.transAxes, ha='center',
                 fontsize=7.2, color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor='#1B2A4A', alpha=0.7))
        
        # Subplot 3: Attention Rollout Heatmap
        ax3 = fig.add_subplot(gs[r, 2])
        im3 = ax3.imshow(attn_map, cmap='turbo', vmin=0, vmax=1)
        ax3.axis('off')
        ax3.text(0.5, 0.04, "12-Layer Rollout", transform=ax3.transAxes, ha='center',
                 fontsize=7.2, color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Subplot 4: Blended Overlay
        ax4 = fig.add_subplot(gs[r, 3])
        ax4.imshow(blended)
        ax4.axis('off')
        ax4.text(0.5, 0.04, "Morphology Overlay", transform=ax4.transAxes, ha='center',
                 fontsize=7.2, color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor='#1B2A4A', alpha=0.85))

        # Subplot 5: Context Scale (4.0x) Attention Concentration
        ax5 = fig.add_subplot(gs[r, 4])
        uint8_ctx = (context_crop * 255.0).astype(np.uint8)
        rgb_ctx = cv2.cvtColor(uint8_ctx, cv2.COLOR_GRAY2RGB)
        heat_ctx = cv2.applyColorMap((attn_ctx * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        heat_ctx = cv2.cvtColor(heat_ctx, cv2.COLOR_BGR2RGB)
        blended_ctx = cv2.addWeighted(rgb_ctx, 0.55, heat_ctx, 0.45, 0)
        ax5.imshow(blended_ctx)
        ax5.axis('off')
        ax5.text(0.5, 0.04, "Context RoI (4.0x)", transform=ax5.transAxes, ha='center',
                 fontsize=7.2, color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor='#1B2A4A', alpha=0.85))

    # Column 6: Formulation & Scientific Governance Card (Spanning both rows)
    ax6 = fig.add_subplot(gs[:, 5])
    ax6.axis('off')
    
    theory_text = (
        "ATTENTION ROLLOUT FORMULATION\n"
        "====================================\n\n"
        "1. Layer Attention Matrix:\n"
        "   A^(l) = Softmax( Q^(l) K^(l)T / sqrt(d) )\n"
        "   Aggregated across H=12 attention heads\n"
        "   via mean reduction.\n\n"
        "2. Residual Identity Accounting:\n"
        "   A_hat^(l) = 0.5 * A^(l) + 0.5 * I\n"
        "   Where identity matrix I represents the\n"
        "   Transformer residual skip connection.\n\n"
        "3. Recursive 12-Layer Rollout:\n"
        "   R^(l) = A_hat^(l) * R^(l-1)\n"
        "   Accumulates token interactions from\n"
        "   Patch Embedding (Layer 0) through the\n"
        "   fine-tuned Blocks (Layers 8-11).\n\n"
        "4. Class Token Saliency Extraction:\n"
        "   S_patch = R^(12)[CLS, 1:197]\n"
        "   Interpolated to 224x224 input canvas.\n\n"
        "====================================\n"
        "MANDATORY SCIENTIFIC DISCLAIMER:\n"
        "------------------------------------\n"
        "• Attention maps are descriptive representations\n"
        "  of self-attention token weights.\n"
        "• High saliency indicates feature correlation\n"
        "  within the ViT receptive field, NOT clinical\n"
        "  or pathological causation.\n"
        "• AI-assisted research prototype only."
    )
    
    ax6.text(0.04, 0.98, theory_text, transform=ax6.transAxes, va='top', ha='left',
             fontsize=7.4, family='monospace', linespacing=1.28,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#F8F9FA', edgecolor='#78909C', lw=1.2))

    out_dir = 'results/research_paper_figures/figure_09_explainability'
    os.makedirs(out_dir, exist_ok=True)
    
    png_path = os.path.join(out_dir, 'fig09_attention_explainability.png')
    pdf_path = os.path.join(out_dir, 'fig09_attention_explainability.pdf')
    svg_path = os.path.join(out_dir, 'fig09_attention_explainability.svg')
    
    print(f"Exporting Figure 9 to {out_dir}...")
    fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[OK] Figure 9 successfully generated: PNG (600 DPI), PDF, SVG.")

if __name__ == '__main__':
    generate_fig09_explainability()
