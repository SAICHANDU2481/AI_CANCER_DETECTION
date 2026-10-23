"""
================================================================================
Figure 4: U-KAN Lesion Segmentation Results & Error Characterization
================================================================================
Generates publication-quality multi-panel visualization of U-KAN segmentation:
- 4 Genuine Representative Clinical Cases:
  * Case 1: Malignant Spiculated Mass (High-precision correct localization)
  * Case 2: Benign Well-Circumscribed Mass (Accurate boundary delineation)
  * Case 3: Clustered Microcalcifications (Partial punctate coverage)
  * Case 4: Diffuse Benign Calcification (Challenging false negative / missed lesion)
- Columns per case:
  (1) Raw FFDM Canvas
  (2) Wiener + CLAHE Preprocessed
  (3) Ground-Truth Expert Annotation Mask
  (4) U-KAN Predicted Probability Heatmap (Threshold tau=0.15)
  (5) Multi-Contour Overlay (GT [Green] vs Prediction [Magenta])
  (6) Pixel-Level Error Decomposition (TP: Green, FP: Red, FN: Cyan)
- Displays exact calculated Dice coefficient and IoU for each case.
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
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from segmentation.v6.ukan_v6 import MultiScaleUKANV6
from app.config import UKAN_V6_PATH, SEGMENTATION_THRESHOLD, resolve_path
from app.preprocessing import preprocess_mammogram_full

def compute_dice_iou(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """Computes exact Dice and IoU between binary masks."""
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    iou = float(intersection) / float(union) if union > 0 else 0.0
    
    total_area = pred_bin.sum() + gt_bin.sum()
    dice = (2.0 * float(intersection)) / float(total_area) if total_area > 0 else 0.0
    return dice, iou

def generate_fig04_segmentation():
    # Load metadata
    metadata_path = 'data/processed/metadata.csv'
    df = pd.read_csv(metadata_path)
    
    # 4 verified case IDs across strata
    target_cases = [
        {
            'id': 'MASS_P_00015_LEFT_MLO_ab1',
            'title': 'Case 1: Malignant Spiculated Mass (Correct Localization)',
            'type': 'Malignant Mass',
            'desc': 'Strong detection & margin alignment'
        },
        {
            'id': 'MASS_P_00081_RIGHT_CC_ab1',
            'title': 'Case 2: Benign Circumscribed Mass (Correct Localization)',
            'type': 'Benign Mass',
            'desc': 'Clean boundary segmentation'
        },
        {
            'id': 'CALCIFICATION_P_00266_LEFT_CC_ab1',
            'title': 'Case 3: Clustered Pleomorphic Calcification (Partial Localization)',
            'type': 'Malignant Calc',
            'desc': 'Punctate cluster partially captured'
        },
        {
            'id': 'CALCIFICATION_P_00013_RIGHT_MLO_ab1',
            'title': 'Case 4: Diffuse Benign Microcalcification (Missed Lesion)',
            'type': 'Benign Calc',
            'desc': 'Failure case: diffuse low-contrast pattern'
        }
    ]
    
    # Setup Device and Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading frozen U-KAN model on {device}...")
    ukan = MultiScaleUKANV6(in_channels=1, out_channels=1)
    ckpt = torch.load(UKAN_V6_PATH, map_location=device, weights_only=False)
    ukan.load_state_dict(ckpt.get('model_state_dict', ckpt))
    ukan.to(device).eval()
    
    # Prepare figure: 4 rows x 6 columns
    fig, axes = plt.subplots(4, 6, figsize=(16, 11), facecolor='white', dpi=300)
    plt.subplots_adjust(wspace=0.10, hspace=0.25, left=0.04, right=0.96, top=0.94, bottom=0.05)
    
    col_headers = [
        "(a) Raw Mammogram",
        "(b) Preprocessed Input",
        "(c) Ground-Truth Mask",
        "(d) U-KAN Probability",
        "(e) Contour Overlay",
        "(f) Pixel Error Map"
    ]
    
    for c, header in enumerate(col_headers):
        axes[0, c].text(0.5, 1.14, header, transform=axes[0, c].transAxes,
                        ha='center', va='bottom', fontsize=9.2, fontweight='bold', color='#1A237E')

    for r, case_info in enumerate(target_cases):
        cid = case_info['id']
        sub = df[df['case_id'] == cid]
        if len(sub) == 0:
            print(f"Warning: case {cid} not found in metadata. Skipping.")
            continue
        row = sub.iloc[0]
        
        # Resolve filepaths (check Desktop and OneDrive)
        img_p = row['image_path']
        mask_p = row['mask_path']
        
        # Replace drive path if needed
        for prefix in [r'C:\Users\sys\Desktop\Capstone\datasets\jpeg', r'C:\Users\sys\OneDrive\Desktop\Capstone\datasets\jpeg']:
            tail = img_p.split('jpeg')[-1].lstrip('\\/')
            cand_img = os.path.join(prefix, tail)
            if os.path.exists(cand_img):
                img_p = cand_img
                break
                
        for prefix in [r'C:\Users\sys\Desktop\Capstone\datasets\jpeg', r'C:\Users\sys\OneDrive\Desktop\Capstone\datasets\jpeg']:
            tail = mask_p.split('jpeg')[-1].lstrip('\\/')
            cand_mask = os.path.join(prefix, tail)
            if os.path.exists(cand_mask):
                mask_p = cand_mask
                break
                
        raw_bgr = cv2.imread(img_p)
        mask_raw = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        
        if raw_bgr is None or mask_raw is None:
            print(f"Could not load image/mask for {cid}")
            continue
            
        raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        
        # Preprocessing
        prep = preprocess_mammogram_full(raw_gray)
        img_preproc = prep["input_512"]
        img_768 = prep["input_768"]
        
        # Resize GT mask to 512x512
        gt_512 = cv2.resize(mask_raw, (512, 512), interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt_512 > 127).astype(np.uint8)
        
        # Run U-KAN Forward Pass
        t_in = torch.from_numpy(img_768).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.inference_mode():
            logits_768 = ukan(t_in)
            prob_768 = torch.sigmoid(logits_768).squeeze().float().cpu().numpy()
            
        prob_512 = cv2.resize(prob_768, (512, 512), interpolation=cv2.INTER_LINEAR)
        pred_bin = (prob_512 >= SEGMENTATION_THRESHOLD).astype(np.uint8)
        
        # Compute exact Dice & IoU
        dice, iou = compute_dice_iou(pred_bin, gt_bin)
        print(f"[{cid}] Dice: {dice:.4f}, IoU: {iou:.4f}")
        
        # Subplot 1: Raw Mammogram
        ax1 = axes[r, 0]
        ax1.imshow(prep["original"], cmap='bone')
        ax1.axis('off')
        # Row identifier
        ax1.text(-0.15, 0.5, f"Case {r+1}\n{case_info['type']}", transform=ax1.transAxes,
                 ha='right', va='center', fontsize=8.8, fontweight='bold', color='#2C3E50', rotation=90)
        
        # Subplot 2: Preprocessed Image (Wiener + CLAHE)
        ax2 = axes[r, 1]
        ax2.imshow(img_preproc, cmap='bone')
        ax2.axis('off')
        
        # Subplot 3: Ground-Truth Lesion Mask
        ax3 = axes[r, 2]
        ax3.imshow(gt_bin, cmap='gray', vmin=0, vmax=1)
        ax3.axis('off')
        gt_area = int(gt_bin.sum())
        ax3.text(0.5, 0.05, f"GT Area: {gt_area:,} px", transform=ax3.transAxes,
                 ha='center', va='bottom', fontsize=7.5, color='yellow',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
        
        # Subplot 4: U-KAN Predicted Probability Heatmap
        ax4 = axes[r, 3]
        im4 = ax4.imshow(prob_512, cmap='viridis', vmin=0, vmax=1)
        ax4.axis('off')
        ax4.text(0.5, 0.05, f"Max Prob: {prob_512.max():.2f}", transform=ax4.transAxes,
                 ha='center', va='bottom', fontsize=7.5, color='white',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
        
        # Subplot 5: Overlay (GT in Cyan/Green, Pred in Magenta)
        ax5 = axes[r, 4]
        overlay = cv2.cvtColor((img_preproc * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # GT contours (Green)
        gt_contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, gt_contours, -1, (0, 255, 128), 2)
        # Pred contours (Magenta)
        pred_contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, pred_contours, -1, (255, 0, 255), 2)
        
        ax5.imshow(overlay)
        ax5.axis('off')
        ax5.text(0.5, 0.05, f"Dice: {dice:.3f} | IoU: {iou:.3f}", transform=ax5.transAxes,
                 ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='white',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#1A237E', alpha=0.85))

        # Subplot 6: Pixel Error Map (TP=Green, FP=Red, FN=Cyan, TN=Dark)
        ax6 = axes[r, 5]
        error_map = np.zeros((512, 512, 3), dtype=np.uint8)
        # TN is background (dark slate gray from image)
        bg = (img_preproc * 50).astype(np.uint8)
        error_map[:, :, 0] = bg
        error_map[:, :, 1] = bg
        error_map[:, :, 2] = bg
        
        tp = np.logical_and(pred_bin == 1, gt_bin == 1)
        fp = np.logical_and(pred_bin == 1, gt_bin == 0)
        fn = np.logical_and(pred_bin == 0, gt_bin == 1)
        
        error_map[tp] = [46, 204, 113]   # TP: Vibrant Emerald Green
        error_map[fp] = [231, 76, 60]    # FP: Coral Red
        error_map[fn] = [52, 152, 219]   # FN: Sky Blue
        
        ax6.imshow(error_map)
        ax6.axis('off')
        
        status_text = "HIT" if iou >= 0.10 else "MISS"
        status_col = '#2ECC71' if status_text == 'HIT' else '#E74C3C'
        ax6.text(0.5, 0.05, f"Status: {status_text}", transform=ax6.transAxes,
                 ha='center', va='bottom', fontsize=8.0, fontweight='bold', color='white',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor=status_col, alpha=0.9))

    # Shared Legend at the bottom
    legend_elements = [
        Patch(facecolor='#2ECC71', edgecolor='#27AE60', label='True Positive (TP) Pixels'),
        Patch(facecolor='#E74C3C', edgecolor='#C0392B', label='False Positive (FP) Over-segmentation'),
        Patch(facecolor='#3498DB', edgecolor='#2980B9', label='False Negative (FN) Under-segmentation'),
        Patch(facecolor='none', edgecolor='#00FF80', linewidth=2, label='Ground Truth Boundary (Green)'),
        Patch(facecolor='none', edgecolor='#FF00FF', linewidth=2, label='U-KAN Predicted Boundary (Magenta)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, frameon=True,
               facecolor='#F8F9FA', edgecolor='#CFD8DC', fontsize=8.5, bbox_to_anchor=(0.5, 0.01))

    out_dir = 'results/research_paper_figures/figure_04_segmentation'
    os.makedirs(out_dir, exist_ok=True)
    
    png_path = os.path.join(out_dir, 'fig04_ukan_segmentation_evaluation.png')
    pdf_path = os.path.join(out_dir, 'fig04_ukan_segmentation_evaluation.pdf')
    svg_path = os.path.join(out_dir, 'fig04_ukan_segmentation_evaluation.svg')
    
    print(f"Exporting Figure 4 to {out_dir}...")
    fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("[OK] Figure 4 successfully generated: PNG (600 DPI), PDF, SVG.")

if __name__ == '__main__':
    generate_fig04_segmentation()
