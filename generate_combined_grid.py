import os
import sys

# Ensure local workspace is on sys.path
sys.path.insert(0, os.path.abspath("."))

import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch

from segmentation.ukan import build_ukan_from_config
from utils.metrics import calculate_segmentation_metrics
from preprocessing.enhancement import enhance_mammogram

def main():
    vis_dir = 'results/segmentation/visualizations'
    os.makedirs(vis_dir, exist_ok=True)

    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_ukan_from_config(cfg).to(device)
    ckpt = torch.load('models/ukan_best.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    df = pd.read_csv('data/processed/verified_test.csv')

    selected_cases = [
        {
            'case_id': 'MASS_P_00226_LEFT_MLO_ab1',
            'title': '1. Malignant Mass (P_00226, L-MLO)'
        },
        {
            'case_id': 'MASS_P_00453_LEFT_MLO_ab1',
            'title': '2. Benign Mass (P_00453, L-MLO)'
        },
        {
            'case_id': 'CALCIFICATION_P_00063_RIGHT_MLO_ab1',
            'title': '3. Malignant Calcification (P_00063, R-MLO)'
        },
        {
            'case_id': 'CALCIFICATION_P_01660_RIGHT_MLO_ab1',
            'title': '4. Benign Calcification (P_01660, R-MLO)'
        },
        {
            'case_id': 'MASS_P_00146_RIGHT_CC_ab1',
            'title': '5. Malignant Mass (P_00146, R-CC)'
        },
        {
            'case_id': 'CALCIFICATION_P_00357_LEFT_CC_ab1',
            'title': '6. Benign Calcification (P_00357, L-CC)'
        }
    ]

    def create_overlay(base_img, mask, color=(0, 255, 0), alpha=0.45):
        rgb = np.stack([base_img, base_img, base_img], axis=-1)
        rgb = (rgb * 255).astype(np.uint8)
        mask_u8 = (mask > 0.5).astype(np.uint8)
        
        color_mask = np.zeros_like(rgb)
        for c in range(3):
            color_mask[:, :, c] = color[c]
        
        overlay = rgb.copy()
        idx = mask_u8 > 0
        if np.any(idx):
            overlay[idx] = cv2.addWeighted(rgb[idx], 1 - alpha, color_mask[idx], alpha, 0)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
            
        return overlay

    fig, axes = plt.subplots(6, 6, figsize=(22, 22), dpi=200)
    fig.patch.set_facecolor('#0b1120')

    col_titles = [
        'Raw Mammogram',
        'Wiener + CLAHE',
        'Ground-Truth Mask',
        'U-KAN Prediction (p >= 0.5)',
        'Ground-Truth Overlay',
        'Prediction Overlay'
    ]

    for col_idx, col_title in enumerate(col_titles):
        axes[0, col_idx].set_title(col_title, color='#38bdf8', fontsize=13, fontweight='bold', pad=12)

    for row_idx, case_info in enumerate(selected_cases):
        cid = case_info['case_id']
        row = df[df['case_id'] == cid].iloc[0]
        
        raw_img_path = row['image_path'].replace('OneDrive\\Desktop', 'Desktop')
        raw_mask_path = row['mask_path'].replace('OneDrive\\Desktop', 'Desktop')
        
        pil_raw = Image.open(raw_img_path).convert('L')
        raw_512 = np.array(pil_raw.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        
        cache_path = f'data/cache/segmentation_512/{cid}.npz'
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            prep_img = cached['image']
            gt_mask = cached['mask']
        else:
            prep_dict = enhance_mammogram(raw_512, config=cfg.get('preprocessing', {}))
            prep_img = prep_dict['enhanced']
            pil_mask = Image.open(raw_mask_path).convert('L')
            gt_mask = (np.array(pil_mask.resize((512, 512), Image.Resampling.NEAREST)) > 127).astype(np.float32)
            
        tensor_in = torch.from_numpy(prep_img).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(tensor_in)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_bin = (prob >= 0.5).astype(np.float32)
            
        m = calculate_segmentation_metrics(prob, gt_mask, threshold=0.5)
        dice_val = m['dice']
        iou_val = m['iou']
        
        gt_overlay = create_overlay(prep_img, gt_mask, color=(0, 230, 115), alpha=0.45)
        pred_overlay = create_overlay(prep_img, pred_bin, color=(255, 75, 75), alpha=0.45)
        
        imgs = [
            (raw_512, 'gray'),
            (prep_img, 'gray'),
            (gt_mask, 'viridis'),
            (pred_bin, 'magma'),
            (gt_overlay, None),
            (pred_overlay, None)
        ]
        
        for c_idx, (img_data, cmap) in enumerate(imgs):
            ax = axes[row_idx, c_idx]
            ax.set_facecolor('#1e293b')
            if cmap is not None:
                ax.imshow(img_data, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(img_data)
            ax.axis('off')
            
            if c_idx == 3:
                ax.text(0.5, -0.06, f'Dice: {dice_val:.3f} | IoU: {iou_val:.3f}',
                        transform=ax.transAxes, color='#facc15', ha='center', va='top', fontsize=9, fontweight='bold')
                
        row_label = f"{case_info['title']}\n({row['pathology']})"
        axes[row_idx, 0].text(-0.08, 0.5, row_label,
                              transform=axes[row_idx, 0].transAxes, color='#f1f5f9',
                              ha='right', va='center', fontsize=11, fontweight='bold', rotation=0)

    fig.suptitle(
        'U-KAN Lesion Segmentation on Held-Out CBIS-DDSM Test Cohort\nRepresentative 6-Case Multi-Panel Benchmark (Model: models/ukan_best.pth | Epoch 10)',
        color='#f8fafc', fontsize=16, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0.14, 0.02, 0.98, 0.95])

    grid_path = os.path.join(vis_dir, 'ukan_test_qualitative_grid.png')
    plt.savefig(grid_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved combined grid figure: {grid_path}')

if __name__ == '__main__':
    main()
