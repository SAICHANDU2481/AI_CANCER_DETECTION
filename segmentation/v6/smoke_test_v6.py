"""
================================================================================
V6 Segmentation Smoke Test: Validates all 7 Pilot Ablation Arms
================================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast

# Ensure project root is in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from segmentation.v6.ukan_v6 import build_ukan_v6
from segmentation.v6.losses_v6 import (
    BCEDiceLoss, FocalDiceLoss, TverskyDiceLoss, FocalTverskyBoundaryLoss,
    DeepSupervisedCompositeLoss, SmallLesionWeightedLoss
)
from segmentation.v6.dataset_v6 import get_v6_segmentation_dataloaders

def run_smoke_test():
    print("=" * 80)
    print("STARTING V6 SEGMENTATION PILOT ARMS SMOKE TEST")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Train and Val dataframes
    train_df = pd.read_csv("data/processed/train.csv").head(20)
    val_df = pd.read_csv("data/processed/val.csv").head(10)

    train_loader, val_loader = get_v6_segmentation_dataloaders(
        train_df, val_df, target_size=512, batch_size=2, num_workers=0
    )

    sample_batch = next(iter(train_loader))
    imgs = sample_batch['image'].to(device)
    masks = sample_batch['mask'].to(device)
    print(f"Sample batch loaded: Images {imgs.shape}, Masks {masks.shape}, Range [{imgs.min():.2f}, {imgs.max():.2f}]")

    # Define 7 pilot ablation configurations
    arms = [
        {
            "arm_id": "Arm_A",
            "name": "Baseline U-KAN",
            "model": build_ukan_v6(use_detail_branch=False, use_multi_scale_blocks=False, deep_supervision=False),
            "loss": BCEDiceLoss()
        },
        {
            "arm_id": "Arm_B",
            "name": "V6 Multi-Scale U-KAN",
            "model": build_ukan_v6(use_detail_branch=False, use_multi_scale_blocks=True, deep_supervision=False),
            "loss": BCEDiceLoss()
        },
        {
            "arm_id": "Arm_C",
            "name": "V6 + Deep Supervision",
            "model": build_ukan_v6(use_detail_branch=False, use_multi_scale_blocks=True, deep_supervision=True),
            "loss": DeepSupervisedCompositeLoss(base_loss=BCEDiceLoss(), w_main=0.60, w_aux1=0.25, w_aux2=0.15)
        },
        {
            "arm_id": "Arm_D",
            "name": "V6 + Small-Lesion-Aware Loss",
            "model": build_ukan_v6(use_detail_branch=False, use_multi_scale_blocks=True, deep_supervision=False),
            "loss": SmallLesionWeightedLoss(base_loss=TverskyDiceLoss(alpha=0.3, beta=0.7))
        },
        {
            "arm_id": "Arm_E",
            "name": "V6 + Dedicated Detail Branch",
            "model": build_ukan_v6(use_detail_branch=True, use_multi_scale_blocks=True, deep_supervision=False),
            "loss": BCEDiceLoss()
        },
        {
            "arm_id": "Arm_F",
            "name": "V6 + Deep Supervision + Detail Branch",
            "model": build_ukan_v6(use_detail_branch=True, use_multi_scale_blocks=True, deep_supervision=True),
            "loss": DeepSupervisedCompositeLoss(base_loss=FocalDiceLoss(), w_main=0.60, w_aux1=0.25, w_aux2=0.15)
        },
        {
            "arm_id": "Arm_G",
            "name": "V6 Full + Calcification Focus",
            "model": build_ukan_v6(use_detail_branch=True, use_multi_scale_blocks=True, deep_supervision=True),
            "loss": DeepSupervisedCompositeLoss(
                base_loss=FocalTverskyBoundaryLoss(focal_weight=0.35, tversky_weight=0.45, boundary_weight=0.20, beta_tversky=0.75),
                w_main=0.60, w_aux1=0.25, w_aux2=0.15
            )
        }
    ]

    for arm in arms:
        arm_id = arm["arm_id"]
        name = arm["name"]
        model = arm["model"].to(device)
        criterion = arm["loss"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        model.train()
        optimizer.zero_grad()

        with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(imgs)
            loss = criterion(outputs, masks)

        assert not torch.isnan(loss).item(), f"NaN in {arm_id} loss"
        assert not torch.isinf(loss).item(), f"Inf in {arm_id} loss"

        loss.backward()
        optimizer.step()

        # Validation forward test
        model.eval()
        with torch.no_grad():
            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                val_out = model(imgs)
                if isinstance(val_out, (tuple, list)):
                    val_out = val_out[0]
                val_probs = torch.sigmoid(val_out).cpu().numpy()

        assert val_probs.shape == (2, 1, 512, 512)
        params = sum(p.numel() for p in model.parameters())

        print(f"[{arm_id}] {name:<40} | Params: {params/1e6:.2f}M | Train Loss: {loss.item():.4f} | Val Prob: [{val_probs.min():.4f}, {val_probs.max():.4f}] (PASSED)")

    print("=" * 80)
    print("ALL 7 PILOT ABLATION ARMS PASSED SMOKE TEST SUCCESSFULLY (100% SUCCESS)")
    print("=" * 80)

if __name__ == "__main__":
    run_smoke_test()
