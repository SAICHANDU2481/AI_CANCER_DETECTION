"""
Unit test for V5 High-Resolution MultiScaleUKAN architecture across resolutions.
"""

import sys
import os
import torch

PROJECT_ROOT = r'C:\Users\sys\Desktop\AI_CANCER_DETECTION'
ONEDRIVE_ROOT = r'C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION'
for r in [PROJECT_ROOT, ONEDRIVE_ROOT]:
    if r not in sys.path and os.path.exists(r):
        sys.path.insert(0, r)

from segmentation.ukan_v5 import MultiScaleUKANV5
from segmentation.losses_v5 import get_segmentation_loss_v5


def test_ukan_v5():
    print("Testing MultiScaleUKANV5 Architecture...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = MultiScaleUKANV5(in_channels=1, out_channels=1, init_features=32).to(device)
    loss_fn = get_segmentation_loss_v5('focal_dice')

    # Test 512x512
    x_512 = torch.randn(2, 1, 512, 512, device=device)
    target_512 = torch.randint(0, 2, (2, 1, 512, 512), device=device).float()
    logits_512 = model(x_512)
    assert logits_512.shape == (2, 1, 512, 512), f"Expected (2, 1, 512, 512), got {logits_512.shape}"
    loss_512 = loss_fn(logits_512, target_512)
    loss_512.backward()
    print("  [PASS] 512x512 forward & backward pass passed. Loss:", float(loss_512.item()))

    # Test 768x768
    model.zero_grad()
    x_768 = torch.randn(1, 1, 768, 768, device=device)
    target_768 = torch.randint(0, 2, (1, 1, 768, 768), device=device).float()
    logits_768 = model(x_768)
    assert logits_768.shape == (1, 1, 768, 768), f"Expected (1, 1, 768, 768), got {logits_768.shape}"
    loss_768 = loss_fn(logits_768, target_768)
    loss_768.backward()
    print("  [PASS] 768x768 forward & backward pass passed. Loss:", float(loss_768.item()))

    print("MultiScaleUKANV5 Verification: 100% SUCCESS")


if __name__ == '__main__':
    test_ukan_v5()
