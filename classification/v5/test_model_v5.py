"""
Unit test for V5 MultiScaleViTV5 across K=1, 3, 5 and aggregation modes.
"""

import sys
import os
import torch

PROJECT_ROOT = r'C:\Users\sys\Desktop\AI_CANCER_DETECTION'
ONEDRIVE_ROOT = r'C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION'
for r in [PROJECT_ROOT, ONEDRIVE_ROOT]:
    if r not in sys.path and os.path.exists(r):
        sys.path.insert(0, r)

from classification.v5.model_v5 import MultiScaleViTV5


def test_vit_v5():
    print("Testing MultiScaleViTV5 Architecture...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    scales = ('local', 'medium', 'large', 'context')
    B = 2

    for k in [1, 3, 5]:
        for agg_mode in ['attention_weighted', 'max_prob', 'learned_pooling']:
            print(f"  Testing K={k}, Aggregation={agg_mode}...")
            model = MultiScaleViTV5(
                use_full_image=True,
                scales_used=scales,
                aggregation_mode=agg_mode,
                pretrained=False,
                k=k
            ).to(device)

            candidate_dict = {}
            for s in scales:
                candidate_dict[s] = [torch.randn(B, 3, 224, 224, device=device) for _ in range(k)]

            full_img = torch.randn(B, 3, 224, 224, device=device)
            target = torch.randint(0, 2, (B,), device=device)

            logits = model(candidate_dict, full_img)
            assert logits.shape == (B, 2), f"Expected shape (B, 2), got {logits.shape}"
            loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            print(f"    [PASS] K={k}, Agg={agg_mode} logits={logits.shape}, loss={float(loss.item()):.4f}")

    print("MultiScaleViTV5 Verification: 100% SUCCESS")


if __name__ == '__main__':
    test_vit_v5()
