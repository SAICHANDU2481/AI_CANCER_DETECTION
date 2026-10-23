import hashlib
import os

ROOTS = [r'C:\Users\sys\Desktop\AI_CANCER_DETECTION', r'C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION']

paths = {
    'V1 U-KAN': 'models/ukan_best.pth',
    'V1 ViT': 'models/vit_best.pth',
    'V2-B U-KAN': 'models/v2b/segmentation/ukan_v2b_best.pth',
    'V2-A ViT': 'models/v2/vit_v2a_best.pth',
    'V3-A Stage 1': 'models/v3/checkpoints/vit_v3a_stage1_best.pth',
    'V3-A Stage 2': 'models/v3/checkpoints/vit_v3a_stage2_best.pth',
    'V3-A Best': 'models/v3/checkpoints/vit_v3a_best.pth',
    'V3-B ViT': 'models/v3b/checkpoints/vit_v3b_best.pth',
    'V4-A ViT': 'models/v4/vit_v4_best.pth'
}

for name, rel_path in paths.items():
    found = False
    for root in ROOTS:
        p = os.path.join(root, rel_path)
        if os.path.exists(p):
            h = hashlib.md5(open(p, 'rb').read()).hexdigest()
            print(f"{name:15}: {h} ({p})")
            found = True
            break
    if not found:
        print(f"{name:15}: NOT FOUND ({rel_path})")

