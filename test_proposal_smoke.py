import os
import sys
import cv2
import torch
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from v7_stage0_base import resolve_image_path, FROZEN_UKAN_PATH
from segmentation.v6.ukan_v6 import build_ukan_v6
from v7_proposal_generators import (
    extract_breast_bounding_box, V6UKANProposalGenerator, HighResUKANProposalGenerator,
    CalcificationProposalGenerator, SaliencyProposalGenerator, ContextualCandidateBank,
    RandomCropBaseline, FixedGridBaseline, non_max_suppression_proposals
)

def main():
    val_df = pd.read_csv('data/processed/val.csv')
    model = build_ukan_v6(use_detail_branch=True, use_multi_scale_blocks=True, deep_supervision=True)
    ckpt = torch.load(FROZEN_UKAN_PATH, map_location='cpu')
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    gen_a = V6UKANProposalGenerator(model)
    gen_b = HighResUKANProposalGenerator(model, target_res=768)
    gen_c = CalcificationProposalGenerator(mode='all')
    gen_d = SaliencyProposalGenerator()
    gen_e = ContextualCandidateBank()
    gen_f = RandomCropBaseline(seed=42)
    gen_g = FixedGridBaseline()

    for i in range(3):
        row = val_df.iloc[i]
        case_id = str(row['case_id'])
        img_p = resolve_image_path(str(row['image_path']))
        mask_p = resolve_image_path(str(row.get('mask_path', '')))
        
        img_bgr = cv2.imread(img_p)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        props_a = gen_a.generate(img_gray, top_k=5)
        props_b = gen_b.generate(img_gray, top_k=5)
        props_c = gen_c.generate(img_gray, top_k=5)
        props_d = gen_d.generate(img_gray, top_k=5)
        props_e = gen_e.generate(img_gray, top_k=5)
        props_f = gen_f.generate(img_gray, top_k=5)
        props_g = gen_g.generate(img_gray, top_k=5)
        
        combined = non_max_suppression_proposals(props_a + props_b + props_c + props_d + props_e, top_k=5)
        print(f"Case {i} ({case_id}): A={len(props_a)}, B={len(props_b)}, C={len(props_c)}, D={len(props_d)}, E={len(props_e)} -> Combined={len(combined)}")
        for j, p in enumerate(combined):
            print(f"  Top-{j+1}: {p['proposal_source']}, Score={p['score']:.3f}, BBox={p['bbox']}")
            
    print("Smoke test successfully completed!")

if __name__ == '__main__':
    main()
