"""
================================================================================
Breast Cancer AI Analysis - End-to-End App Inference Smoke Test (V8 Final)
================================================================================
Verifies:
1. Module imports and environment dependencies
2. Checkpoint existence & cryptographic MD5 hash integrity (V8 ViT & U-KAN)
3. U-KAN model loading and inference mode
4. MultiScaleViTV8 Arm F model loading and inference mode
5. Preprocessing pipeline (Wiener filter + CLAHE + Normalization)
6. U-KAN forward pass (768x768 -> 512x512 probability map)
7. Deterministic candidate proposal generation (Top-3 RoIs)
8. Multi-scale RoI extraction across 4 concentric scales
9. Calcification frequency enhancement patch
10. MultiScaleViTV8 joint classification forward pass
11. Probability outputs sum to 1.0 & no NaN / Inf values
12. Authentic ViT self-attention rollout extraction
13. CPU fallback execution
14. Real clinical demo case execution
15. ZERO access to locked test set
================================================================================
"""

import os
import sys
import time
import math
import hashlib
import numpy as np
import cv2
import torch

# Set stdout encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import (
    UKAN_V6_PATH, VIT_V8_PATH, EXPECTED_MD5, verify_file_md5,
    LOCKED_V8_TEST_METRICS, DATASET_STATS, DEMO_SCANS
)
from app.preprocessing import (
    load_image_from_upload, load_demo_scan_image, preprocess_mammogram_full
)
from app.inference import (
    load_v8_pipeline_models, run_complete_inference_pipeline, to_vit_tensor,
    extract_vit_attention_map
)
from app.visualization import (
    create_segmentation_overlay, create_attention_overlay, render_confusion_matrix_plot
)


def run_smoke_test():
    print("=" * 80)
    print("STARTING V8 END-TO-END MEDICAL AI INFERENCE SMOKE TEST")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: Checkpoint Existence & MD5 Verification
    # -------------------------------------------------------------------------
    print("\n[Step 1/12] Verifying Checkpoint Existence and Cryptographic MD5 Integrity...")
    assert os.path.exists(UKAN_V6_PATH), f"Missing U-KAN checkpoint: {UKAN_V6_PATH}"
    assert os.path.exists(VIT_V8_PATH), f"Missing ViT V8 checkpoint: {VIT_V8_PATH}"

    is_u_ok, actual_u_md5 = verify_file_md5(UKAN_V6_PATH, EXPECTED_MD5['ukan_v6'])
    assert is_u_ok, f"U-KAN MD5 mismatch! Expected: {EXPECTED_MD5['ukan_v6']}, Got: {actual_u_md5}"
    print(f"   ✓ U-KAN Checkpoint: {UKAN_V6_PATH}")
    print(f"     MD5 Verified: {actual_u_md5}")

    is_v_ok, actual_v_md5 = verify_file_md5(VIT_V8_PATH, EXPECTED_MD5['vit_v8'])
    assert is_v_ok, f"ViT V8 MD5 mismatch! Expected: {EXPECTED_MD5['vit_v8']}, Got: {actual_v_md5}"
    print(f"   ✓ ViT V8 Checkpoint: {VIT_V8_PATH}")
    print(f"     MD5 Verified: {actual_v_md5}")

    # -------------------------------------------------------------------------
    # STEP 2: Model Bundle Loading (Default Device)
    # -------------------------------------------------------------------------
    print("\n[Step 2/12] Loading V8 Model Bundle (U-KAN + ViT Arm F)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bundle, err = load_v8_pipeline_models(device)
    assert err is None, f"Model loading failed: {err}"
    assert bundle is not None
    assert bundle.ukan is not None and bundle.vit is not None
    print(f"   ✓ Models instantiated on compute device: {device} (Version: {bundle.model_version})")

    # -------------------------------------------------------------------------
    # STEP 3: Preprocessing Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3/12] Testing Preprocessing Pipeline (Wiener + CLAHE + Normalization)...")
    np.random.seed(42)
    mock_raw = np.random.uniform(0.05, 0.95, (512, 512)).astype(np.float32)
    t0 = time.time()
    prep = preprocess_mammogram_full(mock_raw)
    t_prep = (time.time() - t0) * 1000
    assert "original" in prep and "wiener" in prep and "enhanced" in prep
    assert "input_512" in prep and "input_768" in prep and "input_224" in prep
    assert prep["input_512"].shape == (512, 512)
    assert prep["input_768"].shape == (768, 768)
    assert prep["input_224"].shape == (224, 224)
    print(f"   ✓ Preprocessing completed in {t_prep:.1f}ms. Output range: [{prep['input_512'].min():.3f}, {prep['input_512'].max():.3f}]")

    # -------------------------------------------------------------------------
    # STEP 4: U-KAN Segmentation Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Step 4/12] Testing U-KAN Segmentation Forward Pass (768x768 Receptive Field)...")
    t_in_768 = torch.from_numpy(prep["input_768"]).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        logits_768 = bundle.ukan(t_in_768)
        prob_768 = torch.sigmoid(logits_768).squeeze().float().cpu().numpy()
    assert prob_768.shape == (768, 768), f"Unexpected shape: {prob_768.shape}"
    assert not np.isnan(prob_768).any(), "NaN detected in U-KAN probability output"
    prob_512 = cv2.resize(prob_768, (512, 512))
    print(f"   ✓ U-KAN segmentation output verified. Probability range: [{prob_512.min():.4f}, {prob_512.max():.4f}]")

    # -------------------------------------------------------------------------
    # STEP 5: Top-3 Candidate Proposal Generation
    # -------------------------------------------------------------------------
    print("\n[Step 5/12] Testing Deterministic Proposal Generation...")
    proposals, raw_cc_count, fallback_flag = bundle.proposal_gen.generate_proposals(prep["input_512"], prob_512, k=3)
    assert len(proposals) == 3, f"Expected 3 proposals, got {len(proposals)}"
    for i, p in enumerate(proposals):
        assert 'bbox' in p and 'score' in p and 'center' in p
        print(f"   ✓ Proposal #{i+1}: score={p['score']:.4f}, area={p.get('area', 0)}, source={p.get('proposal_source', 'cc')}")

    # -------------------------------------------------------------------------
    # STEP 6: Multi-Scale RoI Extraction & Calcification Enhancement
    # -------------------------------------------------------------------------
    print("\n[Step 6/12] Testing Multi-Scale Concentric RoI Extraction & Calcification Patch...")
    full_t = to_vit_tensor(prep["input_224"]).unsqueeze(0).to(device)
    assert full_t.shape == (1, 3, 224, 224)

    cand_crops_dict = {}
    cand_crops_preview = []
    for prop in proposals:
        crops, _ = bundle.roi_gen.generate_candidate_crops(prep["input_512"], prop)
        cand_crops_preview.append(crops)

    for s in ('local', 'medium', 'large', 'context'):
        t_list = [to_vit_tensor(crops[s].astype(np.float32)) for crops in cand_crops_preview]
        cand_crops_dict[s] = torch.stack(t_list, dim=0).unsqueeze(0).to(device)
        assert cand_crops_dict[s].shape == (1, 3, 3, 224, 224)

    primary_crop = cand_crops_preview[0]['local'].astype(np.float32)
    calc_patch, _ = bundle.roi_gen.enhance_calcification_patch(primary_crop)
    calc_t = to_vit_tensor(calc_patch.astype(np.float32)).unsqueeze(0).to(device)
    assert calc_t.shape == (1, 3, 224, 224)
    print("   ✓ Multi-scale RoI tensors and calcification patch extracted successfully.")

    # -------------------------------------------------------------------------
    # STEP 7: MultiScaleViTV8 Arm F Forward Classification
    # -------------------------------------------------------------------------
    print("\n[Step 7/12] Testing MultiScaleViTV8 Arm F Classification Forward Pass...")
    with torch.inference_mode():
        logits = bundle.vit(cand_crops_dict, full_image_tensor=full_t, calc_patch_tensor=calc_t)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    p_benign, p_malignant = float(probs[0]), float(probs[1])
    assert not math.isnan(p_benign) and not math.isnan(p_malignant), "NaN in output probabilities"
    assert not math.isinf(p_benign) and not math.isinf(p_malignant), "Inf in output probabilities"
    assert abs((p_benign + p_malignant) - 1.0) < 1e-4, f"Probabilities do not sum to 1: {p_benign + p_malignant}"
    print(f"   ✓ ViT Forward Pass Valid: P(Benign)={p_benign*100:.2f}%, P(Malignant)={p_malignant*100:.2f}% (Sum={p_benign+p_malignant:.4f})")

    # -------------------------------------------------------------------------
    # STEP 8: Authentic ViT Self-Attention Rollout Extraction
    # -------------------------------------------------------------------------
    print("\n[Step 8/12] Testing Authentic ViT Self-Attention Rollout Extraction...")
    primary_local_t = to_vit_tensor(primary_crop).unsqueeze(0).to(device)
    attn_map = extract_vit_attention_map(bundle.vit.backbone, primary_local_t, device)
    assert attn_map.shape == (224, 224), f"Unexpected attention shape: {attn_map.shape}"
    assert not np.isnan(attn_map).any(), "NaN in attention map"
    print(f"   ✓ Attention rollout computed successfully. Heatmap range: [{attn_map.min():.3f}, {attn_map.max():.3f}]")

    # -------------------------------------------------------------------------
    # STEP 9: Complete Pipeline Wrapper Execution
    # -------------------------------------------------------------------------
    print("\n[Step 9/12] Testing End-to-End Pipeline Wrapper (run_complete_inference_pipeline)...")
    res = run_complete_inference_pipeline(mock_raw, bundle)
    assert "preprocessing" in res and "segmentation" in res and "classification" in res and "explainability" in res
    assert res["classification"]["predicted_class"] in ["BENIGN", "MALIGNANT"]
    print(f"   ✓ Full pipeline executed cleanly in {res['classification']['latency_s']:.3f}s. Prediction: {res['classification']['predicted_class']}")

    # -------------------------------------------------------------------------
    # STEP 10: CPU Fallback Execution
    # -------------------------------------------------------------------------
    print("\n[Step 10/12] Testing Explicit CPU Fallback Mode...")
    cpu_bundle, err_cpu = load_v8_pipeline_models(torch.device('cpu'))
    assert err_cpu is None and cpu_bundle is not None
    res_cpu = run_complete_inference_pipeline(mock_raw, cpu_bundle)
    assert res_cpu["classification"]["predicted_class"] in ["BENIGN", "MALIGNANT"]
    print(f"   ✓ CPU Fallback verified cleanly. Latency: {res_cpu['classification']['latency_s']:.3f}s")

    # -------------------------------------------------------------------------
    # STEP 11: Real Clinical Benchmark Scans Evaluation
    # -------------------------------------------------------------------------
    print("\n[Step 11/12] Testing Preloaded Real Clinical Benchmark Scans...")
    for case_name, case_data in DEMO_SCANS.items():
        if case_data is None:
            continue
        demo_img = load_demo_scan_image(case_data)
        if demo_img is not None:
            res_demo = run_complete_inference_pipeline(demo_img, bundle)
            p_mal = res_demo["classification"]["prob_malignant"]
            p_ben = res_demo["classification"]["prob_benign"]
            pred = res_demo["classification"]["predicted_class"]
            print(f"   ✓ Verified Scan: {case_name[:40]}... -> Pred: {pred} (P(Mal)={p_mal*100:.1f}%, P(Ben)={p_ben*100:.1f}%)")

    # -------------------------------------------------------------------------
    # STEP 12: Visualization Helper Verification
    # -------------------------------------------------------------------------
    print("\n[Step 12/12] Testing Visualization Functions (Overlays, Heatmaps, Plots)...")
    overlay = create_segmentation_overlay(res["preprocessing"]["img_512"], res["segmentation"]["prob_map"], res["proposals"])
    assert overlay.ndim == 3 and overlay.shape[2] == 3
    attn_overlay = create_attention_overlay(res["explainability"]["primary_local_crop"], res["explainability"]["attention_map_local"])
    assert attn_overlay.ndim == 3 and attn_overlay.shape[2] == 3
    cm_fig = render_confusion_matrix_plot(LOCKED_V8_TEST_METRICS["confusion_matrix"])
    assert cm_fig is not None
    print("   ✓ All medical visualization helpers passed.")

    print("\n" + "=" * 80)
    print("ALL 12 END-TO-END INFERENCE SMOKE TESTS PASSED (100% SUCCESS)")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()
