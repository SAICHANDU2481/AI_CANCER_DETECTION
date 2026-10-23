"""
================================================================================
V8 Phase 1: Localization Diagnostic & Multi-Source Proposal Benchmark Suite
================================================================================
Evaluates all proposal arms on the validation cohort (541 scans / 235 patients):
- Arm A: V6 512 Baseline
- Arm B: V6 768
- Arm C: Breast-Region Crop + V6 U-KAN
- Arm D1-D6: Calcification-Specific Detectors (Wiener, CLAHE, Top-Hat, DoG, Laplacian, Local Std)
- Arm E: V6 + High-Resolution
- Arm F: V6 + Calcification
- Arm G: V6 + High-Res + Calcification (Primary Combined Benchmark)
- Arm H: Random Proposal Baseline
- Arm I: Fixed-Grid Proposal Baseline

STRICT GOVERNANCE:
- 0 Test Access (test.csv is NEVER read).
- Label-Free Proposal Generation (no GT IoU/pathology for ranking).
- B = 1000 Patient-Clustered Bootstrap.
================================================================================
"""

import os
import sys
import time
import json
import math
import hashlib
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.config import (
    UKAN_V6_PATH, EXPECTED_MD5, verify_file_md5, resolve_path
)
from app.preprocessing import preprocess_mammogram_full
from segmentation.v6.ukan_v6 import MultiScaleUKANV6
from classification.v6.topk_proposals_v6 import TopKProposalGeneratorV6

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def resolve_file_path(p: Optional[str]) -> Optional[str]:
    if p is None or pd.isna(p) or str(p).strip() == '':
        return None
    p_str = str(p).strip()
    if os.path.exists(p_str):
        return p_str
    alt1 = p_str.replace('OneDrive\\\\Desktop', 'Desktop').replace('OneDrive/Desktop', 'Desktop').replace('OneDrive\\Desktop', 'Desktop')
    if os.path.exists(alt1):
        return alt1
    alt2 = p_str.replace('Desktop', 'OneDrive\\Desktop')
    if os.path.exists(alt2):
        return alt2
    return None


def extract_gt_bbox(mask_512: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Extracts bounding box (ymin, xmin, ymax, xmax) from binary ground truth mask."""
    ys, xs = np.where(mask_512 > 0)
    if len(ys) == 0:
        return None
    return int(np.min(ys)), int(np.min(xs)), int(np.max(ys)), int(np.max(xs))


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    yA_min, xA_min, yA_max, xA_max = boxA
    yB_min, xB_min, yB_max, xB_max = boxB

    y_inter_min = max(yA_min, yB_min)
    x_inter_min = max(xA_min, xB_min)
    y_inter_max = min(yA_max, yB_max)
    x_inter_max = min(xA_max, xB_max)

    inter_h = max(0, y_inter_max - y_inter_min)
    inter_w = max(0, x_inter_max - x_inter_min)
    inter_area = inter_h * inter_w

    areaA = max(1, (yA_max - yA_min) * (xA_max - xA_min))
    areaB = max(1, (yB_max - yB_min) * (xB_max - xB_min))
    union_area = areaA + areaB - inter_area

    return float(inter_area / max(1, union_area))


def compute_containment(prop_box: Tuple[int, int, int, int], gt_mask_512: np.ndarray) -> float:
    """Computes fraction of GT lesion mask pixels inside the proposal bounding box."""
    gt_total = np.sum(gt_mask_512 > 0)
    if gt_total == 0:
        return 0.0
    ymin, xmin, ymax, xmax = prop_box
    prop_mask = np.zeros_like(gt_mask_512)
    prop_mask[ymin:ymax, xmin:xmax] = 1.0
    inside = np.sum((gt_mask_512 > 0) & (prop_mask > 0))
    return float(inside / gt_total)


# -----------------------------------------------------------------------------
# DETECTOR IMPLEMENTATIONS
# -----------------------------------------------------------------------------

def detect_breast_region(img_512: np.ndarray) -> Tuple[int, int, int, int]:
    """Deterministic, label-free breast parenchyma bounding box detection."""
    blurred = cv2.GaussianBlur(img_512, (9, 9), 0)
    u8 = (np.clip(blurred, 0, 1) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if num_labels <= 1:
        return (0, 0, 512, 512)
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = int(np.argmax(areas)) + 1
    x = int(stats[max_idx, cv2.CC_STAT_LEFT])
    y = int(stats[max_idx, cv2.CC_STAT_TOP])
    w = int(stats[max_idx, cv2.CC_STAT_WIDTH])
    h = int(stats[max_idx, cv2.CC_STAT_HEIGHT])
    
    pad_x = int(w * 0.05)
    pad_y = int(h * 0.05)
    ymin = max(0, y - pad_y)
    xmin = max(0, x - pad_x)
    ymax = min(512, y + h + pad_y)
    xmax = min(512, x + w + pad_x)
    
    if (ymax - ymin) < 64 or (xmax - xmin) < 64:
        return (0, 0, 512, 512)
        
    return (ymin, xmin, ymax, xmax)


def generate_calcification_proposals(
    img_512: np.ndarray,
    detector_type: str = 'tophat',
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Generates high-frequency candidate proposals using deterministic filters.
    Suppresses image borders, burned-in markers, and extreme edges without GT.
    """
    u8 = (np.clip(img_512, 0, 1) * 255).astype(np.uint8)
    
    if detector_type == 'wiener':
        filt = cv2.GaussianBlur(img_512, (5, 5), 1.0)
        hf = np.abs(img_512 - filt)
    elif detector_type == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enh = clahe.apply(u8).astype(np.float32) / 255.0
        filt = cv2.GaussianBlur(enh, (5, 5), 1.0)
        hf = np.maximum(0, enh - filt)
    elif detector_type == 'tophat':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tophat = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, kernel)
        hf = tophat.astype(np.float32) / 255.0
    elif detector_type == 'dog':
        g1 = cv2.GaussianBlur(img_512, (3, 3), 1.0)
        g2 = cv2.GaussianBlur(img_512, (9, 9), 3.0)
        hf = np.maximum(0, g1 - g2)
    elif detector_type == 'laplacian':
        g = cv2.GaussianBlur(img_512, (5, 5), 1.0)
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
        hf = np.abs(lap)
        hf = hf / (np.max(hf) + 1e-8)
    elif detector_type == 'local_std':
        mean = cv2.blur(img_512, (5, 5))
        sq_mean = cv2.blur(img_512 ** 2, (5, 5))
        var = np.maximum(0, sq_mean - mean ** 2)
        hf = np.sqrt(var)
    else:
        hf = np.zeros_like(img_512)

    # Border and marker suppression mask
    suppress_mask = np.zeros_like(hf)
    suppress_mask[16:496, 16:496] = 1.0
    suppress_mask = suppress_mask * (img_512 > 0.05).astype(np.float32)
    
    hf_masked = hf * suppress_mask
    
    tissue_vals = hf_masked[suppress_mask > 0]
    thresh_val = float(np.percentile(tissue_vals, 98.0)) if len(tissue_vals) > 0 else 0.5
    bin_map = (hf_masked >= thresh_val).astype(np.uint8)
    
    kernel_conn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bin_dilated = cv2.dilate(bin_map, kernel_conn)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_dilated)
    candidates = []
    
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 8 or area > 50000:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        
        cx = x + w / 2.0
        cy = y + h / 2.0
        bw = max(48, int(w * 1.5))
        bh = max(48, int(h * 1.5))
        ymin = max(0, int(cy - bh / 2.0))
        xmin = max(0, int(cx - bw / 2.0))
        ymax = min(512, ymin + bh)
        xmax = min(512, xmin + bw)
        
        comp_crop = hf_masked[ymin:ymax, xmin:xmax]
        score = float(np.mean(comp_crop) * 2.0 + np.max(comp_crop))
        
        candidates.append({
            'source': f'calc_{detector_type}',
            'bbox': (ymin, xmin, ymax, xmax),
            'score': score,
            'area': (ymax - ymin) * (xmax - xmin),
            'center': (cy, cx)
        })
        
    candidates.sort(key=lambda c: c['score'], reverse=True)
    return candidates[:k]


def generate_random_proposals(k: int = 5, seed: int = 42) -> List[Dict[str, Any]]:
    """Deterministic random candidate boxes for baseline sanity check."""
    rng = np.random.RandomState(seed)
    props = []
    for i in range(k):
        bw = rng.randint(48, 160)
        bh = rng.randint(48, 160)
        xmin = rng.randint(16, 512 - bw - 16)
        ymin = rng.randint(16, 512 - bh - 16)
        props.append({
            'source': 'random_baseline',
            'bbox': (ymin, xmin, ymin + bh, xmin + bw),
            'score': float(1.0 / (i + 1)),
            'area': bw * bh,
            'center': (ymin + bh / 2.0, xmin + bw / 2.0)
        })
    return props


def generate_fixed_grid_proposals(k: int = 5) -> List[Dict[str, Any]]:
    """Deterministic fixed-grid candidate boxes covering quadrants and center."""
    grid_boxes = [
        (128, 128, 384, 384), # Center (256x256)
        (32, 32, 256, 256),   # Top-Left
        (32, 256, 256, 480),  # Top-Right
        (256, 32, 480, 256),  # Bottom-Left
        (256, 256, 480, 480)  # Bottom-Right
    ]
    props = []
    for i, bbox in enumerate(grid_boxes[:k]):
        ymin, xmin, ymax, xmax = bbox
        props.append({
            'source': 'grid_baseline',
            'bbox': bbox,
            'score': float(1.0 / (i + 1)),
            'area': (ymax - ymin) * (xmax - xmin),
            'center': ((ymin + ymax) / 2.0, (xmin + xmax) / 2.0)
        })
    return props


def merge_candidate_proposals(
    cand_lists: List[List[Dict[str, Any]]],
    k: int = 5,
    iou_suppress_thresh: float = 0.65
) -> List[Dict[str, Any]]:
    """
    Merges multiple candidate proposal lists with label-free Non-Maximum Suppression (NMS).
    Preserves top distinct regions based on detector confidence.
    """
    all_props = []
    for cl in cand_lists:
        all_props.extend(cl)
        
    all_props.sort(key=lambda p: p['score'], reverse=True)
    
    selected = []
    for cand in all_props:
        box = cand['bbox']
        overlap = False
        for s in selected:
            if compute_iou(box, s['bbox']) > iou_suppress_thresh:
                overlap = True
                break
        if not overlap:
            selected.append(cand)
        if len(selected) == k:
            break
            
    if len(selected) < k:
        for cand in all_props:
            if cand not in selected:
                selected.append(cand)
            if len(selected) == k:
                break
                
    return selected


# -----------------------------------------------------------------------------
# MAIN BENCHMARK RUNNER
# -----------------------------------------------------------------------------

def run_phase1_benchmark():
    print("=" * 80)
    print("STARTING V8 PHASE 1: LOCALIZATION & MULTI-SOURCE PROPOSAL BENCHMARK")
    print("=" * 80)

    # 1. Verify Checkpoints
    is_u_ok, u_md5 = verify_file_md5(UKAN_V6_PATH, EXPECTED_MD5['ukan_v6'])
    assert is_u_ok, f"U-KAN checkpoint verification failed! Found: {u_md5}"
    print(f"U-KAN V6 verified (MD5: {u_md5})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Load U-KAN Models
    ukan = MultiScaleUKANV6(in_channels=1, out_channels=1)
    ckpt = torch.load(UKAN_V6_PATH, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    ukan.load_state_dict(sd)
    ukan.to(device).eval()

    # Proposal generator
    prop_gen_512 = TopKProposalGeneratorV6(min_area=15, threshold=0.15, default_k=5)

    # 3. Load Validation Dataset
    val_csv_path = 'data/processed/val.csv'
    df_val = pd.read_csv(val_csv_path)
    n_scans = len(df_val)
    n_patients = len(df_val['patient_id'].unique())
    print(f"\nLoaded Validation Split: {n_scans} scans across {n_patients} patients.")

    # 4. Process all validation scans
    arm_names = [
        'Arm A (V6 512)',
        'Arm B (V6 768)',
        'Arm C (Breast Crop + V6)',
        'Arm D1 (Wiener)',
        'Arm D2 (CLAHE)',
        'Arm D3 (Top-Hat)',
        'Arm D4 (DoG)',
        'Arm D5 (Laplacian)',
        'Arm D6 (Local Std)',
        'Arm E (V6 + High-Res)',
        'Arm F (V6 + Calc)',
        'Arm G (V6 + High-Res + Calc)',
        'Arm H (Random)',
        'Arm I (Fixed Grid)'
    ]

    case_results = []
    t_start = time.time()

    print("\nExecuting proposal extraction across all validation scans...")
    for idx, row in df_val.iterrows():
        if (idx + 1) % 50 == 0 or (idx + 1) == n_scans:
            elapsed = time.time() - t_start
            print(f"   Processing scan {idx + 1}/{n_scans} ({elapsed:.1f}s, {(idx+1)/elapsed:.1f} scans/s)...")

        case_id = str(row['case_id']) if 'case_id' in row else f"scan_{idx}"
        patient_id = str(row['patient_id'])
        abnormality_type = str(row.get('abnormality_type', 'UNKNOWN')).upper()
        pathology = 'MALIGNANT' if str(row.get('pathology', '')).upper() == 'MALIGNANT' else 'BENIGN'
        image_view = str(row.get('image_view', 'UNKNOWN')).upper()

        img_p = resolve_file_path(row['image_path'])
        mask_p = resolve_file_path(row.get('mask_path', ''))

        raw_img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_mask_raw = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE) if mask_p else np.zeros_like(raw_img)
        gt_mask_512 = cv2.resize((gt_mask_raw > 127).astype(np.float32), (512, 512), interpolation=cv2.INTER_NEAREST)

        gt_bbox = extract_gt_bbox(gt_mask_512)
        gt_area = int(np.sum(gt_mask_512 > 0))

        if gt_area < 1000:
            size_cat = 'SMALL'
        elif gt_area < 5000:
            size_cat = 'MEDIUM'
        else:
            size_cat = 'LARGE'

        # Preprocessing
        prep = preprocess_mammogram_full(raw_img)
        img_512 = prep["input_512"].astype(np.float32)
        img_768 = prep["input_768"].astype(np.float32)

        # --- RUN DETECTORS ---

        # 1. U-KAN 512
        t_512 = torch.from_numpy(img_512).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.inference_mode():
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                p_512_map = torch.sigmoid(ukan(t_512)).squeeze().float().cpu().numpy().astype(np.float32)
        props_a, cc_a, _ = prop_gen_512.generate_proposals(img_512, p_512_map, k=5)

        # 2. U-KAN 768
        t_768 = torch.from_numpy(img_768).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.inference_mode():
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                p_768_map = torch.sigmoid(ukan(t_768)).squeeze().float().cpu().numpy().astype(np.float32)
        p_768_to_512 = cv2.resize(p_768_map, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        props_b, cc_b, _ = prop_gen_512.generate_proposals(img_512, p_768_to_512, k=5)

        # 3. Breast-Region Crop + U-KAN
        b_ymin, b_xmin, b_ymax, b_xmax = detect_breast_region(img_512)
        breast_crop = img_512[b_ymin:b_ymax, b_xmin:b_xmax]
        b_h, b_w = breast_crop.shape
        b_crop_512 = cv2.resize(breast_crop, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        t_bc = torch.from_numpy(b_crop_512).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.inference_mode():
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                p_bc = torch.sigmoid(ukan(t_bc)).squeeze().float().cpu().numpy().astype(np.float32)
        
        p_bc_unscaled = cv2.resize(p_bc, (b_w, b_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        p_c_map = np.zeros_like(img_512, dtype=np.float32)
        p_c_map[b_ymin:b_ymax, b_xmin:b_xmax] = p_bc_unscaled
        props_c, _, _ = prop_gen_512.generate_proposals(img_512, p_c_map, k=5)

        # 4. Calcification Detectors D1-D6
        props_d1 = generate_calcification_proposals(img_512, 'wiener', k=5)
        props_d2 = generate_calcification_proposals(img_512, 'clahe', k=5)
        props_d3 = generate_calcification_proposals(img_512, 'tophat', k=5)
        props_d4 = generate_calcification_proposals(img_512, 'dog', k=5)
        props_d5 = generate_calcification_proposals(img_512, 'laplacian', k=5)
        props_d6 = generate_calcification_proposals(img_512, 'local_std', k=5)

        # 5. Combined Arms
        best_calc = merge_candidate_proposals([props_d3, props_d4], k=5)
        props_e = merge_candidate_proposals([props_a, props_b], k=5)
        props_f = merge_candidate_proposals([props_b, best_calc], k=5)
        props_g = merge_candidate_proposals([props_a, props_b, best_calc], k=5)

        # 6. Baselines
        props_h = generate_random_proposals(k=5, seed=SEED + idx)
        props_i = generate_fixed_grid_proposals(k=5)

        arm_proposals_map = {
            'Arm A (V6 512)': props_a,
            'Arm B (V6 768)': props_b,
            'Arm C (Breast Crop + V6)': props_c,
            'Arm D1 (Wiener)': props_d1,
            'Arm D2 (CLAHE)': props_d2,
            'Arm D3 (Top-Hat)': props_d3,
            'Arm D4 (DoG)': props_d4,
            'Arm D5 (Laplacian)': props_d5,
            'Arm D6 (Local Std)': props_d6,
            'Arm E (V6 + High-Res)': props_e,
            'Arm F (V6 + Calc)': props_f,
            'Arm G (V6 + High-Res + Calc)': props_g,
            'Arm H (Random)': props_h,
            'Arm I (Fixed Grid)': props_i
        }

        record = {
            'case_id': case_id,
            'patient_id': patient_id,
            'abnormality_type': abnormality_type,
            'pathology': pathology,
            'image_view': image_view,
            'size_cat': size_cat,
            'gt_area': gt_area
        }

        for arm_name, props in arm_proposals_map.items():
            for k_val in [1, 3, 5]:
                k_props = props[:k_val]
                if gt_bbox is not None and len(k_props) > 0:
                    ious = [compute_iou(p['bbox'], gt_bbox) for p in k_props]
                    contains = [compute_containment(p['bbox'], gt_mask_512) for p in k_props]
                    max_iou = float(np.max(ious))
                    max_contain = float(np.max(contains))
                else:
                    max_iou = 0.0
                    max_contain = 0.0

                record[f"{arm_name}_k{k_val}_max_iou"] = max_iou
                record[f"{arm_name}_k{k_val}_max_contain"] = max_contain
                record[f"{arm_name}_k{k_val}_hit_iou10"] = int(max_iou >= 0.10)
                record[f"{arm_name}_k{k_val}_hit_iou30"] = int(max_iou >= 0.30)
                record[f"{arm_name}_k{k_val}_hit_iou50"] = int(max_iou >= 0.50)
                record[f"{arm_name}_k{k_val}_hit_contain"] = int(max_contain >= 0.50)

            areas = [p.get('area', 0) for p in props[:5]]
            record[f"{arm_name}_prop_count"] = len(props)
            record[f"{arm_name}_mean_area"] = float(np.mean(areas)) if len(areas) > 0 else 0.0
            record[f"{arm_name}_is_zero_prop"] = int(len(props) == 0 or (len(props) == 1 and props[0].get('proposal_source') == 'zero_fallback'))

        case_results.append(record)

    df_res = pd.DataFrame(case_results)
    
    # Save case-level predictions
    case_pred_path = os.path.join("results", "v8", "phase1", "phase1_case_level_predictions.csv")
    df_res.to_csv(case_pred_path, index=False)
    print(f"\nSaved case-level predictions to {case_pred_path}")

    # 5. Compute Aggregate Benchmark Table
    summary_rows = []
    print("\nComputing aggregate metrics per arm...")
    for arm_name in arm_names:
        row_dict = {'Arm': arm_name}
        for k_val in [1, 3, 5]:
            r_iou10 = df_res[f"{arm_name}_k{k_val}_hit_iou10"].mean() * 100.0
            r_iou30 = df_res[f"{arm_name}_k{k_val}_hit_iou30"].mean() * 100.0
            r_iou50 = df_res[f"{arm_name}_k{k_val}_hit_iou50"].mean() * 100.0
            r_cont = df_res[f"{arm_name}_k{k_val}_hit_contain"].mean() * 100.0

            row_dict[f"Recall@{k_val} (IoU>=0.10)"] = round(r_iou10, 2)
            row_dict[f"Recall@{k_val} (IoU>=0.30)"] = round(r_iou30, 2)
            row_dict[f"Recall@{k_val} (IoU>=0.50)"] = round(r_iou50, 2)
            row_dict[f"Containment@{k_val}"] = round(r_cont, 2)

        row_dict["Zero-Proposal Rate (%)"] = round(df_res[f"{arm_name}_is_zero_prop"].mean() * 100.0, 2)
        row_dict["Mean Proposal Area (px)"] = round(df_res[f"{arm_name}_mean_area"].mean(), 1)
        summary_rows.append(row_dict)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join("results", "v8", "phase1", "phase1_results.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved summary benchmark table to {summary_csv}")

    # 6. Patient-Clustered Bootstrap (B = 1000)
    print("\nRunning B = 1000 Patient-Clustered Bootstrap across arms...")
    unique_patients = df_res['patient_id'].unique()
    n_p = len(unique_patients)
    rng = np.random.RandomState(SEED)

    bootstrap_records = []
    metrics_to_boot = ['hit_iou10', 'hit_iou30', 'hit_iou50', 'hit_contain']
    patient_to_indices = {pid: df_res[df_res['patient_id'] == pid].index.values for pid in unique_patients}

    boot_indices_list = [
        np.concatenate([patient_to_indices[pid] for pid in rng.choice(unique_patients, size=n_p, replace=True)])
        for _ in range(1000)
    ]

    for arm_name in arm_names:
        for m_name in metrics_to_boot:
            col = f"{arm_name}_k5_{m_name}"
            boot_means = [df_res.iloc[idx_sample][col].mean() * 100.0 for idx_sample in boot_indices_list]
            mean_est = float(np.mean(boot_means))
            ci_low = float(np.percentile(boot_means, 2.5))
            ci_high = float(np.percentile(boot_means, 97.5))

            bootstrap_records.append({
                'Arm': arm_name,
                'Metric': f"K=5 {m_name.replace('hit_', '').upper()}",
                'Bootstrap Mean (%)': round(mean_est, 2),
                '95% CI Lower (%)': round(ci_low, 2),
                '95% CI Upper (%)': round(ci_high, 2),
                '95% CI Format': f"[{ci_low:.2f}% - {ci_high:.2f}%]"
            })

    df_boot = pd.DataFrame(bootstrap_records)
    boot_csv = os.path.join("results", "v8", "phase1", "phase1_bootstrap_ci.csv")
    df_boot.to_csv(boot_csv, index=False)
    print(f"Saved patient-clustered bootstrap CIs to {boot_csv}")

    # 7. Subgroup Breakdown Metrics
    print("\nComputing Subgroup Recall Breakdown...")
    subgroups = {
        'Overall': df_res,
        'Type: MASS': df_res[df_res['abnormality_type'] == 'MASS'],
        'Type: CALCIFICATION': df_res[df_res['abnormality_type'] == 'CALCIFICATION'],
        'Pathology: BENIGN': df_res[df_res['pathology'] == 'BENIGN'],
        'Pathology: MALIGNANT': df_res[df_res['pathology'] == 'MALIGNANT'],
        'View: CC': df_res[df_res['image_view'] == 'CC'],
        'View: MLO': df_res[df_res['image_view'] == 'MLO'],
        'Size: SMALL (<1k px)': df_res[df_res['size_cat'] == 'SMALL'],
        'Size: MEDIUM (1k-5k px)': df_res[df_res['size_cat'] == 'MEDIUM'],
        'Size: LARGE (>5k px)': df_res[df_res['size_cat'] == 'LARGE']
    }

    subgroup_rows = []
    for sg_name, sg_df in subgroups.items():
        n_sg = len(sg_df)
        if n_sg == 0:
            continue
        for arm_name in ['Arm A (V6 512)', 'Arm B (V6 768)', 'Arm D3 (Top-Hat)', 'Arm E (V6 + High-Res)', 'Arm G (V6 + High-Res + Calc)', 'Arm I (Fixed Grid)']:
            subgroup_rows.append({
                'Subgroup': sg_name,
                'Sample Count (N)': n_sg,
                'Arm': arm_name,
                'Recall@5 (IoU>=0.10)': round(sg_df[f"{arm_name}_k5_hit_iou10"].mean() * 100.0, 2),
                'Recall@5 (IoU>=0.30)': round(sg_df[f"{arm_name}_k5_hit_iou30"].mean() * 100.0, 2),
                'Recall@5 (IoU>=0.50)': round(sg_df[f"{arm_name}_k5_hit_iou50"].mean() * 100.0, 2),
                'Containment@5': round(sg_df[f"{arm_name}_k5_hit_contain"].mean() * 100.0, 2)
            })

    df_subgroups = pd.DataFrame(subgroup_rows)
    sg_csv = os.path.join("results", "v8", "phase1", "phase1_subgroup_metrics.csv")
    df_subgroups.to_csv(sg_csv, index=False)
    print(f"Saved subgroup breakdown to {sg_csv}")

    # 8. Recovery Matrix Analysis
    print("\nComputing Recovery Matrix...")
    v6_hits = df_res['Arm A (V6 512)_k5_hit_iou10'] == 1
    v6_misses = ~v6_hits

    hres_hits = df_res['Arm B (V6 768)_k5_hit_iou10'] == 1
    calc_hits = df_res['Arm D3 (Top-Hat)_k5_hit_iou10'] == 1
    comb_hits = df_res['Arm G (V6 + High-Res + Calc)_k5_hit_iou10'] == 1

    recovery_matrix = [
        {'Category': 'Total Validation Cohort', 'Count': len(df_res), 'Percentage': '100.0%'},
        {'Category': 'V6 Arm A Hits (IoU>=0.10)', 'Count': int(v6_hits.sum()), 'Percentage': f"{v6_hits.mean()*100:.2f}%"},
        {'Category': 'V6 Arm A Misses', 'Count': int(v6_misses.sum()), 'Percentage': f"{v6_misses.mean()*100:.2f}%"},
        {'Category': 'V6 Missed -> High-Res Recovered', 'Count': int((v6_misses & hres_hits).sum()), 'Percentage': f"{(v6_misses & hres_hits).sum()/v6_misses.sum()*100:.2f}% of missed"},
        {'Category': 'V6 Missed -> Calcification Recovered', 'Count': int((v6_misses & calc_hits).sum()), 'Percentage': f"{(v6_misses & calc_hits).sum()/v6_misses.sum()*100:.2f}% of missed"},
        {'Category': 'V6 Missed -> Combined Arm G Recovered', 'Count': int((v6_misses & comb_hits).sum()), 'Percentage': f"{(v6_misses & comb_hits).sum()/v6_misses.sum()*100:.2f}% of missed"},
        {'Category': 'Arm G Total Hits', 'Count': int(comb_hits.sum()), 'Percentage': f"{comb_hits.mean()*100:.2f}%"},
        {'Category': 'Persistent Misses in Arm G', 'Count': int((~comb_hits).sum()), 'Percentage': f"{(~comb_hits).mean()*100:.2f}%"}
    ]
    df_rec = pd.DataFrame(recovery_matrix)
    rec_csv = os.path.join("results", "v8", "phase1", "phase1_recovery_matrix.csv")
    df_rec.to_csv(rec_csv, index=False)
    print(f"Saved recovery matrix to {rec_csv}")

    # 9. Save Complete Results JSON
    results_json_path = os.path.join("results", "v8", "phase1", "phase1_results.json")
    json_out = {
        'benchmark_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'governance': {
            'validation_scans': n_scans,
            'validation_patients': n_patients,
            'test_set_access': 0,
            'label_free_ranking': True
        },
        'summary_results': summary_rows,
        'recovery_summary': recovery_matrix
    }
    with open(results_json_path, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f"Saved full JSON results to {results_json_path}")

    # 10. Qualitative Visualizations
    print("\nGenerating Qualitative Visualizations for Representative Validation Cases...")
    qual_dir = os.path.join("results", "v8", "phase1", "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    strong_idx = df_res[df_res['Arm G (V6 + High-Res + Calc)_k5_max_iou'] > 0.60].index[:5]
    weak_idx = df_res[df_res['Arm G (V6 + High-Res + Calc)_k5_max_iou'] < 0.10].index[:5]
    mass_idx = df_res[(df_res['abnormality_type'] == 'MASS') & (df_res['Arm G (V6 + High-Res + Calc)_k5_hit_iou30'] == 1)].index[:5]
    calc_idx = df_res[(df_res['abnormality_type'] == 'CALCIFICATION') & (df_res['Arm G (V6 + High-Res + Calc)_k5_hit_iou30'] == 1)].index[:5]
    small_idx = df_res[(df_res['size_cat'] == 'SMALL') & (df_res['Arm G (V6 + High-Res + Calc)_k5_hit_iou10'] == 1)].index[:5]

    selected_cases = [
        ('strong', strong_idx),
        ('weak', weak_idx),
        ('mass', mass_idx),
        ('calc', calc_idx),
        ('small', small_idx)
    ]

    for cat_name, idx_list in selected_cases:
        for count, c_idx in enumerate(idx_list):
            row = df_val.iloc[c_idx]
            img_p = resolve_file_path(row['image_path'])
            mask_p = resolve_file_path(row.get('mask_path', ''))
            
            raw_img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            gt_mask_raw = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE) if mask_p else np.zeros_like(raw_img)
            gt_mask_512 = cv2.resize((gt_mask_raw > 127).astype(np.float32), (512, 512), interpolation=cv2.INTER_NEAREST)
            
            prep = preprocess_mammogram_full(raw_img)
            img_512 = prep["input_512"].astype(np.float32)
            img_768 = prep["input_768"].astype(np.float32)
            
            t_512 = torch.from_numpy(img_512).unsqueeze(0).unsqueeze(0).float().to(device)
            t_768 = torch.from_numpy(img_768).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.inference_mode():
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    p_512 = torch.sigmoid(ukan(t_512)).squeeze().float().cpu().numpy().astype(np.float32)
                    p_768 = torch.sigmoid(ukan(t_768)).squeeze().float().cpu().numpy().astype(np.float32)
            p_768_512 = cv2.resize(p_768, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            
            p_a, _, _ = prop_gen_512.generate_proposals(img_512, p_512, k=3)
            p_b, _, _ = prop_gen_512.generate_proposals(img_512, p_768_512, k=3)
            p_d3 = generate_calcification_proposals(img_512, 'tophat', k=3)
            p_g = merge_candidate_proposals([p_a, p_b, p_d3], k=3)
            
            base_rgb = cv2.cvtColor((np.clip(img_512, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            
            # Panel 1: Original
            pan_orig = base_rgb.copy()
            
            # Panel 2: GT Mask Overlay
            pan_gt = base_rgb.copy()
            gt_rgb = np.zeros_like(base_rgb)
            gt_rgb[:, :, 1] = (gt_mask_512 * 255).astype(np.uint8)
            pan_gt = cv2.addWeighted(pan_gt, 0.7, gt_rgb, 0.3, 0)
            gt_box = extract_gt_bbox(gt_mask_512)
            if gt_box:
                cv2.rectangle(pan_gt, (gt_box[1], gt_box[0]), (gt_box[3], gt_box[2]), (0, 255, 0), 2)
                cv2.putText(pan_gt, "GT Ref", (gt_box[1], max(16, gt_box[0]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # Panel 3: Arm A Proposals
            pan_a = base_rgb.copy()
            for rank, p in enumerate(p_a[:3]):
                ymin, xmin, ymax, xmax = p['bbox']
                color = (255, 50, 50) if rank == 0 else (200, 100, 100)
                cv2.rectangle(pan_a, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(pan_a, f"A#{rank+1}", (xmin, max(16, ymin-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Panel 4: Arm D3 Calc Proposals
            pan_d3 = base_rgb.copy()
            for rank, p in enumerate(p_d3[:3]):
                ymin, xmin, ymax, xmax = p['bbox']
                color = (50, 200, 255) if rank == 0 else (100, 150, 200)
                cv2.rectangle(pan_d3, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(pan_d3, f"Calc#{rank+1}", (xmin, max(16, ymin-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Panel 5: Arm G Combined Proposals
            pan_g = base_rgb.copy()
            if gt_box:
                cv2.rectangle(pan_g, (gt_box[1], gt_box[0]), (gt_box[3], gt_box[2]), (0, 255, 0), 1)
            for rank, p in enumerate(p_g[:3]):
                ymin, xmin, ymax, xmax = p['bbox']
                color = (255, 0, 255) if rank == 0 else (255, 165, 0)
                cv2.rectangle(pan_g, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(pan_g, f"G#{rank+1}", (xmin, max(16, ymin-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            grid_5 = np.hstack([pan_orig, pan_gt, pan_a, pan_d3, pan_g])
            fname = f"{cat_name}_case_{count+1}_{row['patient_id']}.png"
            cv2.imwrite(os.path.join(qual_dir, fname), grid_5)

    print(f"Saved 25 qualitative visualization panels to {qual_dir}")
    print("\n" + "=" * 80)
    print("V8 PHASE 1 BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_phase1_benchmark()
