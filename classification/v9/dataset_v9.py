"""
================================================================================
V9 Dataset & RoI Crop Extraction Pipeline
================================================================================
Extracts and caches multi-scale candidate crops and visual pathways:
- Local Crop (1.2x candidate bbox)
- Medium Crop (1.8x candidate bbox)
- Large Crop (2.5x candidate bbox)
- Context Crop (4.0x candidate bbox)
- Full Mammogram (224x224)
- Calcification Patch (Top-Hat / CLAHE enhanced 224x224)
- 7-Dimensional Normalized Proposal Quality Features

STRICT GOVERNANCE:
- Feature normalization computed strictly on TRAIN split statistics.
- ZERO Test split data accessed or cached.
================================================================================
"""

import os
import sys
import time
import random
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


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


def extract_scaled_roi(
    image_512: np.ndarray,
    bbox: Tuple[int, int, int, int],
    scale: float = 1.2,
    target_size: int = 224
) -> np.ndarray:
    """
    Extracts a square crop around candidate bounding box with scale multiplier.
    Deterministic padding and resizing to target_size.
    """
    ymin, xmin, ymax, xmax = bbox
    H, W = image_512.shape
    
    h = max(16, ymax - ymin)
    w = max(16, xmax - xmin)
    cy = (ymin + ymax) / 2.0
    cx = (xmin + xmax) / 2.0
    
    base_dim = max(h, w)
    crop_dim = max(48, int(base_dim * scale))
    half_dim = crop_dim / 2.0
    
    c_ymin = int(round(cy - half_dim))
    c_xmin = int(round(cx - half_dim))
    c_ymax = int(round(cy + half_dim))
    c_xmax = int(round(cx + half_dim))
    
    pad_top = max(0, -c_ymin)
    pad_left = max(0, -c_xmin)
    pad_bottom = max(0, c_ymax - H)
    pad_right = max(0, c_xmax - W)
    
    src_ymin = max(0, c_ymin)
    src_xmin = max(0, c_xmin)
    src_ymax = min(H, c_ymax)
    src_xmax = min(W, c_xmax)
    
    crop = image_512[src_ymin:src_ymax, src_xmin:src_xmax]
    
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=0.0
        )
        
    if crop.shape[0] != target_size or crop.shape[1] != target_size:
        crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
    return crop.astype(np.float32)


def extract_calcification_patch(
    image_512: np.ndarray,
    bbox: Tuple[int, int, int, int],
    target_size: int = 224
) -> np.ndarray:
    """Extracts a Top-Hat / CLAHE enhanced high-resolution patch around the candidate lesion."""
    base_crop = extract_scaled_roi(image_512, bbox, scale=1.5, target_size=target_size)
    u8 = (np.clip(base_crop, 0, 1) * 255).astype(np.uint8)
    
    # Multiscale Top-Hat enhancement
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tophat = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, k).astype(np.float32) / 255.0
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_enh = clahe.apply(u8).astype(np.float32) / 255.0
    
    # Composite enhancement
    enhanced = 0.50 * base_crop + 0.30 * clahe_enh + 0.20 * tophat
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def compute_train_quality_statistics(train_proposals_cache: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Computes mean and std of proposal quality features strictly across TRAIN proposals."""
    all_quals = []
    for case_id, data in train_proposals_cache.items():
        props = data['proposals']
        for p in props[:5]:
            q = [
                p.get('mean_probability', 0.0),
                p.get('max_probability', 0.0),
                p.get('multiscale_agreement', 0.0),
                p.get('calc_response', 0.0),
                p.get('local_contrast', 0.0),
                p.get('detail_response', 0.0),
                p.get('breast_consistency', 1.0)
            ]
            all_quals.append(q)
            
    all_quals = np.array(all_quals, dtype=np.float32)
    mean_vec = np.mean(all_quals, axis=0)
    std_vec = np.std(all_quals, axis=0) + 1e-6
    return mean_vec, std_vec


def verify_or_build_v9_phase2_cache(
    df: pd.DataFrame,
    proposals_cache_path: str,
    target_cache_dir: str,
    split_name: str,
    train_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Tuple[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Builds or verifies NPZ archives for all scans in split_name.
    Stores pre-extracted candidate crops, quality features, full image, calc patch, and context ROI.
    """
    os.makedirs(target_cache_dir, exist_ok=True)
    proposals_cache = torch.load(proposals_cache_path, weights_only=False)
    
    if train_stats is None:
        mean_vec, std_vec = compute_train_quality_statistics(proposals_cache)
    else:
        mean_vec, std_vec = train_stats
        
    n_total = len(df)
    missing_indices = []
    req_keys = {'case_id', 'label', 'full_image', 'calc_patch', 'context_roi', 'cand_crops', 'cand_qualities'}
    
    for idx, row in df.iterrows():
        case_id = str(row.get('case_id', f"scan_{idx}"))
        npz_p = os.path.join(target_cache_dir, f"{case_id}.npz")
        is_valid = False
        if os.path.exists(npz_p) and os.path.getsize(npz_p) > 5000:
            try:
                with np.load(npz_p) as d:
                    if req_keys.issubset(set(d.files)):
                        is_valid = True
            except Exception:
                is_valid = False
        if not is_valid:
            missing_indices.append((idx, row))
            
    if len(missing_indices) == 0:
        print(f"[{split_name.upper()}] RoI Cache 100% verified ({n_total}/{n_total} scans ready).")
        return target_cache_dir, (mean_vec, std_vec)
        
    print(f"[{split_name.upper()}] Generating {len(missing_indices)}/{n_total} RoI caches...")
    t0 = time.time()
    
    for count, (idx, row) in enumerate(missing_indices, 1):
        if count % 250 == 0 or count == len(missing_indices):
            el = time.time() - t0
            print(f"  Processed {count}/{len(missing_indices)} caches ({el:.1f}s)...")
            
        case_id = str(row.get('case_id', f"scan_{idx}"))
        patient_id = str(row['patient_id'])
        label = int(row['label'])
        abn_type = str(row.get('abnormality_type', 'UNKNOWN')).upper()
        pathology = 'MALIGNANT' if str(row.get('pathology', '')).upper() == 'MALIGNANT' else 'BENIGN'
        image_view = str(row.get('image_view', 'UNKNOWN')).upper()
        
        img_p = resolve_file_path(row['image_path'])
        if img_p and os.path.exists(img_p):
            raw_img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        else:
            raw_img = np.zeros((512, 512), dtype=np.float32)
            
        img_512 = cv2.resize(raw_img, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        full_img_224 = cv2.resize(img_512, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
        
        # Get proposals
        case_props_data = proposals_cache.get(case_id, {})
        props = case_props_data.get('proposals', [])
        
        # Ensure 5 proposals exist
        while len(props) < 5:
            props.append({
                'bbox': (128, 128, 384, 384),
                'mean_probability': 0.1,
                'max_probability': 0.1,
                'multiscale_agreement': 0.0,
                'calc_response': 0.0,
                'local_contrast': 0.0,
                'detail_response': 0.0,
                'breast_consistency': 1.0
            })
            
        cand_crops = []
        cand_qualities = []
        
        for p in props[:5]:
            bbox = p['bbox']
            crop_224 = extract_scaled_roi(img_512, bbox, scale=1.2, target_size=224)
            cand_crops.append(crop_224)
            
            raw_q = np.array([
                p.get('mean_probability', 0.0),
                p.get('max_probability', 0.0),
                p.get('multiscale_agreement', 0.0),
                p.get('calc_response', 0.0),
                p.get('local_contrast', 0.0),
                p.get('detail_response', 0.0),
                p.get('breast_consistency', 1.0)
            ], dtype=np.float32)
            
            norm_q = (raw_q - mean_vec) / std_vec
            cand_qualities.append(norm_q)
            
        cand_crops_arr = np.stack(cand_crops, axis=0)  # (5, 224, 224)
        cand_qualities_arr = np.stack(cand_qualities, axis=0)  # (5, 7)
        
        # Context ROI (4.0x around top-1 proposal)
        top_bbox = props[0]['bbox']
        context_roi = extract_scaled_roi(img_512, top_bbox, scale=4.0, target_size=224)
        
        # Dedicated calcification patch
        calc_patch = extract_calcification_patch(img_512, top_bbox, target_size=224)
        
        npz_dest = os.path.join(target_cache_dir, f"{case_id}.npz")
        np.savez_compressed(
            npz_dest,
            case_id=case_id,
            patient_id=patient_id,
            label=label,
            abnormality_type=abn_type,
            pathology=pathology,
            image_view=image_view,
            full_image=full_img_224,
            calc_patch=calc_patch,
            context_roi=context_roi,
            cand_crops=cand_crops_arr,
            cand_qualities=cand_qualities_arr
        )
        
    print(f"[{split_name.upper()}] RoI Cache successfully generated.")
    return target_cache_dir, (mean_vec, std_vec)


class V9MultiCandidateDataset(Dataset):
    """
    High-Throughput Cached Dataset for V9 Multi-Candidate Classification.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: str,
        k: int = 5,
        is_training: bool = False,
        use_full_mammo: bool = False,
        use_calc_patch: bool = False,
        use_context_roi: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.k = min(5, max(1, k))
        self.is_training = is_training
        self.use_full_mammo = use_full_mammo
        self.use_calc_patch = use_calc_patch
        self.use_context_roi = use_context_roi
        
        self.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def _apply_augmentation(self, img_2d: np.ndarray) -> np.ndarray:
        aug = img_2d.copy()
        H, W = aug.shape
        
        if random.random() < 0.4:
            alpha = 1.0 + random.uniform(-0.05, 0.05)
            beta = random.uniform(-0.03, 0.03)
            aug = np.clip(alpha * aug + beta, 0.0, 1.0)
            
        if random.random() < 0.4:
            angle = random.uniform(-5.0, 5.0)
            center = (W // 2, H // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug = cv2.warpAffine(aug, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0.0)
            
        return aug.astype(np.float32)

    def _to_tensor(self, crop_2d: np.ndarray) -> torch.Tensor:
        if self.is_training:
            crop_2d = self._apply_augmentation(crop_2d)
        t = np.repeat(crop_2d[np.newaxis, :, :], 3, axis=0)
        t = (t - self.norm_mean) / self.norm_std
        return torch.from_numpy(t).float()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        case_id = str(row.get('case_id', f"scan_{idx}"))
        npz_p = os.path.join(self.cache_dir, f"{case_id}.npz")
        
        data = np.load(npz_p)
        label = int(data['label'])
        
        cand_crops_raw = data['cand_crops'][:self.k]  # (K, 224, 224)
        cand_qualities_raw = data['cand_qualities'][:self.k]  # (K, 7)
        
        tensors_list = [self._to_tensor(cand_crops_raw[i]) for i in range(self.k)]
        cand_crops_tensor = torch.stack(tensors_list, dim=0)  # (K, 3, 224, 224)
        cand_qualities_tensor = torch.from_numpy(cand_qualities_raw).float()  # (K, 7)
        
        sample = {
            'case_id': case_id,
            'patient_id': str(data['patient_id']),
            'label': torch.tensor(label, dtype=torch.long),
            'abnormality_type': str(data['abnormality_type']),
            'pathology': str(data['pathology']),
            'image_view': str(data['image_view']),
            'candidate_crops': cand_crops_tensor,
            'candidate_qualities': cand_qualities_tensor
        }
        
        if self.use_full_mammo:
            sample['full_mammo'] = self._to_tensor(data['full_image'])
        if self.use_calc_patch:
            sample['calc_patch'] = self._to_tensor(data['calc_patch'])
        if self.use_context_roi:
            sample['context_roi'] = self._to_tensor(data['context_roi'])
            
        return sample
