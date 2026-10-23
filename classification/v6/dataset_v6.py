"""
================================================================================
V6 Dataset Loader: High-Throughput Cached Multi-Scale & Global Context Dataset
================================================================================
Strict Governance:
- Safe medical augmentations only during training (contrast variation, mild affine)
- Synchronized tensor transformations for multi-scale crops
- 100% Isolated Train (2501) and Val (541) cohorts
- Reads from data/cache/v6_machine_rois/
================================================================================
"""

import os
import random
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TRAIN_CACHE_DIR = os.path.join(ROOT, "data", "cache", "v6_machine_rois", "train")
VAL_CACHE_DIR = os.path.join(ROOT, "data", "cache", "v6_machine_rois", "val")


class V6CachedMultiScaleDataset(Dataset):
    """
    High-throughput cached Dataset for V6 Multi-Scale Top-K + Full-Mammogram Breast Cancer Classification.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: str,
        is_training: bool = False,
        k: int = 3,
        use_full_image: bool = True,
        use_calc_enhanced: bool = True,
        scales: Tuple[str, ...] = ('local', 'medium', 'large', 'context')
    ):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.is_training = is_training
        self.k = k
        self.use_full_image = use_full_image
        self.use_calc_enhanced = use_calc_enhanced
        self.scales = scales

        self.norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

        if not os.path.exists(self.cache_dir):
            raise FileNotFoundError(f"FATAL: Cache directory not found: {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.df)

    def _apply_medical_augmentation(self, img_2d: np.ndarray) -> np.ndarray:
        """
        Subtle, morphology-preserving medical image augmentation.
        """
        augmented = img_2d.copy()
        H, W = augmented.shape

        # 1. Intensity & Contrast variation (+/- 6%)
        if random.random() < 0.4:
            alpha = 1.0 + random.uniform(-0.06, 0.06)
            beta = random.uniform(-0.04, 0.04)
            augmented = np.clip(alpha * augmented + beta, 0.0, 1.0)

        # 2. Mild affine rotation & translation (+/- 5 deg, +/- 3%)
        if random.random() < 0.4:
            angle = random.uniform(-5.0, 5.0)
            center = (W // 2, H // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += random.uniform(-0.03 * W, 0.03 * W)
            M[1, 2] += random.uniform(-0.03 * H, 0.03 * H)
            augmented = cv2.warpAffine(augmented, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0.0)

        return augmented.astype(np.float32)

    def _to_tensor(self, crop_2d: np.ndarray) -> torch.Tensor:
        if self.is_training:
            crop_2d = self._apply_medical_augmentation(crop_2d)
        t = np.repeat(crop_2d[np.newaxis, :, :], 3, axis=0)
        t = (t - self.norm_mean) / self.norm_std
        return torch.from_numpy(t).float()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        case_id = str(row['case_id'])
        label = int(row['label'])

        npz_p = os.path.join(self.cache_dir, f"{case_id}.npz")
        data = np.load(npz_p)

        sample = {
            'case_id': case_id,
            'label': torch.tensor(label, dtype=torch.long),
            'abnormality_type': str(row.get('abnormality_type', '')),
            'scores': torch.from_numpy(data.get('proposal_scores', np.ones(self.k, dtype=np.float32)[:self.k])).float()
        }

        # 1. Full Image Global Canvas
        if self.use_full_image:
            sample['full_image'] = self._to_tensor(data['full_image'].astype(np.float32))

        # 2. Enhanced Calcification Patch
        if self.use_calc_enhanced:
            if "calc_enhanced" in data:
                calc_crop = data["calc_enhanced"].astype(np.float32)
            else:
                calc_crop = data["cand1_local"].astype(np.float32)
            sample["calc_enhanced"] = self._to_tensor(calc_crop)

        # 3. Candidate multi-scale crops
        for s in self.scales:
            scale_tensors = []
            for k_i in range(1, self.k + 1):
                key = f'cand{k_i}_{s}'
                if key in data:
                    crop = data[key].astype(np.float32)
                else:
                    crop = data[f'cand1_{s}'].astype(np.float32)
                scale_tensors.append(self._to_tensor(crop))
            # Shape of stacked scale: (K, 3, 224, 224)
            sample[s] = torch.stack(scale_tensors, dim=0)

        return sample


def get_v6_classification_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 16,
    k: int = 3,
    use_full_image: bool = True,
    use_calc_enhanced: bool = True,
    scales: Tuple[str, ...] = ('local', 'medium', 'large', 'context'),
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    train_ds = V6CachedMultiScaleDataset(
        train_df,
        cache_dir=TRAIN_CACHE_DIR,
        is_training=True,
        k=k,
        use_full_image=use_full_image,
        use_calc_enhanced=use_calc_enhanced,
        scales=scales
    )
    val_ds = V6CachedMultiScaleDataset(
        val_df,
        cache_dir=VAL_CACHE_DIR,
        is_training=False,
        k=k,
        use_full_image=use_full_image,
        use_calc_enhanced=use_calc_enhanced,
        scales=scales
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader
