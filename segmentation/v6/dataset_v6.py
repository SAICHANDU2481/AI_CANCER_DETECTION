"""
================================================================================
V6 Segmentation Dataset & Augmentation Pipeline
================================================================================
Strict Governance:
- Safe medical morphology-preserving augmentations during training only.
- Validation preprocessing remains strictly deterministic.
- Zero test set access.
- Blazing-fast RAM preloading for high-throughput training.
================================================================================
"""

import os
import random
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.enhancement import enhance_mammogram


class V6SegmentationDataset(Dataset):
    """
    High-Throughput V6 Mammography Lesion Segmentation Dataset.
    Preloads enhanced mammograms and masks to memory for instant GPU feed.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        target_size: int = 512,
        is_training: bool = False,
        preload_to_ram: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.target_size = target_size
        self.is_training = is_training
        self.preload_to_ram = preload_to_ram

        self.case_ids = self.df['case_id'].astype(str).tolist()
        self.patient_ids = self.df['patient_id'].astype(str).tolist()
        self.labels = self.df['label'].astype(int).tolist()
        self.abnormality_types = self.df.get('abnormality_type', pd.Series(['UNKNOWN']*len(self.df))).astype(str).tolist()
        self.pathologies = self.df.get('pathology', pd.Series(['UNKNOWN']*len(self.df))).astype(str).tolist()

        if self.preload_to_ram:
            self._preload_all()

    def _resolve_path(self, p: str) -> Optional[str]:
        if os.path.exists(p):
            return p
        alt1 = p.replace("OneDrive\\Desktop", "Desktop")
        if os.path.exists(alt1):
            return alt1
        alt2 = p.replace("Desktop", "OneDrive\\Desktop")
        if os.path.exists(alt2):
            return alt2
        return None

    def _load_single_case(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        case_id = self.case_ids[idx]
        cache_candidates = [
            os.path.join("data", "cache", "v6_seg_512", f"{case_id}.npz"),
            os.path.join(r"C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION\data\cache\v6_seg_512", f"{case_id}.npz"),
            os.path.join(r"C:\Users\sys\Desktop\AI_CANCER_DETECTION\data\cache\v6_seg_512", f"{case_id}.npz")
        ]

        img_512, mask_512 = None, None
        for cp in cache_candidates:
            if os.path.exists(cp):
                try:
                    d = np.load(cp)
                    img_512 = (np.clip(d['image'], 0.0, 1.0) * 255.0).astype(np.uint8)
                    mask_512 = (d['mask'] > 0.5).astype(np.uint8)
                    break
                except Exception:
                    pass

        if img_512 is None:
            row = self.df.iloc[idx]
            raw_p = self._resolve_path(str(row['image_path']))
            if raw_p and os.path.exists(raw_p):
                try:
                    pil_img = Image.open(raw_p).convert("L")
                    arr = np.array(pil_img.resize((self.target_size, self.target_size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
                    prep = enhance_mammogram(arr)
                    img_512 = (np.clip(prep["enhanced"], 0.0, 1.0) * 255.0).astype(np.uint8)
                except Exception:
                    img_512 = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
            else:
                img_512 = np.zeros((self.target_size, self.target_size), dtype=np.uint8)

        if mask_512 is None:
            row = self.df.iloc[idx]
            mask_p = self._resolve_path(str(row.get('mask_path', '')))
            if mask_p and os.path.exists(mask_p):
                try:
                    pil_mask = Image.open(mask_p).convert("L")
                    mask_512 = (np.array(pil_mask.resize((self.target_size, self.target_size), Image.Resampling.NEAREST), dtype=np.float32) > 127).astype(np.uint8)
                except Exception:
                    mask_512 = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
            else:
                mask_512 = np.zeros((self.target_size, self.target_size), dtype=np.uint8)

        return img_512, mask_512

    def _preload_all(self):
        N = len(self.df)
        self.images = np.zeros((N, self.target_size, self.target_size), dtype=np.uint8)
        self.masks = np.zeros((N, self.target_size, self.target_size), dtype=np.uint8)
        self.lesion_areas = np.zeros(N, dtype=np.float32)

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(self._load_single_case, range(N)))

        for i, (img, msk) in enumerate(results):
            self.images[i] = img
            self.masks[i] = msk
            self.lesion_areas[i] = float(np.sum(msk > 0))

    def __len__(self) -> int:
        return len(self.df)

    def _apply_medical_augmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Morphology-preserving medical image and mask synchronized augmentation.
        """
        img_aug = image.copy()
        mask_aug = mask.copy()
        H, W = img_aug.shape

        # 1. Random horizontal flip (50% chance)
        if random.random() < 0.5:
            img_aug = np.fliplr(img_aug).copy()
            mask_aug = np.fliplr(mask_aug).copy()

        # 2. Subtle rotation (+/- 5 degrees)
        if random.random() < 0.35:
            angle = random.uniform(-5.0, 5.0)
            center = (W // 2, H // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_aug = cv2.warpAffine(img_aug, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            mask_aug = cv2.warpAffine(mask_aug, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)

        return img_aug, mask_aug

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.preload_to_ram:
            img_u8 = self.images[idx]
            mask_u8 = self.masks[idx]
            area = self.lesion_areas[idx]
        else:
            img_u8, mask_u8 = self._load_single_case(idx)
            area = float(np.sum(mask_u8 > 0))

        if self.is_training:
            img_u8, mask_u8 = self._apply_medical_augmentation(img_u8, mask_u8)

        img_f32 = img_u8.astype(np.float32) / 255.0
        mask_f32 = mask_u8.astype(np.float32)

        return {
            'case_id': self.case_ids[idx],
            'patient_id': self.patient_ids[idx],
            'abnormality_type': self.abnormality_types[idx],
            'pathology': self.pathologies[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'image': torch.from_numpy(img_f32).unsqueeze(0).float(),
            'mask': torch.from_numpy(mask_f32).unsqueeze(0).float(),
            'lesion_area': float(area)
        }


def get_v6_segmentation_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    train_ds = V6SegmentationDataset(train_df, target_size=target_size, is_training=True, preload_to_ram=True)
    val_ds = V6SegmentationDataset(val_df, target_size=target_size, is_training=False, preload_to_ram=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader
