import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Dict, Any, Tuple, Optional, List

class MultiScaleCancerDatasetV4(Dataset):
    """
    High-Performance V4 Multi-Scale Dataset with RAM pre-caching,
    Training-Only Robust Augmentation, and Strict Programmatic Quarantine Guard against test.csv.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        crops_dir: str = 'data/cache/v4/crops',
        is_training: bool = False,
        quality_normalizer: Optional[Any] = None,
        csv_source_path: Optional[str] = None
    ):
        # Programmatic Quarantine Guard
        if csv_source_path is not None and 'test.csv' in os.path.basename(csv_source_path).lower() and is_training:
            raise RuntimeError("CRITICAL VIOLATION: Test cohort data loaded inside training dataset!")
            
        self.df = df.reset_index(drop=True)
        self.crops_dir = crops_dir
        self.is_training = is_training
        self.quality_normalizer = quality_normalizer
        
        self.norm_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Preload all crops and metadata into RAM
        N = len(self.df)
        self.local_arr = np.zeros((N, 224, 224), dtype=np.float32)
        self.medium_arr = np.zeros((N, 224, 224), dtype=np.float32)
        self.large_arr = np.zeros((N, 224, 224), dtype=np.float32)
        self.context_arr = np.zeros((N, 224, 224), dtype=np.float32)
        self.labels = np.zeros(N, dtype=np.int64)
        self.quality_vecs = np.zeros((N, 16), dtype=np.float32)
        self.quality_scores = np.zeros(N, dtype=np.float32)
        
        from classification.v4.roi_quality import QUALITY_FEATURE_NAMES
        
        for idx in range(N):
            row = self.df.iloc[idx]
            case_id = str(row['case_id'])
            pathology = str(row.get('pathology', '')).upper()
            self.labels[idx] = 1 if 'MALIGNANT' in pathology else 0
            self.quality_scores[idx] = float(row.get('roi_quality_score', 0.50))
            
            crop_path = os.path.join(self.crops_dir, f'{case_id}.npz')
            if os.path.exists(crop_path):
                data = np.load(crop_path)
                self.local_arr[idx] = data['local'].astype(np.float32)
                self.medium_arr[idx] = data['medium'].astype(np.float32)
                self.large_arr[idx] = data['large'].astype(np.float32)
                self.context_arr[idx] = data['context'].astype(np.float32)
                
            q_raw = np.array([float(row.get(k, 0.0)) for k in QUALITY_FEATURE_NAMES], dtype=np.float32)
            if self.quality_normalizer is not None:
                self.quality_vecs[idx] = self.quality_normalizer.transform(q_raw)
            else:
                self.quality_vecs[idx] = q_raw

    def __len__(self) -> int:
        return len(self.df)

    def _apply_training_augmentation(self, crop: np.ndarray) -> np.ndarray:
        if not self.is_training:
            return crop
            
        aug_crop = crop.copy()
        
        # 1. Mild brightness/contrast (prob 0.5)
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.92, 1.08)  # contrast
            beta = np.random.uniform(-0.04, 0.04)   # brightness
            aug_crop = np.clip(alpha * aug_crop + beta, 0.0, 1.0)
            
        # 2. Mild Gaussian noise (prob 0.3)
        if np.random.rand() > 0.7:
            noise = np.random.normal(0.0, 0.012, size=aug_crop.shape).astype(np.float32)
            aug_crop = np.clip(aug_crop + noise, 0.0, 1.0)
            
        return aug_crop.astype(np.float32)

    def _to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(crop).unsqueeze(0).repeat(3, 1, 1)
        t = self.norm_transform(t)
        return t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        c_loc = self._apply_training_augmentation(self.local_arr[idx])
        c_med = self._apply_training_augmentation(self.medium_arr[idx])
        c_lrg = self._apply_training_augmentation(self.large_arr[idx])
        c_ctx = self._apply_training_augmentation(self.context_arr[idx])
        
        return {
            'local_crop': self._to_tensor(c_loc),
            'medium_crop': self._to_tensor(c_med),
            'large_crop': self._to_tensor(c_lrg),
            'context_crop': self._to_tensor(c_ctx),
            'quality_vector': torch.from_numpy(self.quality_vecs[idx]).float(),
            'roi_quality_score': torch.tensor(self.quality_scores[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'case_id': str(row['case_id']),
            'patient_id': str(row.get('patient_id', '')),
            'abnormality_type': str(row.get('abnormality_type', '')).upper(),
            'pathology': str(row.get('pathology', '')).upper(),
            'roi_source': str(row.get('roi_source', 'machine_roi')),
            'seg_dice': float(row.get('seg_dice', 0.0)),
            'seg_iou': float(row.get('seg_iou', 0.0))
        }
