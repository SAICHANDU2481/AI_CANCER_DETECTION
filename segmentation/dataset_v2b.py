"""
================================================================================
V2-B Mammography Segmentation Dataset with Medical-Safe Augmentations & RAM Cache
================================================================================
Strictly adheres to V2-B governance:
- Preserves patient-level split isolation.
- Medical-safe synchronous augmentations (translations, slight rotations, intensity/contrast).
- STRICTLY NO horizontal or vertical flips to respect mammographic anatomical orientation.
- Reads from pre-cached float32 arrays in RAM for maximum GPU throughput.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetV2B")


class MammographySegmentationDatasetV2B(Dataset):
    """
    Patient-level stratified Dataset for V2-B Multi-Scale U-KAN Lesion Segmentation.
    Loads paired mammograms and binary segmentation masks with synchronous medical-safe augmentations.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        img_size: Tuple[int, int] = (512, 512),
        is_training: bool = False,
        cache_dir: str = "data/cache/segmentation_512",
        preload_to_ram: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.is_training = is_training
        self.cache_dir = cache_dir
        self.preload_to_ram = preload_to_ram

        self.images = []
        self.masks = []
        self.case_ids = []
        self.patient_ids = []
        self.abnormality_types = []
        self.pathologies = []

        if self.preload_to_ram:
            logger.info(f"Preloading {len(self.df)} scans into RAM from '{cache_dir}'...")
            for idx, row in self.df.iterrows():
                case_id = str(row["case_id"]) if "case_id" in row else f"case_{idx}"
                cache_path = os.path.join(self.cache_dir, f"{case_id}.npz")
                if not os.path.exists(cache_path):
                    raise FileNotFoundError(f"Missing cache file: {cache_path}")
                data = np.load(cache_path)
                self.images.append(data["image"].astype(np.float32))
                self.masks.append(data["mask"].astype(np.float32))
                self.case_ids.append(case_id)
                self.patient_ids.append(str(row.get("patient_id", "")))
                self.abnormality_types.append(str(row.get("abnormality_type", "")))
                self.pathologies.append(str(row.get("pathology", "")))
            logger.info(f"Preloaded {len(self.images)} scans successfully into RAM.")

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies medically-safe affine and photometric augmentations synchronously.
        STRICTLY NO FLIPS (horizontal or vertical).
        """
        h, w = image.shape

        # 1. Random slight rotation (-8 to +8 degrees)
        angle = np.random.uniform(-8.0, 8.0)
        center = (w // 2, h // 2)

        # 2. Random translation (+-5% of image dimensions)
        tx = np.random.uniform(-0.05 * w, 0.05 * w)
        ty = np.random.uniform(-0.05 * h, 0.05 * h)

        # Affine matrix combining rotation and translation
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot_mat[0, 2] += tx
        rot_mat[1, 2] += ty

        image_aug = cv2.warpAffine(
            image, rot_mat, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0
        )
        mask_aug = cv2.warpAffine(
            mask, rot_mat, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0
        )

        # 3. Photometric Intensity Jitter (scale 0.92 to 1.08)
        intensity_scale = np.random.uniform(0.92, 1.08)
        image_aug = image_aug * intensity_scale

        # 4. Contrast adjustment (power curve gamma 0.90 to 1.10)
        gamma = np.random.uniform(0.90, 1.10)
        image_aug = np.clip(image_aug, 0.0, 1.0) ** gamma

        image_aug = np.clip(image_aug, 0.0, 1.0).astype(np.float32)
        mask_aug = (mask_aug > 0.5).astype(np.float32)

        return image_aug, mask_aug

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.preload_to_ram:
            image = self.images[idx]
            mask = self.masks[idx]
            case_id = self.case_ids[idx]
            patient_id = self.patient_ids[idx]
            abnormality_type = self.abnormality_types[idx]
            pathology = self.pathologies[idx]
        else:
            row = self.df.iloc[idx]
            case_id = str(row["case_id"]) if "case_id" in row else f"case_{idx}"
            cache_path = os.path.join(self.cache_dir, f"{case_id}.npz")
            data = np.load(cache_path)
            image = data["image"].astype(np.float32)
            mask = data["mask"].astype(np.float32)
            patient_id = str(row.get("patient_id", ""))
            abnormality_type = str(row.get("abnormality_type", ""))
            pathology = str(row.get("pathology", ""))

        if self.is_training:
            image, mask = self._augment(image, mask)

        # Convert to PyTorch Tensors [1, H, W]
        img_tensor = torch.from_numpy(image).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "case_id": case_id,
            "patient_id": patient_id,
            "abnormality_type": abnormality_type,
            "pathology": pathology
        }
