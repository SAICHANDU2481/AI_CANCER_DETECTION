"""
================================================================================
V3-A Multi-Scale Dataset & Data Loader
================================================================================
Yields 3-scale contextual crop triplets [3, 3, 224, 224] for shared-weight ViT:
- Local Crop (1.0x with 10% safety margin)
- Medium Crop (1.5x)
- Large Crop (2.0x)
- Pre-caches U-KAN predicted masks and bounding boxes for ultra-fast training.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from classification.roi_generator_v3 import (
    extract_primary_component_bbox,
    compute_multiscale_boxes,
    crop_and_resize,
    generate_v3_triplet_crops
)

logger = logging.getLogger("DatasetV3")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MammographyMultiScaleDatasetV3(Dataset):
    """
    Multi-Scale Mammography Dataset yielding synchronized (Local, Medium, Large) crops.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        is_training: bool = False,
        cache_dir: str = "data/cache/v3_rois",
        preload_to_ram: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.is_training = is_training
        self.cache_dir = cache_dir
        self.preload_to_ram = preload_to_ram
        self.ram_cache = {}

        # PyTorch ImageNet Normalization transform
        self.norm_transform = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Ensure split cache directory exists
        split_name = "train" if is_training else "val"
        self.split_cache_dir = os.path.join(self.cache_dir, split_name)
        os.makedirs(self.split_cache_dir, exist_ok=True)

        logger.info(f"Initialized DatasetV3 ({split_name}): {len(self.df)} samples.")

    def __len__(self) -> int:
        return len(self.df)

    def _apply_training_augmentation(
        self,
        image_512: np.ndarray,
        boxes: Dict[str, Tuple[int, int, int, int]]
    ) -> Tuple[np.ndarray, Dict[str, Tuple[int, int, int, int]]]:
        """
        Applies subtle, orientation-preserving augmentation during training:
        - Small center translation jitter (+/- 5%)
        - Margin jitter (+/- 5%)
        - Mild gamma variation [0.95, 1.05]
        Strictly NO FLIPS.
        """
        H, W = image_512.shape[:2]
        
        # 1. Subtle gamma variation
        if np.random.rand() > 0.5:
            gamma = np.random.uniform(0.95, 1.05)
            img_aug = np.power(np.clip(image_512, 0.0, 1.0), gamma)
        else:
            img_aug = image_512.copy()

        # 2. Synchronized bounding box jitter
        aug_boxes = {}
        for scale_name, box in boxes.items():
            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1

            # Center shift +/- 5% of box width/height
            dx = np.random.uniform(-0.05, 0.05) * bw
            dy = np.random.uniform(-0.05, 0.05) * bh

            # Margin variation +/- 5%
            scale_factor = np.random.uniform(0.95, 1.05)
            new_bw = bw * scale_factor
            new_bh = bh * scale_factor

            cx = (x1 + x2) / 2.0 + dx
            cy = (y1 + y2) / 2.0 + dy

            nx1 = max(0, min(W - 8, int(round(cx - new_bw / 2.0))))
            nx2 = max(nx1 + 8, min(W, int(round(cx + new_bw / 2.0))))
            ny1 = max(0, min(H - 8, int(round(cy - new_bh / 2.0))))
            ny2 = max(ny1 + 8, min(H, int(round(cy + new_bh / 2.0))))

            aug_boxes[scale_name] = (nx1, ny1, nx2, ny2)

        return img_aug, aug_boxes

    def _to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        """
        Converts 2D float32 array in [0, 1] (224, 224) to normalized 3-channel Tensor [3, 224, 224].
        """
        # Replicate grayscale to 3 channels: (3, 224, 224)
        tensor = torch.from_numpy(crop).float().unsqueeze(0).repeat(3, 1, 1)
        return self.norm_transform(tensor)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        case_id = str(row["case_id"])
        patient_id = str(row.get("patient_id", ""))
        abnormality_type = str(row.get("abnormality_type", "")).upper()
        pathology = str(row.get("pathology", "")).upper()

        # Binary label mapping
        label = 1 if "MALIGNANT" in pathology else 0

        # Check RAM cache
        if self.preload_to_ram and case_id in self.ram_cache:
            data = self.ram_cache[case_id]
            image_512 = data["image_512"]
            boxes = data["boxes"]
            roi_source = data["roi_source"]
            mask_area_px = data["mask_area_px"]
        else:
            # Load from preprocessed 512x512 cache
            npz_path = os.path.join("data/cache/segmentation_512", f"{case_id}.npz")
            if os.path.exists(npz_path):
                npz_data = np.load(npz_path)
                image_512 = npz_data["image"].astype(np.float32)
                prob_mask = npz_data["mask"].astype(np.float32)
            else:
                image_512 = np.zeros((512, 512), dtype=np.float32)
                prob_mask = np.zeros((512, 512), dtype=np.float32)

            # Extract multi-scale bounding boxes
            roi_detected, raw_bbox, mask_area_px = extract_primary_component_bbox(prob_mask)
            roi_source = "machine" if roi_detected else "fallback"
            boxes = compute_multiscale_boxes(raw_bbox, roi_detected, img_shape=(512, 512))

            if self.preload_to_ram:
                self.ram_cache[case_id] = {
                    "image_512": image_512,
                    "boxes": boxes,
                    "roi_source": roi_source,
                    "mask_area_px": mask_area_px
                }

        # Apply augmentation if training
        if self.is_training:
            img_in, crop_boxes = self._apply_training_augmentation(image_512, boxes)
        else:
            img_in = image_512
            crop_boxes = boxes

        # Extract 3 crops
        local_crop = crop_and_resize(img_in, crop_boxes["local"], target_size=(224, 224))
        medium_crop = crop_and_resize(img_in, crop_boxes["medium"], target_size=(224, 224))
        large_crop = crop_and_resize(img_in, crop_boxes["large"], target_size=(224, 224))

        # Convert to 3-channel normalized tensors
        t_local = self._to_tensor(local_crop)
        t_medium = self._to_tensor(medium_crop)
        t_large = self._to_tensor(large_crop)

        # Stack into [3, 3, 224, 224] (Scale 0: Local, 1: Medium, 2: Large)
        triplet_tensor = torch.stack([t_local, t_medium, t_large], dim=0)

        return {
            "image": triplet_tensor,  # [3, 3, 224, 224]
            "label": torch.tensor(label, dtype=torch.long),
            "case_id": case_id,
            "patient_id": patient_id,
            "abnormality_type": abnormality_type,
            "pathology": pathology,
            "roi_source": roi_source,
            "mask_area_px": mask_area_px
        }
