import os
import logging
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml
import cv2

from preprocessing.enhancement import enhance_mammogram
from preprocessing.normalization import normalize_intensity

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("SegmentationDataset")


class MammographySegmentationDataset(Dataset):
    """
    Patient-level stratified PyTorch Dataset for Mammogram Lesion Segmentation.
    Loads paired mammograms and binary segmentation masks with synchronized augmentations.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        img_size: Tuple[int, int] = (512, 512),
        is_training: bool = False,
        apply_preprocessing: bool = True,
        config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = "data/cache/segmentation_512"
    ):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.is_training = is_training
        self.apply_preprocessing = apply_preprocessing
        self.config = config
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies medically-safe affine and geometric augmentations synchronously to image and mask.
        """
        # 1. Random Horizontal Flip (50% chance)
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        # 2. Random slight rotation (-10 to +10 degrees)
        angle = np.random.uniform(-10.0, 10.0)
        h, w = image.shape
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 3. Subtle contrast jitter on image only
        contrast_factor = np.random.uniform(0.9, 1.1)
        image = np.clip(image * contrast_factor, 0.0, 1.0)

        return image, mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        mask_path = str(row["mask_path"]) if "mask_path" in row and pd.notna(row["mask_path"]) else None
        patient_id = str(row["patient_id"]) if "patient_id" in row else f"P_{idx}"
        case_id = str(row["case_id"]) if "case_id" in row else f"case_{idx}"
        label = int(row["label"]) if "label" in row and pd.notna(row["label"]) else -1

        cache_file = os.path.join(self.cache_dir, f"{case_id}.npz") if self.cache_dir else None

        if cache_file and os.path.exists(cache_file):
            try:
                cached = np.load(cache_file)
                processed_img = cached["image"]
                mask_arr = cached["mask"]
            except Exception:
                processed_img, mask_arr = None, None
        else:
            processed_img, mask_arr = None, None

        if processed_img is None or mask_arr is None:
            # 1. Load image and resize to 512x512
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            raw_img = Image.open(img_path).convert("L")
            img_arr = np.array(raw_img.resize(self.img_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

            # 2. Preprocess (Wiener + CLAHE + Normalization)
            if self.apply_preprocessing:
                prep_dict = enhance_mammogram(img_arr, config=self.config)
                processed_img = prep_dict["enhanced"].astype(np.float32)
            else:
                processed_img = normalize_intensity(img_arr, 0.0, 1.0).astype(np.float32)

            # 3. Load or create corresponding mask (Nearest Neighbor)
            if mask_path and os.path.exists(mask_path):
                raw_mask = Image.open(mask_path).convert("L")
                raw_mask_arr = np.array(raw_mask.resize(self.img_size, Image.Resampling.NEAREST), dtype=np.float32)
                # Binarize mask
                mask_arr = (raw_mask_arr > 127).astype(np.float32)
            else:
                mask_arr = np.zeros(self.img_size, dtype=np.float32)

            # Save to cache
            if cache_file:
                try:
                    np.savez_compressed(cache_file, image=processed_img, mask=mask_arr)
                except Exception as e:
                    logger.debug(f"Failed to cache {cache_file}: {e}")

        # Copy arrays before augmentation
        processed_img = np.array(processed_img, copy=True)
        mask_arr = np.array(mask_arr, copy=True)

        # 5. Apply augmentations during training
        if self.is_training:
            processed_img, mask_arr = self._augment(processed_img, mask_arr)

        # 6. Convert to PyTorch tensors (C, H, W)
        tensor_img = torch.from_numpy(processed_img).unsqueeze(0).float() # (1, H, W)
        tensor_mask = torch.from_numpy(mask_arr).unsqueeze(0).float()     # (1, H, W)

        return {
            "image": tensor_img,
            "mask": tensor_mask,
            "patient_id": patient_id,
            "img_path": img_path,
            "label": label
        }


def get_segmentation_loaders(
    config_path: str = "config.yaml"
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Instantiates DataLoader objects for train, val, and test splits strictly split at patient level.
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return None, None, None

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    meta_path = cfg.get("dataset", {}).get("metadata_csv", "data/processed/metadata.csv")
    if not os.path.exists(meta_path):
        logger.warning(f"Dataset metadata file does not exist at '{meta_path}'. DataLoaders cannot be created.")
        return None, None, None

    df = pd.read_csv(meta_path)
    
    # Filter only verified rows with valid masks for segmentation training
    if "is_valid_for_segmentation" in df.columns:
        df_seg = df[df["is_valid_for_segmentation"] == True].copy()
    else:
        df_seg = df[df["mask_path"].notna() & (df["mask_path"].astype(str).str.strip() != "")].copy()

    if len(df_seg) == 0:
        logger.warning("No valid segmentation records found in metadata CSV.")
        return None, None, None

    img_size = tuple(cfg.get("preprocessing", {}).get("target_size", [512, 512]))
    batch_size = cfg.get("segmentation", {}).get("training", {}).get("batch_size", 8)
    num_workers = cfg.get("project", {}).get("num_workers", 0)

    train_df = df_seg[df_seg["split"] == "train"]
    val_df = df_seg[df_seg["split"] == "val"]
    test_df = df_seg[df_seg["split"] == "test"]

    train_ds = MammographySegmentationDataset(train_df, img_size=img_size, is_training=True, config=cfg.get("preprocessing", {}))
    val_ds = MammographySegmentationDataset(val_df, img_size=img_size, is_training=False, config=cfg.get("preprocessing", {}))
    test_ds = MammographySegmentationDataset(test_df, img_size=img_size, is_training=False, config=cfg.get("preprocessing", {}))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) if len(train_ds) > 0 else None
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if len(val_ds) > 0 else None
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers) if len(test_ds) > 0 else None

    logger.info(f"Segmentation datasets loaded: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    return train_loader, val_loader, test_loader
