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
from segmentation.inference import extract_lesion_roi

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ClassificationDataset")


def resolve_file_path(p: Optional[str]) -> Optional[str]:
    """
    Robustly resolves a dataset file path across local and mirror filesystem configurations.
    """
    if p is None or pd.isna(p) or str(p).strip() == "":
        return None
    p_str = str(p).strip()
    if os.path.exists(p_str):
        return p_str
    # Swap OneDrive paths
    alt1 = p_str.replace("OneDrive\\Desktop", "Desktop").replace("OneDrive/Desktop", "Desktop")
    if os.path.exists(alt1):
        return alt1
    alt2 = p_str.replace("Desktop", "OneDrive\\Desktop")
    if os.path.exists(alt2):
        return alt2
    return p_str


class MammographyROIDataset(Dataset):
    """
    Patient-level stratified PyTorch Dataset for Breast Cancer Malignancy Classification (Benign vs Malignant).
    Prepares normalized 3-channel (224x224) lesion ROI tensors for Vision Transformer (ViT) fine-tuning.

    Image Strategy:
    1. Primary: Uses high-resolution lesion crop ('crop_path') if available, applying Wiener + CLAHE enhancement.
    2. Secondary: If crop is absent, crops lesion region from full mammogram using ground-truth mask bounding box + 20% margin.
    3. Fallback: If neither mask nor crop is available, extracts central contextual crop from preprocessed mammogram.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        target_size: Tuple[int, int] = (224, 224),
        is_training: bool = False,
        config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = "data/cache/classification_224"
    ):
        self.df = df.reset_index(drop=True)
        self.target_size = target_size
        self.is_training = is_training
        self.config = config
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Pre-resolve and validate label mappings
        if "label" not in self.df.columns:
            raise ValueError("Dataframe must contain 'label' column (0 for Benign, 1 for Malignant).")

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, roi: np.ndarray) -> np.ndarray:
        """
        Applies medically-safe affine and photometric augmentations to lesion ROI.
        """
        # 1. Random horizontal flip (50% chance)
        if np.random.rand() > 0.5:
            roi = np.fliplr(roi)

        # 2. Random vertical flip (50% chance)
        if np.random.rand() > 0.5:
            roi = np.flipud(roi)

        # 3. Small rotation (-10 to +10 degrees)
        angle = np.random.uniform(-10.0, 10.0)
        h, w = roi.shape[:2]
        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        roi = cv2.warpAffine(roi, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # 4. Subtle brightness and contrast variation
        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-0.03, 0.03)
        roi = np.clip(alpha * roi + beta, 0.0, 1.0)

        return roi

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        case_id = str(row["case_id"]) if "case_id" in row else f"case_{idx}"
        patient_id = str(row["patient_id"]) if "patient_id" in row else f"P_{idx}"
        label = int(row["label"]) # 0 = Benign, 1 = Malignant

        raw_img_path = resolve_file_path(row.get("image_path", None))
        raw_crop_path = resolve_file_path(row.get("crop_path", None))
        raw_mask_path = resolve_file_path(row.get("mask_path", None))

        cache_file = os.path.join(self.cache_dir, f"{case_id}_roi224.npy") if self.cache_dir else None
        roi = None

        # Check preprocessed cache
        if cache_file and os.path.exists(cache_file):
            try:
                roi = np.load(cache_file)
            except Exception:
                roi = None

        if roi is None:
            # 1. Strategy A: Load native CBIS-DDSM crop image if available
            if raw_crop_path and os.path.exists(raw_crop_path):
                raw_crop = Image.open(raw_crop_path).convert("L")
                crop_arr = np.array(raw_crop, dtype=np.float32)
                prep_dict = enhance_mammogram(crop_arr, config=self.config)
                roi_enhanced = prep_dict["enhanced"]
                roi = cv2.resize(roi_enhanced, self.target_size, interpolation=cv2.INTER_AREA)

            # 2. Strategy B: Crop from full mammogram using ground-truth mask
            elif raw_img_path and os.path.exists(raw_img_path) and raw_mask_path and os.path.exists(raw_mask_path):
                raw_img = Image.open(raw_img_path).convert("L")
                img_arr = np.array(raw_img, dtype=np.float32)
                prep_dict = enhance_mammogram(img_arr, config=self.config)
                enhanced_img = prep_dict["enhanced"]

                raw_mask = Image.open(raw_mask_path).convert("L")
                mask_arr = np.array(raw_mask, dtype=np.float32)
                mask_resized = cv2.resize((mask_arr > 127).astype(np.float32), (enhanced_img.shape[1], enhanced_img.shape[0]), interpolation=cv2.INTER_NEAREST)

                roi_res = extract_lesion_roi(
                    image=enhanced_img,
                    predicted_mask=mask_resized,
                    margin_ratio=0.20,
                    target_roi_size=self.target_size,
                    fallback_strategy="center_crop"
                )
                roi = roi_res["roi"]

            # 3. Strategy C: Full mammogram fallback (center-crop)
            elif raw_img_path and os.path.exists(raw_img_path):
                raw_img = Image.open(raw_img_path).convert("L")
                img_arr = np.array(raw_img, dtype=np.float32)
                prep_dict = enhance_mammogram(img_arr, config=self.config)
                enhanced_img = prep_dict["enhanced"]
                h, w = enhanced_img.shape
                ch, cw = int(h * 0.5), int(w * 0.5)
                y_s = (h - ch) // 2
                x_s = (w - cw) // 2
                crop_fallback = enhanced_img[y_s:y_s + ch, x_s:x_s + cw]
                roi = cv2.resize(crop_fallback, self.target_size, interpolation=cv2.INTER_AREA)

            else:
                # Blank fallback if no image file exists
                roi = np.zeros(self.target_size, dtype=np.float32)

            roi = normalize_intensity(roi, 0.0, 1.0).astype(np.float32)

            # Save to cache
            if cache_file:
                try:
                    np.save(cache_file, roi)
                except Exception as e:
                    logger.debug(f"Failed to cache {cache_file}: {e}")

        roi = np.array(roi, copy=True)

        # Apply augmentations during training
        if self.is_training:
            roi = self._augment(roi)

        # Convert to 3-channel tensor and normalize with ImageNet statistics for ViT
        roi_3ch = np.stack([roi, roi, roi], axis=0).astype(np.float32) # (3, 224, 224)
        tensor_roi = torch.from_numpy(roi_3ch).float()

        # ImageNet standardization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor_roi = (tensor_roi - mean) / std

        return {
            "image": tensor_roi,
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": patient_id,
            "case_id": case_id,
            "img_path": str(raw_img_path) if raw_img_path else ""
        }


def get_classification_loaders(
    config_path: str = "config.yaml"
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], Optional[torch.Tensor]]:
    """
    Instantiates patient-stratified DataLoaders for train, val, and test splits
    and computes class inverse-frequency weights for handling imbalance.
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return None, None, None, None

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Check verified split CSVs
    train_csv = "data/processed/verified_train.csv"
    val_csv = "data/processed/verified_val.csv"
    test_csv = "data/processed/verified_test.csv"
    meta_csv = cfg.get("dataset", {}).get("metadata_csv", "data/processed/verified_metadata.csv")

    if os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv):
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
    elif os.path.exists(meta_csv):
        df_full = pd.read_csv(meta_csv)
        train_df = df_full[df_full["split"] == "train"].copy()
        val_df = df_full[df_full["split"] == "val"].copy()
        test_df = df_full[df_full["split"] == "test"].copy()
    else:
        logger.warning(f"No classification CSV found at {meta_csv}.")
        return None, None, None, None

    # Filter is_valid_for_classification if present
    if "is_valid_for_classification" in train_df.columns:
        train_df = train_df[train_df["is_valid_for_classification"] == True].copy()
        val_df = val_df[val_df["is_valid_for_classification"] == True].copy()
        test_df = test_df[test_df["is_valid_for_classification"] == True].copy()

    img_size = tuple(cfg.get("roi_extraction", {}).get("target_roi_size", [224, 224]))
    batch_size = cfg.get("classification", {}).get("training", {}).get("batch_size", 16)
    num_workers = cfg.get("project", {}).get("num_workers", 0)

    train_ds = MammographyROIDataset(train_df, target_size=img_size, is_training=True, config=cfg.get("preprocessing", {}))
    val_ds = MammographyROIDataset(val_df, target_size=img_size, is_training=False, config=cfg.get("preprocessing", {}))
    test_ds = MammographyROIDataset(test_df, target_size=img_size, is_training=False, config=cfg.get("preprocessing", {}))

    # Compute inverse frequency class weights: w_c = N_total / (N_classes * N_c)
    class_weights = None
    if len(train_df) > 0:
        labels = train_df["label"].values
        class_counts = np.bincount(labels, minlength=2)
        total = len(labels)
        if (class_counts > 0).all():
            weights = total / (2.0 * class_counts)
            class_weights = torch.tensor(weights, dtype=torch.float32)
            logger.info(f"Class Distribution: Benign={class_counts[0]}, Malignant={class_counts[1]} -> Class Weights: Benign (0) = {weights[0]:.3f}, Malignant (1) = {weights[1]:.3f}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) if len(train_ds) > 0 else None
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if len(val_ds) > 0 else None
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers) if len(test_ds) > 0 else None

    logger.info(f"Classification DataLoaders loaded: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    return train_loader, val_loader, test_loader, class_weights
