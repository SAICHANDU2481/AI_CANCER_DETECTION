import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader

from preprocessing.wiener_filter import local_adaptive_wiener, apply_wiener_filter, frequency_domain_wiener
from preprocessing.enhancement import apply_clahe_enhancement, enhance_mammogram
from segmentation.dataset import MammographySegmentationDataset
from classification.dataset import MammographyROIDataset
from segmentation.inference import extract_lesion_roi

def test_real_pipeline_components():
    print("=== TEST 1: Real Image Preprocessing with Wiener Filter & CLAHE ===")
    meta_df = pd.read_csv("data/processed/verified_metadata.csv")
    sample_row = meta_df[meta_df["is_valid_for_segmentation"]].iloc[0]
    
    img_path = sample_row["image_path"]
    mask_path = sample_row["mask_path"]
    
    print(f"Sample Case: {sample_row['case_id']}")
    print(f"Image Path: {img_path}")
    print(f"Mask Path: {mask_path}")
    
    # Open real image (downsample for quick unit test check)
    raw_pil = Image.open(img_path).convert("L")
    raw_img = np.array(raw_pil.resize((512, 512)), dtype=np.float32) / 255.0
    print(f"Raw Resized Image Shape: {raw_img.shape}, Range: [{raw_img.min():.3f}, {raw_img.max():.3f}]")
    
    # Wiener Filter
    wiener_img = local_adaptive_wiener(raw_img, window_size=(5, 5))
    print(f"Wiener Filtered Shape: {wiener_img.shape}, Range: [{wiener_img.min():.3f}, {wiener_img.max():.3f}]")
    assert not np.isnan(wiener_img).any(), "Wiener output contains NaNs"
    
    # Full Preprocessing Pipeline
    proc_dict = enhance_mammogram(raw_img)
    proc_img = proc_dict["enhanced"]
    print(f"Preprocessed Pipeline Shape: {proc_img.shape}, Range: [{proc_img.min():.3f}, {proc_img.max():.3f}]")
    assert not np.isnan(proc_img).any(), "Preprocessed output contains NaNs"
    print("[PASS] Preprocessing verified on real data.")

    print("\n=== TEST 2: Real U-KAN Segmentation DataLoader ===")
    train_df = pd.read_csv("data/processed/verified_train.csv")
    seg_train_df = train_df[train_df["is_valid_for_segmentation"]]
    seg_dataset = MammographySegmentationDataset(
        df=seg_train_df,
        img_size=(256, 256),
        is_training=True,
        apply_preprocessing=True
    )
    print(f"Segmentation Train Dataset Size: {len(seg_dataset)} samples")
    seg_loader = DataLoader(seg_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    batch = next(iter(seg_loader))
    images, masks = batch["image"], batch["mask"]
    print(f"Batch Images Tensor: {images.shape}, dtype={images.dtype}, range=[{images.min():.3f}, {images.max():.3f}]")
    print(f"Batch Masks Tensor: {masks.shape}, dtype={masks.dtype}, range=[{masks.min():.3f}, {masks.max():.3f}]")
    assert images.shape == (4, 1, 256, 256), f"Unexpected image batch shape: {images.shape}"
    assert masks.shape == (4, 1, 256, 256), f"Unexpected mask batch shape: {masks.shape}"
    assert not torch.isnan(images).any(), "Images contain NaNs"
    assert not torch.isnan(masks).any(), "Masks contain NaNs"
    print("[PASS] U-KAN Segmentation DataLoader verified on real data.")

    print("\n=== TEST 3: Real ViT Classification DataLoader ===")
    cls_train_df = train_df[train_df["is_valid_for_classification"]]
    cls_dataset = MammographyROIDataset(
        df=cls_train_df,
        target_size=(224, 224),
        is_training=True
    )
    print(f"Classification Train Dataset Size: {len(cls_dataset)} samples")
    cls_loader = DataLoader(cls_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    cls_batch = next(iter(cls_loader))
    cls_imgs, cls_labels = cls_batch["image"], cls_batch["label"]
    print(f"Batch Cls Images Tensor: {cls_imgs.shape}, dtype={cls_imgs.dtype}, range=[{cls_imgs.min():.3f}, {cls_imgs.max():.3f}]")
    print(f"Batch Cls Labels: {cls_labels}")
    assert cls_imgs.shape == (4, 3, 224, 224), f"Unexpected classification image batch shape: {cls_imgs.shape}"
    assert cls_labels.shape == (4,), f"Unexpected label shape: {cls_labels.shape}"
    assert not torch.isnan(cls_imgs).any(), "Classification images contain NaNs"
    print("[PASS] ViT Classification DataLoader verified on real data.")

    print("\n=== TEST 4: Morphological ROI Extraction on Real Ground Truth Mask ===")
    sample_mask_arr = np.array(Image.open(mask_path).convert("L").resize((512, 512)), dtype=np.float32) / 255.0
    roi_result = extract_lesion_roi(raw_img, sample_mask_arr, threshold=0.5, target_roi_size=(224, 224))
    roi_crop = roi_result["roi"]
    bbox = roi_result["bounding_box"]
    print(f"Extracted ROI Crop Shape: {roi_crop.shape}, Detected: {roi_result['roi_detected']}, Bounding Box: {bbox}")
    assert roi_crop.shape == (224, 224), f"ROI crop shape mismatch: {roi_crop.shape}"
    assert not np.isnan(roi_crop).any(), "ROI crop contains NaNs"
    print("[PASS] Morphological ROI extraction verified on real mask.")

    print("\n========================================================")
    print("ALL REAL DATA INTEGRITY AND LOADER TESTS PASSED 100%!")
    print("========================================================")

if __name__ == "__main__":
    test_real_pipeline_components()
