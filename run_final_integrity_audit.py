import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Ensure workspace in sys.path
sys.path.insert(0, os.path.abspath("."))

from classification.dataset import resolve_file_path, MammographyROIDataset, get_classification_loaders
from classification.vit_model import VisionTransformerClassifier

def audit():
    meta = pd.read_csv("data/processed/verified_metadata.csv")
    train = pd.read_csv("data/processed/verified_train.csv")
    val = pd.read_csv("data/processed/verified_val.csv")
    test = pd.read_csv("data/processed/verified_test.csv")

    print("="*70)
    print("FINAL CLASSIFICATION INTEGRITY AUDIT")
    print("="*70)

    # 1. Multi-lesion / multi-record mammograms
    img_counts = meta["image_path"].value_counts()
    multi_imgs = img_counts[img_counts > 1]
    print("\n--- 1. Multi-Lesion Mammogram Analysis ---")
    print(f"Total unique physical mammogram files: {len(img_counts)}")
    print(f"Physical mammograms with >1 lesion abnormality: {len(multi_imgs)}")
    print(f"Total abnormality records on these multi-lesion mammograms: {multi_imgs.sum()}")
    print("Distribution of lesion counts per mammogram:")
    for num_lesions, count in img_counts.value_counts().sort_index().items():
        print(f"  {num_lesions} lesion(s) per image: {count} mammograms ({num_lesions * count} records)")

    # Inspecting multi-lesion crops
    sample_multi_img = multi_imgs.index[0]
    sample_records = meta[meta["image_path"] == sample_multi_img]
    print(f"\nSample multi-lesion mammogram ({len(sample_records)} lesions):")
    for _, r in sample_records.iterrows():
        print(f"  Case ID: {r['case_id']}, Abnormality: {r['abnormality_type']}, Pathology: {r['pathology']}, Label: {r['label']}")
        print(f"    Crop Path: {r['crop_path']}")
        print(f"    Mask Path: {r['mask_path']}")

    # 2. Patient containment in exactly one split
    print("\n--- 2. Patient-Level Split Containment ---")
    pat_to_splits = meta.groupby("patient_id")["split"].nunique()
    multi_split_pats = pat_to_splits[pat_to_splits > 1]
    print(f"Patients spanning multiple splits: {len(multi_split_pats)}")

    train_pts = set(train["patient_id"].unique())
    val_pts = set(val["patient_id"].unique())
    test_pts = set(test["patient_id"].unique())

    print(f"Train patients: {len(train_pts)}")
    print(f"Val patients:   {len(val_pts)}")
    print(f"Test patients:  {len(test_pts)}")
    print(f"Train & Val intersection:  {len(train_pts.intersection(val_pts))}")
    print(f"Train & Test intersection: {len(train_pts.intersection(test_pts))}")
    print(f"Val & Test intersection:   {len(val_pts.intersection(test_pts))}")

    # 3. Patient P_01563 Detailed Audit
    print("\n--- 3. Patient P_01563 Audit ---")
    p1563_records = meta[meta["patient_id"] == "P_01563"]
    print(f"Total records for patient P_01563: {len(p1563_records)}")
    for _, r in p1563_records.iterrows():
        print(f"  Case: {r['case_id']}, Abnormality ID: {r['abnormality_id']}, Type: {r['abnormality_type']}, Pathology: {r['pathology']}, Split: {r['split']}")
        print(f"  Full Image: {r['image_path']} (Exists: {os.path.exists(resolve_file_path(r['image_path']))})")
        print(f"  Crop Path:  {r['crop_path']} (Exists: {resolve_file_path(r['crop_path']) is not None and os.path.exists(resolve_file_path(r['crop_path']))})")
        print(f"  Mask Path:  {r['mask_path']} (Exists: {resolve_file_path(r['mask_path']) is not None and os.path.exists(resolve_file_path(r['mask_path']))})")
        print(f"  Seg Valid:  {r['is_valid_for_segmentation']}, Cls Valid: {r['is_valid_for_classification']}")

    # 4. Class weights calculation audit
    print("\n--- 4. Class Weights Source Audit ---")
    train_labels = train["label"].values
    val_labels = val["label"].values
    test_labels = test["label"].values

    train_counts = np.bincount(train_labels, minlength=2)
    val_counts = np.bincount(val_labels, minlength=2)
    test_counts = np.bincount(test_labels, minlength=2)

    w0 = len(train_labels) / (2.0 * train_counts[0])
    w1 = len(train_labels) / (2.0 * train_counts[1])
    print(f"Train Label Counts: Benign (0) = {train_counts[0]}, Malignant (1) = {train_counts[1]}")
    print(f"Computed Class Weights (FROM TRAIN SPLIT ONLY): w0 = {w0:.4f}, w1 = {w1:.4f}")
    print(f"Val Label Counts:   Benign (0) = {val_counts[0]}, Malignant (1) = {val_counts[1]}")
    print(f"Test Label Counts:  Benign (0) = {test_counts[0]}, Malignant (1) = {test_counts[1]}")

    # 5. Attention Hook Non-Interference Check
    print("\n--- 5. Attention Hook Non-Interference Audit ---")
    vit = VisionTransformerClassifier(model_name="vit_base_patch16_224", num_classes=2, pretrained=True)
    dummy_x = torch.randn(4, 3, 224, 224)
    
    # Forward without hook collection
    vit.eval()
    with torch.no_grad():
        out1 = vit(dummy_x)
        out2, attns = vit.forward_with_attention(dummy_x)

    diff = torch.max(torch.abs(out1 - out2)).item()
    print(f"Max absolute difference between standard forward and forward_with_attention: {diff:.8e}")
    print(f"Captured attention maps count: {len(attns)}, shape per map: {attns[0].shape}")
    assert diff == 0.0, "Attention hooks modified the forward pass logits!"
    print("Attention Hook Audit: PASS (Exact zero difference, hooks are strictly read-only)")

    # 6. Normalization Consistency Check
    print("\n--- 6. Normalization & 3-Channel Consistency Audit ---")
    _, _, test_loader, _ = get_classification_loaders("config.yaml")
    test_batch = next(iter(test_loader))
    t_img = test_batch["image"]
    print(f"Test Tensor shape: {t_img.shape} (Batch, Channels, Height, Width)")
    print(f"Tensor mean per channel across batch: {t_img.mean(dim=(0,2,3)).tolist()}")
    print(f"Tensor std per channel across batch:  {t_img.std(dim=(0,2,3)).tolist()}")
    print(f"Tensor dtype: {t_img.dtype}")

    print("\n" + "="*70)
    print("AUDIT EXECUTION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    audit()
