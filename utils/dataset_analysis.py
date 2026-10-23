import os
import argparse
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import yaml
from PIL import Image

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetAnalysis")


def analyze_dataset(
    metadata_csv: Optional[str] = "data/processed/metadata.csv",
    raw_dir: Optional[str] = "data/raw",
    config_path: Optional[str] = "config.yaml"
) -> Dict[str, Any]:
    """
    Analyzes medical dataset metadata and files for patient counts, class balance,
    mask availability, dimensions, missing files, and potential patient leakage.

    Args:
        metadata_csv: Path to metadata CSV file.
        raw_dir: Path to raw images folder.
        config_path: Path to config.yaml.

    Returns:
        Structured dictionary of dataset statistics.
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            metadata_csv = cfg.get("dataset", {}).get("metadata_csv", metadata_csv)
            raw_dir = cfg.get("dataset", {}).get("raw_dir", raw_dir)

    results = {
        "status": "missing",
        "total_patients": 0,
        "total_images": 0,
        "benign_samples": 0,
        "malignant_samples": 0,
        "samples_with_masks": 0,
        "missing_image_files": 0,
        "missing_mask_files": 0,
        "duplicate_images": 0,
        "dimension_summary": {},
        "split_summary": {},
        "patient_leakage_detected": False,
        "notes": ""
    }

    if not metadata_csv or not os.path.exists(metadata_csv):
        logger.warning(f"Metadata file not found at: {metadata_csv}. Dataset not configured yet.")
        results["notes"] = f"Metadata file not found at '{metadata_csv}'. Please configure CBIS-DDSM or provide metadata."
        return results

    try:
        df = pd.read_csv(metadata_csv)
    except Exception as e:
        logger.error(f"Error reading metadata CSV {metadata_csv}: {e}")
        results["notes"] = f"Failed to parse CSV: {str(e)}"
        return results

    results["status"] = "present"
    results["total_images"] = len(df)
    
    # 1. Patients count
    if "patient_id" in df.columns:
        unique_patients = df["patient_id"].unique()
        results["total_patients"] = len(unique_patients)
    else:
        results["total_patients"] = "Unknown (patient_id column missing)"

    # 2. Class distribution
    if "label" in df.columns:
        # Standardize numeric or string labels
        # 0 = Benign, 1 = Malignant
        benign_cnt = int((df["label"].astype(str).str.upper().isin(["0", "BENIGN", "BENIGN_WITHOUT_CALLBACK"])).sum())
        malignant_cnt = int((df["label"].astype(str).str.upper().isin(["1", "MALIGNANT"])).sum())
        results["benign_samples"] = benign_cnt
        results["malignant_samples"] = malignant_cnt
    
    # 3. Masks availability
    if "mask_path" in df.columns:
        has_mask = df["mask_path"].notna() & (df["mask_path"].astype(str).str.strip() != "")
        results["samples_with_masks"] = int(has_mask.sum())
        
        # Check actual mask file existence
        missing_masks = 0
        for p in df.loc[has_mask, "mask_path"]:
            if not os.path.exists(str(p)):
                missing_masks += 1
        results["missing_mask_files"] = missing_masks

    # 4. Check image file existence & dimensions
    if "image_path" in df.columns:
        missing_imgs = 0
        dims = []
        for p in df["image_path"]:
            if not os.path.exists(str(p)):
                missing_imgs += 1
            else:
                try:
                    with Image.open(p) as im:
                        dims.append(im.size) # (width, height)
                except Exception:
                    pass
        results["missing_image_files"] = missing_imgs
        
        if dims:
            widths, heights = zip(*dims)
            results["dimension_summary"] = {
                "min_width": int(min(widths)),
                "max_width": int(max(widths)),
                "min_height": int(min(heights)),
                "max_height": int(max(heights)),
                "unique_resolutions": len(set(dims))
            }
        
        # Duplicates check by path
        results["duplicate_images"] = int(df["image_path"].duplicated().sum())

    # 5. Split and patient leakage analysis
    if "split" in df.columns and "patient_id" in df.columns:
        split_counts = df["split"].value_counts().to_dict()
        results["split_summary"] = {str(k): int(v) for k, v in split_counts.items()}
        
        # Check for intersection of patients across splits
        train_pts = set(df[df["split"] == "train"]["patient_id"])
        val_pts = set(df[df["split"] == "val"]["patient_id"])
        test_pts = set(df[df["split"] == "test"]["patient_id"])
        
        leakage = (train_pts & val_pts) | (train_pts & test_pts) | (val_pts & test_pts)
        if leakage:
            results["patient_leakage_detected"] = True
            logger.error(f"PATIENT LEAKAGE DETECTED! {len(leakage)} patients overlap across splits: {list(leakage)[:5]}")
        else:
            results["patient_leakage_detected"] = False
            logger.info("Patient-level splitting verified: zero patient leakage across train/val/test splits.")

    logger.info("Dataset analysis completed.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze mammography dataset metadata.")
    parser.add_argument("--csv", type=str, default="data/processed/metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    report = analyze_dataset(metadata_csv=args.csv, config_path=args.config)
    print("\n" + "="*50)
    print("DATASET ANALYSIS REPORT")
    print("="*50)
    for k, v in report.items():
        print(f"{k:25s}: {v}")
    print("="*50 + "\n")
