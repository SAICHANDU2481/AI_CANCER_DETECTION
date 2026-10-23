import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import Image

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("CBISProcessor")

DATASET_ROOT = os.path.abspath(os.path.join("..", "Capstone", "datasets"))
CSV_DIR = os.path.join(DATASET_ROOT, "csv")
JPEG_DIR = os.path.join(DATASET_ROOT, "jpeg")

RESULTS_DIR = os.path.abspath("results/dataset_analysis")
PROCESSED_DIR = os.path.abspath("data/processed")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean_path_str(p: Any) -> str:
    if pd.isna(p):
        return ""
    return str(p).strip().replace("\n", "").replace("\r", "")


def map_pathology(p_str: str) -> int:
    s = str(p_str).strip().upper()
    if s in ["BENIGN", "BENIGN_WITHOUT_CALLBACK"]:
        return 0
    elif s in ["MALIGNANT", "CANCER"]:
        return 1
    else:
        raise ValueError(f"Unknown pathology string: {p_str}")


def process_dataset() -> Dict[str, Any]:
    logger.info("Loading DICOM metadata mapping (dicom_info.csv)...")
    dicom_info_df = pd.read_csv(os.path.join(CSV_DIR, "dicom_info.csv"))
    
    # Clean file_path and image_path
    dicom_info_df["file_path_clean"] = dicom_info_df["file_path"].apply(clean_path_str)
    dicom_info_df["image_path_clean"] = dicom_info_df["image_path"].apply(clean_path_str)

    # Build mapping dictionaries
    # 1. Map relative DICOM path -> full JPEG path
    # In dicom_info.csv:
    # file_path: "CBIS-DDSM/dicom/1.3.6.1.4.../1-172.dcm"
    # image_path: "CBIS-DDSM/jpeg/1.3.6.1.4.../1-172.jpg"
    
    dicom_to_jpeg = {}
    sop_to_jpeg = {}
    series_to_jpegs = {}

    for _, row in dicom_info_df.iterrows():
        rel_jpg = row["image_path_clean"] # e.g. CBIS-DDSM/jpeg/<UID>/<file>.jpg or jpeg/<UID>/<file>.jpg
        
        # Resolve to actual path on disk in Capstone/datasets/jpeg
        # Extract the <UID>/<file>.jpg part
        if "jpeg/" in rel_jpg:
            uid_subpath = rel_jpg.split("jpeg/", 1)[1]
        elif "jpeg\\" in rel_jpg:
            uid_subpath = rel_jpg.split("jpeg\\", 1)[1]
        else:
            uid_subpath = rel_jpg
            
        full_jpeg_disk_path = os.path.abspath(os.path.join(JPEG_DIR, uid_subpath))
        
        # Also store original DICOM file_path reference
        f_path = row["file_path_clean"]
        dicom_to_jpeg[f_path] = full_jpeg_disk_path
        
        # Case descriptions have paths like "Calc-Training_P_00005_RIGHT_CC/1.3.6.1.4.1.../1.3.6.1.4.1.../000000.dcm"
        # Match by SeriesInstanceUID or SOPInstanceUID
        sop_uid = str(row["SOPInstanceUID"]).strip()
        series_uid = str(row["SeriesInstanceUID"]).strip()
        series_desc = str(row["SeriesDescription"]).strip() if pd.notna(row["SeriesDescription"]) else ""
        
        sop_to_jpeg[sop_uid] = full_jpeg_disk_path
        
        if series_uid not in series_to_jpegs:
            series_to_jpegs[series_uid] = []
        series_to_jpegs[series_uid].append({
            "path": full_jpeg_disk_path,
            "sop": sop_uid,
            "desc": series_desc,
            "instance": row.get("InstanceNumber", 1),
            "rows": row.get("Rows", 0),
            "cols": row.get("Columns", 0),
            "orig_dicom": f_path
        })

    logger.info(f"Built mapping for {len(dicom_to_jpeg)} DICOM->JPEG paths and {len(series_to_jpegs)} series UIDs.")

    # Function to resolve case description path to disk JPEG
    def resolve_case_path(dcm_path_str: str, expected_type: str = "full") -> Tuple[str, bool]:
        p = clean_path_str(dcm_path_str)
        if not p:
            return "", False
            
        # Extract UIDs from path
        uids = re.findall(r"1\.3\.6\.1\.4\.1\.9590\.\d+\.\d+\.\d+\.\d+", p)
        for uid in uids:
            if uid in series_to_jpegs:
                items = series_to_jpegs[uid]
                if expected_type == "full":
                    # Full mammograms are typically the largest image in the series or SeriesDescription == 'full mammogram images'
                    for it in items:
                        if "full" in it["desc"].lower():
                            return it["path"], os.path.exists(it["path"])
                    # Fallback to largest resolution
                    largest = max(items, key=lambda x: x["rows"] * x["cols"])
                    return largest["path"], os.path.exists(largest["path"])
                elif expected_type == "mask":
                    # Mask images have SeriesDescription == 'ROI mask images' or smaller dimensions
                    for it in items:
                        if "mask" in it["desc"].lower():
                            return it["path"], os.path.exists(it["path"])
                    # If multiple, check if one has low number of unique values or first item
                    return items[0]["path"], os.path.exists(items[0]["path"])
                elif expected_type == "crop":
                    for it in items:
                        if "crop" in it["desc"].lower():
                            return it["path"], os.path.exists(it["path"])
                    if len(items) > 1:
                        return items[1]["path"], os.path.exists(items[1]["path"])
                    return items[0]["path"], os.path.exists(items[0]["path"])
                    
                # Default first item
                return items[0]["path"], os.path.exists(items[0]["path"])

        return "", False

    # 2. Ingest all 4 case description tables
    case_files = [
        ("calc_case_description_train_set.csv", "CALCIFICATION", "train"),
        ("calc_case_description_test_set.csv", "CALCIFICATION", "test"),
        ("mass_case_description_train_set.csv", "MASS", "train"),
        ("mass_case_description_test_set.csv", "MASS", "test")
    ]

    master_records = []
    missing_files_count = {
        "full_image_missing": 0,
        "mask_missing": 0,
        "crop_missing": 0
    }

    for fname, ab_type, orig_split in case_files:
        fpath = os.path.join(CSV_DIR, fname)
        df_case = pd.read_csv(fpath)
        logger.info(f"Processing {fname} ({len(df_case)} rows)...")
        
        for idx, row in df_case.iterrows():
            patient_id = str(row["patient_id"]).strip()
            breast_side = str(row.get("left or right breast", "")).strip().upper()
            image_view = str(row.get("image view", "")).strip().upper()
            abnormality_id = int(row.get("abnormality id", 1))
            pathology_raw = str(row.get("pathology", "")).strip().upper()
            assessment = row.get("assessment", None)
            subtlety = row.get("subtlety", None)
            breast_density = row.get("breast density", row.get("breast_density", None))
            
            # Additional morphological descriptors
            calc_type = row.get("calc type", None)
            calc_dist = row.get("calc distribution", None)
            mass_shape = row.get("mass shape", None)
            mass_margin = row.get("mass margins", None)

            # Map pathology to binary 0/1
            label = map_pathology(pathology_raw)
            
            # Case identifier
            case_id = f"{ab_type}_{patient_id}_{breast_side}_{image_view}_ab{abnormality_id}"

            # Resolve paths
            raw_img_dcm = row.get("image file path", "")
            raw_mask_dcm = row.get("ROI mask file path", "")
            raw_crop_dcm = row.get("cropped image file path", "")

            full_img_path, full_img_exists = resolve_case_path(raw_img_dcm, expected_type="full")
            mask_path, mask_exists = resolve_case_path(raw_mask_dcm, expected_type="mask")
            crop_path, crop_exists = resolve_case_path(raw_crop_dcm, expected_type="crop")

            if not full_img_exists:
                missing_files_count["full_image_missing"] += 1
            if not mask_exists:
                missing_files_count["mask_missing"] += 1
            if not crop_exists:
                missing_files_count["crop_missing"] += 1

            record = {
                "patient_id": patient_id,
                "case_id": case_id,
                "abnormality_type": ab_type,
                "abnormality_id": abnormality_id,
                "breast_side": breast_side,
                "image_view": image_view,
                "image_path": full_img_path,
                "image_exists": full_img_exists,
                "mask_path": mask_path,
                "mask_exists": mask_exists,
                "crop_path": crop_path,
                "crop_exists": crop_exists,
                "pathology": pathology_raw,
                "label": label,
                "assessment": assessment,
                "subtlety": subtlety,
                "breast_density": breast_density,
                "calc_type": calc_type,
                "calc_distribution": calc_dist,
                "mass_shape": mass_shape,
                "mass_margins": mass_margin,
                "original_split": orig_split,
                "source_csv": fname
            }
            master_records.append(record)

    master_df = pd.DataFrame(master_records)
    logger.info(f"Total compiled master records: {len(master_df)}")

    # 3. Patient-Level Splitting (70% Train, 15% Val, 15% Test)
    # Group by patient_id to prevent any patient leakage
    logger.info("Computing patient-level stratified partitions...")
    
    patient_summary = master_df.groupby("patient_id").agg({
        "label": "max", # Patient-level max label (if any scan malignant -> malignant)
        "abnormality_type": lambda x: "BOTH" if len(set(x)) > 1 else list(x)[0],
        "case_id": "count"
    }).reset_index()

    patients = np.array(patient_summary["patient_id"].tolist(), dtype=object)
    pat_labels = np.array(patient_summary["label"].tolist(), dtype=int)

    np.random.seed(42)
    
    # Stratified split by patient label
    train_val_pats, test_pats, train_val_y, test_y = train_test_split(
        patients, pat_labels, test_size=0.15, stratify=pat_labels, random_state=42
    )

    val_relative_ratio = 0.15 / 0.85 # ~0.17647 of remaining 85%
    train_pats, val_pats, _, _ = train_test_split(
        train_val_pats, train_val_y, test_size=val_relative_ratio, stratify=train_val_y, random_state=42
    )

    train_set = set(train_pats)
    val_set = set(val_pats)
    test_set = set(test_pats)

    # 4. Leakage Verification Check
    overlap_train_val = len(train_set & val_set)
    overlap_train_test = len(train_set & test_set)
    overlap_val_test = len(val_set & test_set)

    logger.info(f"Patient Disjointness Check: Train ∩ Val = {overlap_train_val}, Train ∩ Test = {overlap_train_test}, Val ∩ Test = {overlap_val_test}")
    assert overlap_train_val == 0, "Train and Val patient overlap detected!"
    assert overlap_train_test == 0, "Train and Test patient overlap detected!"
    assert overlap_val_test == 0, "Val and Test patient overlap detected!"

    # Assign split to dataframe
    def assign_split(p_id):
        if p_id in train_set:
            return "train"
        elif p_id in val_set:
            return "val"
        elif p_id in test_set:
            return "test"
        return "train"

    master_df["split"] = master_df["patient_id"].apply(assign_split)

    # 5. Save metadata CSV files
    meta_csv_path = os.path.join(PROCESSED_DIR, "metadata.csv")
    train_csv_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_csv_path = os.path.join(PROCESSED_DIR, "val.csv")
    test_csv_path = os.path.join(PROCESSED_DIR, "test.csv")

    master_df.to_csv(meta_csv_path, index=False)
    master_df[master_df["split"] == "train"].to_csv(train_csv_path, index=False)
    master_df[master_df["split"] == "val"].to_csv(val_csv_path, index=False)
    master_df[master_df["split"] == "test"].to_csv(test_csv_path, index=False)

    logger.info(f"Saved processed metadata to: {meta_csv_path}")
    logger.info(f"Saved split files: Train={train_csv_path}, Val={val_csv_path}, Test={test_csv_path}")

    # 6. Generate Analysis Reports
    dataset_summary = {
        "dataset_name": "CBIS-DDSM",
        "dataset_root": DATASET_ROOT,
        "total_case_records": len(master_df),
        "total_unique_patients": len(patient_summary),
        "unique_patients_train": len(train_set),
        "unique_patients_val": len(val_set),
        "unique_patients_test": len(test_set),
        "patient_split_percentages": {
            "train": f"{len(train_set) / len(patient_summary) * 100:.2f}%",
            "val": f"{len(val_set) / len(patient_summary) * 100:.2f}%",
            "test": f"{len(test_set) / len(patient_summary) * 100:.2f}%"
        },
        "image_record_counts_by_split": {
            "train": int((master_df["split"] == "train").sum()),
            "val": int((master_df["split"] == "val").sum()),
            "test": int((master_df["split"] == "test").sum())
        },
        "total_jpegs_on_disk": len(dicom_info_df),
        "image_file_verification": {
            "full_mammograms_verified_on_disk": int(master_df["image_exists"].sum()),
            "masks_verified_on_disk": int(master_df["mask_exists"].sum()),
            "crops_verified_on_disk": int(master_df["crop_exists"].sum())
        },
        "missing_files": missing_files_count
    }

    class_distribution = {
        "total": {
            "benign_0": int((master_df["label"] == 0).sum()),
            "malignant_1": int((master_df["label"] == 1).sum()),
            "benign_percentage": f"{(master_df['label'] == 0).mean() * 100:.2f}%",
            "malignant_percentage": f"{(master_df['label'] == 1).mean() * 100:.2f}%"
        },
        "by_split": {
            "train": {
                "benign": int(((master_df["split"] == "train") & (master_df["label"] == 0)).sum()),
                "malignant": int(((master_df["split"] == "train") & (master_df["label"] == 1)).sum())
            },
            "val": {
                "benign": int(((master_df["split"] == "val") & (master_df["label"] == 0)).sum()),
                "malignant": int(((master_df["split"] == "val") & (master_df["label"] == 1)).sum())
            },
            "test": {
                "benign": int(((master_df["split"] == "test") & (master_df["label"] == 0)).sum()),
                "malignant": int(((master_df["split"] == "test") & (master_df["label"] == 1)).sum())
            }
        },
        "pathology_breakdown": master_df["pathology"].value_counts().to_dict()
    }

    abnormality_distribution = {
        "mass_cases": int((master_df["abnormality_type"] == "MASS").sum()),
        "calcification_cases": int((master_df["abnormality_type"] == "CALCIFICATION").sum()),
        "mass_by_class": {
            "benign": int(((master_df["abnormality_type"] == "MASS") & (master_df["label"] == 0)).sum()),
            "malignant": int(((master_df["abnormality_type"] == "MASS") & (master_df["label"] == 1)).sum())
        },
        "calc_by_class": {
            "benign": int(((master_df["abnormality_type"] == "CALCIFICATION") & (master_df["label"] == 0)).sum()),
            "malignant": int(((master_df["abnormality_type"] == "CALCIFICATION") & (master_df["label"] == 1)).sum())
        }
    }

    leakage_report = {
        "patient_level_splitting_verified": True,
        "train_val_overlap_count": overlap_train_val,
        "train_test_overlap_count": overlap_train_test,
        "val_test_overlap_count": overlap_val_test,
        "status": "PASS - ZERO PATIENT LEAKAGE"
    }

    with open(os.path.join(RESULTS_DIR, "dataset_summary.json"), "w") as f:
        json.dump(dataset_summary, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "class_distribution.json"), "w") as f:
        json.dump(class_distribution, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "abnormality_distribution.json"), "w") as f:
        json.dump(abnormality_distribution, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "leakage_report.json"), "w") as f:
        json.dump(leakage_report, f, indent=2)

    # 7. Generate Publication Plots
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Class Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.countplot(data=master_df, x="pathology", palette=["#2ca02c", "#d62728", "#1f77b4"], ax=ax)
    ax.set_title("CBIS-DDSM Pathology Class Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pathology Diagnosis", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Mammogram Cases", fontsize=11, fontweight="bold")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "class_distribution.png"), dpi=300)
    plt.close(fig)

    # Abnormality vs Pathology Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.countplot(data=master_df, x="abnormality_type", hue="pathology", palette=["#2ca02c", "#d62728", "#1f77b4"], ax=ax)
    ax.set_title("Abnormality Type vs Pathology Breakdown", fontsize=12, fontweight="bold")
    ax.set_xlabel("Abnormality Type (Mass vs Calcification)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "abnormality_distribution.png"), dpi=300)
    plt.close(fig)

    # Patient Split Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.countplot(data=master_df, x="split", hue="label", palette=["#1f77b4", "#d62728"], ax=ax)
    ax.set_title("Patient-Level Train / Val / Test Partition Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Dataset Split (Patient Disjoint)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Scans", fontsize=11, fontweight="bold")
    ax.legend(title="Class", labels=["0: Benign", "1: Malignant"])
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "patient_splits.png"), dpi=300)
    plt.close(fig)

    logger.info("Dataset analysis, metadata generation, and plots completed successfully.")
    return dataset_summary


if __name__ == "__main__":
    process_dataset()
