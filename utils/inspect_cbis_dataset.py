import os
import glob
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetInspector")

DATASET_ROOT = os.path.abspath(os.path.join("..", "Capstone", "datasets"))
CSV_DIR = os.path.join(DATASET_ROOT, "csv")
JPEG_DIR = os.path.join(DATASET_ROOT, "jpeg")

RESULTS_DIR = os.path.abspath("results/dataset_analysis")
PROCESSED_DIR = os.path.abspath("data/processed")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def inspect_all_csvs() -> Dict[str, Any]:
    """Inspects all 6 CSV files in the csv directory."""
    csv_files = [
        "calc_case_description_train_set.csv",
        "calc_case_description_test_set.csv",
        "mass_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "dicom_info.csv",
        "meta.csv"
    ]
    
    csv_reports = {}
    
    for fname in csv_files:
        fpath = os.path.join(CSV_DIR, fname)
        if not os.path.exists(fpath):
            logger.warning(f"File not found: {fpath}")
            csv_reports[fname] = {"status": "missing"}
            continue
            
        logger.info(f"Reading {fname}...")
        df = pd.read_csv(fpath)
        
        # Identify key columns
        cols = list(df.columns)
        num_rows, num_cols = df.shape
        
        # Summarize nulls
        null_counts = df.isnull().sum().to_dict()
        
        # Sample records (first 3)
        sample_records = df.head(3).to_dict(orient="records")
        
        # Unique value analysis on key columns if present
        unique_stats = {}
        for c in cols:
            if df[c].nunique() < 20: # Categorical-like
                unique_stats[c] = df[c].value_counts(dropna=False).to_dict()
            else:
                unique_stats[c] = f"{df[c].nunique()} unique values"
                
        csv_reports[fname] = {
            "status": "present",
            "file_path": fpath,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "columns": cols,
            "sample_records": sample_records,
            "null_counts": null_counts,
            "unique_stats": unique_stats
        }
        
    return csv_reports


def scan_jpeg_directory() -> Dict[str, Any]:
    """Recursively scans the jpeg directory and checks image dimensions, types, etc."""
    logger.info(f"Scanning jpeg directory: {JPEG_DIR}")
    
    all_jpegs = glob.glob(os.path.join(JPEG_DIR, "**", "*.jpg"), recursive=True)
    all_jpegs.extend(glob.glob(os.path.join(JPEG_DIR, "**", "*.jpeg"), recursive=True))
    all_jpegs = sorted(list(set(all_jpegs)))
    
    total_jpegs = len(all_jpegs)
    logger.info(f"Total JPEG images found: {total_jpegs}")
    
    # Check directory structure
    folder_set = set(os.path.dirname(p) for p in all_jpegs)
    num_folders = len(folder_set)
    
    # Sample dimensions and inspect pixel values (e.g. check if masks are binary)
    sample_size = min(200, total_jpegs)
    np.random.seed(42)
    sample_indices = np.random.choice(total_jpegs, sample_size, replace=False) if total_jpegs > 0 else []
    
    dimensions = []
    channels = []
    modes = []
    pixel_ranges = []
    
    for idx in sample_indices:
        jp = all_jpegs[idx]
        try:
            with Image.open(jp) as im:
                dimensions.append(im.size) # (width, height)
                modes.append(im.mode)
                arr = np.array(im)
                channels.append(arr.ndim)
                pixel_ranges.append((int(arr.min()), int(arr.max()), len(np.unique(arr))))
        except Exception as e:
            logger.warning(f"Error opening image {jp}: {e}")
            
    dim_summary = {}
    if dimensions:
        widths, heights = zip(*dimensions)
        dim_summary = {
            "min_width": int(min(widths)),
            "max_width": int(max(widths)),
            "min_height": int(min(heights)),
            "max_height": int(max(heights)),
            "sampled_modes": list(set(modes)),
            "sampled_channels": list(set(channels)),
            "sample_pixel_uniques_distribution": [p[2] for p in pixel_ranges[:10]]
        }
        
    return {
        "total_jpegs": total_jpegs,
        "num_image_folders": num_folders,
        "dimension_summary": dim_summary,
        "sample_file_paths": all_jpegs[:10] if total_jpegs > 0 else []
    }


def analyze_and_build_dataset(csv_reports: Dict[str, Any], jpeg_report: Dict[str, Any]):
    """Analyzes the relational mapping between CSVs and JPEGs, resolves masks vs full images."""
    logger.info("Analyzing relationships between case CSVs, dicom_info.csv, meta.csv, and JPEGs...")
    
    # Load dicom_info.csv if present
    dicom_info_path = os.path.join(CSV_DIR, "dicom_info.csv")
    dicom_df = pd.read_csv(dicom_info_path) if os.path.exists(dicom_info_path) else None
    
    # Load meta.csv if present
    meta_path = os.path.join(CSV_DIR, "meta.csv")
    meta_df = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
    
    # Load case description CSVs
    case_dfs = []
    for ctype in ["mass", "calc"]:
        for split in ["train", "test"]:
            fname = f"{ctype}_case_description_{split}_set.csv"
            fpath = os.path.join(CSV_DIR, fname)
            if os.path.exists(fpath):
                df_temp = pd.read_csv(fpath)
                df_temp["abnormality_type"] = ctype.upper()
                df_temp["original_split"] = split
                df_temp["source_file"] = fname
                case_dfs.append(df_temp)
                
    if not case_dfs:
        logger.error("No case description CSVs found!")
        return None
        
    cases_df = pd.concat(case_dfs, ignore_index=True)
    logger.info(f"Total merged case description records: {len(cases_df)}")
    
    # Let's inspect dicom_info mapping
    # In CBIS-DDSM, image file path in case description matches PatientID / SeriesInstanceUID / SOPInstanceUID or file_path in dicom_info
    return cases_df, dicom_df, meta_df


if __name__ == "__main__":
    logger.info(f"Dataset root: {DATASET_ROOT}")
    csv_reports = inspect_all_csvs()
    jpeg_report = scan_jpeg_directory()
    
    with open(os.path.join(RESULTS_DIR, "csv_inspection_raw.json"), "w") as f:
        # serialize
        clean_reports = {}
        for k, v in csv_reports.items():
            clean_reports[k] = {
                "num_rows": v.get("num_rows"),
                "num_cols": v.get("num_cols"),
                "columns": v.get("columns"),
                "null_counts": v.get("null_counts"),
                "unique_stats": v.get("unique_stats"),
                "sample_records": v.get("sample_records")
            }
        json.dump(clean_reports, f, indent=2)
        
    with open(os.path.join(RESULTS_DIR, "jpeg_inspection_raw.json"), "w") as f:
        json.dump(jpeg_report, f, indent=2)
        
    logger.info("Raw inspections written to results/dataset_analysis/")
