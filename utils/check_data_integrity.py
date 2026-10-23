import os
import re
import pandas as pd
import numpy as np

DATASET_ROOT = os.path.abspath(os.path.join("..", "Capstone", "datasets"))
CSV_DIR = os.path.join(DATASET_ROOT, "csv")
JPEG_DIR = os.path.join(DATASET_ROOT, "jpeg")

dicom_info = pd.read_csv(os.path.join(CSV_DIR, "dicom_info.csv"))
series_to_items = {}
for _, row in dicom_info.iterrows():
    s_uid = str(row["SeriesInstanceUID"]).strip()
    rel = str(row["image_path"]).strip()
    sub = rel.split("jpeg/")[1] if "jpeg/" in rel else (rel.split("jpeg\\")[1] if "jpeg\\" in rel else rel)
    p = os.path.abspath(os.path.join(JPEG_DIR, sub))
    if s_uid not in series_to_items:
        series_to_items[s_uid] = []
    series_to_items[s_uid].append({
        "path": p,
        "rows": int(row.get("Rows", 0)),
        "cols": int(row.get("Columns", 0))
    })

case_files = [
    "calc_case_description_train_set.csv",
    "calc_case_description_test_set.csv",
    "mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv"
]

ratios = []
for fname in case_files:
    df = pd.read_csv(os.path.join(CSV_DIR, fname))
    for idx, r in df.iterrows():
        img_uids = re.findall(r"1\.3\.6\.1\.4\.1\.9590\.\d+\.\d+\.\d+\.\d+", str(r.get("image file path", "")))
        mask_uids = re.findall(r"1\.3\.6\.1\.4\.1\.9590\.\d+\.\d+\.\d+\.\d+", str(r.get("ROI mask file path", "")))
        
        full_img_shape = None
        for u in img_uids:
            if u in series_to_items:
                items = series_to_items[u]
                largest = max(items, key=lambda x: x["rows"] * x["cols"])
                full_img_shape = (largest["rows"], largest["cols"])
                break
                
        mask_shape = None
        for u in mask_uids:
            if u in series_to_items:
                items = series_to_items[u]
                shape_matches = [it for it in items if (it["rows"], it["cols"]) == full_img_shape]
                if shape_matches:
                    mask_shape = (shape_matches[0]["rows"], shape_matches[0]["cols"])
                else:
                    largest_mask = max(items, key=lambda x: x["rows"] * x["cols"])
                    mask_shape = (largest_mask["rows"], largest_mask["cols"])
                break
                
        if full_img_shape and mask_shape and full_img_shape != mask_shape:
            img_ar = full_img_shape[0] / full_img_shape[1]
            mask_ar = mask_shape[0] / mask_shape[1]
            ar_diff = abs(img_ar - mask_ar)
            scale_h = full_img_shape[0] / mask_shape[0]
            scale_w = full_img_shape[1] / mask_shape[1]
            ratios.append({
                "patient": r.get("patient_id"),
                "file": fname,
                "full_shape": full_img_shape,
                "mask_shape": mask_shape,
                "img_ar": img_ar,
                "mask_ar": mask_ar,
                "ar_diff": ar_diff,
                "scale_h": scale_h,
                "scale_w": scale_w
            })

df_ratios = pd.DataFrame(ratios)
print(f"Total non-equal shape cases: {len(df_ratios)}")
print("Summary of aspect ratio differences:")
print(df_ratios["ar_diff"].describe())
print("\nSummary of scale_h:")
print(df_ratios["scale_h"].describe())
print("\nSummary of scale_w:")
print(df_ratios["scale_w"].describe())
