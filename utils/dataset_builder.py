import os
import re
import glob
import logging
import argparse
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetBuilder")


class CBISDDSMAdapter:
    """
    Adaptive scanner and ingestor for CBIS-DDSM and generic mammography datasets.
    Supports Kaggle CSV structures, TCIA directory formats, DICOM metadata tables,
    and paired image/mask folders.
    """
    def __init__(
        self,
        raw_dir: str = "data/raw",
        output_csv: str = "data/processed/metadata.csv",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ):
        self.raw_dir = raw_dir
        self.output_csv = output_csv
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    @staticmethod
    def extract_patient_id(text: str) -> str:
        """
        Extracts standardized patient identifier (e.g., 'P_00001') from filenames or metadata strings.
        """
        # Match P_XXXXX or Patient_XXXXX or Case_XXXXX
        match = re.search(r"P_\d{5}|Patient_\d+|Case_\d+|P\d{4,5}", text, re.IGNORECASE)
        if match:
            return match.group(0).upper()
        
        # Fallback: first component before underscore/hyphen if informative
        base = os.path.splitext(os.path.basename(text))[0]
        parts = re.split(r"[_\-\s]+", base)
        return parts[0] if parts else base

    @staticmethod
    def map_pathology_label(pathology_str: str) -> Optional[int]:
        """
        Maps standard and variant medical pathology strings to binary classes:
        0 = Benign, 1 = Malignant.
        """
        if pd.isna(pathology_str):
            return None
        p = str(pathology_str).strip().upper()
        if p in ["BENIGN", "BENIGN_WITHOUT_CALLBACK", "0", "NORMAL"]:
            return 0
        elif p in ["MALIGNANT", "1", "CANCER"]:
            return 1
        return None

    def scan_kaggle_cbis_csvs(self, directory: str) -> Optional[pd.DataFrame]:
        """
        Scans for Kaggle CBIS-DDSM CSV files:
        - mass_case_description_train_set.csv
        - mass_case_description_test_set.csv
        - calc_case_description_train_set.csv
        - calc_case_description_test_set.csv
        """
        csv_patterns = [
            "*mass_case_description*.csv",
            "*calc_case_description*.csv",
            "*cbis_ddsm*.csv",
            "metadata*.csv"
        ]
        found_csvs = []
        for pat in csv_patterns:
            found_csvs.extend(glob.glob(os.path.join(directory, "**", pat), recursive=True))

        if not found_csvs:
            return None

        logger.info(f"Detected {len(found_csvs)} potential metadata CSVs in {directory}: {found_csvs}")
        dfs = []
        for cp in set(found_csvs):
            try:
                df_temp = pd.read_csv(cp)
                logger.info(f"Loaded CSV: {cp} with {len(df_temp)} rows. Columns: {list(df_temp.columns)[:6]}...")
                dfs.append(df_temp)
            except Exception as e:
                logger.warning(f"Could not read CSV {cp}: {e}")

        if not dfs:
            return None

        merged = pd.concat(dfs, ignore_index=True)
        return merged

    def scan_image_files(self, directory: str) -> Tuple[List[str], List[str]]:
        """
        Recursively scans directory for image and mask files.
        """
        exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.dcm"]
        all_files = []
        for ext in exts:
            all_files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
            all_files.extend(glob.glob(os.path.join(directory, "**", ext.upper()), recursive=True))

        all_files = sorted(list(set(all_files)))
        
        # Partition masks vs full mammograms
        masks = []
        images = []
        for f in all_files:
            lower = f.lower()
            if "mask" in lower or "roi" in lower or "crop" in lower or "_1." in lower:
                masks.append(f)
            else:
                images.append(f)

        return images, masks

    def build_dataset_metadata(self) -> Dict[str, Any]:
        """
        Ingests the dataset from raw_dir, resolves image/mask pairs,
        standardizes labels, partitions patient-level train/val/test splits,
        and saves data/processed/metadata.csv.
        """
        if not os.path.exists(self.raw_dir):
            msg = f"Raw directory not found at '{self.raw_dir}'."
            logger.warning(msg)
            return {"status": "error", "message": msg}

        logger.info(f"Scanning raw directory: {self.raw_dir}")
        images, masks = self.scan_image_files(self.raw_dir)
        csv_df = self.scan_kaggle_cbis_csvs(self.raw_dir)

        records = []

        if csv_df is not None and len(csv_df) > 0:
            # Normalize column names
            col_map = {c: c.strip().lower().replace(" ", "_") for c in csv_df.columns}
            csv_df = csv_df.rename(columns=col_map)
            
            # Look for image path, mask path, patient id, pathology
            img_col = next((c for c in csv_df.columns if "image_file_path" in c or "image_path" in c or "full_mammogram" in c), None)
            mask_col = next((c for c in csv_df.columns if "roi_mask_file_path" in c or "mask_path" in c or "cropped_image" in c), None)
            pat_col = next((c for c in csv_df.columns if "patient_id" in c or "patient" in c or "subject_id" in c), None)
            pathology_col = next((c for c in csv_df.columns if "pathology" in c or "label" in c or "cancer" in c or "assessment" in c), None)

            logger.info(f"Identified Metadata Columns -> Image: '{img_col}', Mask: '{mask_col}', Patient: '{pat_col}', Pathology: '{pathology_col}'")

            for _, row in csv_df.iterrows():
                # 1. Resolve Image Path
                img_path = str(row[img_col]).strip() if img_col and pd.notna(row[img_col]) else None
                if img_path and not os.path.isabs(img_path):
                    # Check relative to raw_dir
                    candidate = os.path.join(self.raw_dir, img_path)
                    if os.path.exists(candidate):
                        img_path = candidate
                    else:
                        # Search by basename
                        bname = os.path.basename(img_path)
                        matches = [im for im in images if bname in im]
                        img_path = matches[0] if matches else candidate

                # 2. Resolve Mask Path
                mask_path = str(row[mask_col]).strip() if mask_col and pd.notna(row[mask_col]) else None
                if mask_path and not os.path.isabs(mask_path):
                    candidate = os.path.join(self.raw_dir, mask_path)
                    if os.path.exists(candidate):
                        mask_path = candidate
                    else:
                        bname = os.path.basename(mask_path)
                        matches = [m for m in masks if bname in m]
                        mask_path = matches[0] if matches else candidate

                # 3. Patient ID
                patient_id = str(row[pat_col]).strip() if pat_col and pd.notna(row[pat_col]) else None
                if not patient_id and img_path:
                    patient_id = self.extract_patient_id(img_path)

                # 4. Pathology Label
                raw_pathology = row[pathology_col] if pathology_col and pd.notna(row[pathology_col]) else None
                label = self.map_pathology_label(raw_pathology)

                if img_path and patient_id and label is not None:
                    records.append({
                        "patient_id": patient_id,
                        "image_path": img_path,
                        "mask_path": mask_path if mask_path and os.path.exists(mask_path) else "",
                        "label": label,
                        "raw_pathology": str(raw_pathology)
                    })

        # If CSV metadata was not present or didn't match files, match from directory structure
        if not records and images:
            logger.info(f"No CSV mapping resolved. Building metadata directly from {len(images)} scanned image files...")
            for img_p in images:
                pat_id = self.extract_patient_id(img_p)
                
                # Try finding matching mask
                bname = os.path.splitext(os.path.basename(img_p))[0]
                matched_masks = [m for m in masks if bname in os.path.basename(m)]
                mask_p = matched_masks[0] if matched_masks else ""

                # Try inferring pathology from filename / parent dir (e.g. 'benign', 'malignant')
                lower_p = img_p.lower()
                if "malignant" in lower_p or "cancer" in lower_p:
                    lbl = 1
                    raw_lbl = "MALIGNANT"
                elif "benign" in lower_p or "normal" in lower_p:
                    lbl = 0
                    raw_lbl = "BENIGN"
                else:
                    # Default benign placeholder if unspecified
                    lbl = 0
                    raw_lbl = "UNKNOWN_BENIGN"

                records.append({
                    "patient_id": pat_id,
                    "image_path": img_p,
                    "mask_path": mask_p,
                    "label": lbl,
                    "raw_pathology": raw_lbl
                })

        if not records:
            msg = f"No valid mammogram records found in '{self.raw_dir}'. Please ensure images or CSV metadata are placed."
            logger.warning(msg)
            return {"status": "empty", "message": msg}

        df_all = pd.DataFrame(records)
        df_all = df_all.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

        # ---------------------------------------------------------------------
        # Patient-Level Splitting (Zero Leakage)
        # ---------------------------------------------------------------------
        unique_patients = df_all["patient_id"].unique()
        
        # Stratify by predominant patient pathology if possible
        pat_labels = df_all.groupby("patient_id")["label"].max()
        pat_df = pd.DataFrame({"patient_id": pat_labels.index, "label": pat_labels.values})

        # Split patients into Train (70%), Val (15%), Test (15%)
        train_val_ratio = 1.0 - self.test_ratio
        
        try:
            train_val_pats, test_pats = train_test_split(
                pat_df,
                test_size=self.test_ratio,
                stratify=pat_df["label"] if len(pat_df["label"].unique()) > 1 else None,
                random_state=self.random_seed
            )
            
            val_relative = self.val_ratio / train_val_ratio
            train_pats, val_pats = train_test_split(
                train_val_pats,
                test_size=val_relative,
                stratify=train_val_pats["label"] if len(train_val_pats["label"].unique()) > 1 else None,
                random_state=self.random_seed
            )
        except Exception as e:
            logger.warning(f"Stratified patient split fallback to random ({e})")
            np.random.seed(self.random_seed)
            shuffled_pats = np.random.permutation(unique_patients)
            n_total = len(shuffled_pats)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)
            train_pats = pd.DataFrame({"patient_id": shuffled_pats[:n_train]})
            val_pats = pd.DataFrame({"patient_id": shuffled_pats[n_train:n_train + n_val]})
            test_pats = pd.DataFrame({"patient_id": shuffled_pats[n_train + n_val:]})

        train_set = set(train_pats["patient_id"])
        val_set = set(val_pats["patient_id"])
        test_set = set(test_pats["patient_id"])

        # Strict disjointness verification
        leakage = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
        if leakage:
            raise RuntimeError(f"CRITICAL ERROR: Patient leakage detected across splits: {leakage}")

        # Assign split tags
        def get_split(p_id):
            if p_id in train_set:
                return "train"
            elif p_id in val_set:
                return "val"
            elif p_id in test_set:
                return "test"
            return "train"

        df_all["split"] = df_all["patient_id"].map(get_split)

        # Save to metadata CSV
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        df_all.to_csv(self.output_csv, index=False)
        logger.info(f"Successfully generated patient-level metadata CSV with {len(df_all)} entries -> {self.output_csv}")

        # Compile comprehensive summary
        summary = {
            "status": "success",
            "metadata_csv": self.output_csv,
            "total_images": len(df_all),
            "total_patients": len(unique_patients),
            "train_patients": len(train_set),
            "val_patients": len(val_set),
            "test_patients": len(test_set),
            "train_images": int((df_all["split"] == "train").sum()),
            "val_images": int((df_all["split"] == "val").sum()),
            "test_images": int((df_all["split"] == "test").sum()),
            "benign_samples": int((df_all["label"] == 0).sum()),
            "malignant_samples": int((df_all["label"] == 1).sum()),
            "samples_with_masks": int((df_all["mask_path"] != "").sum()),
            "patient_leakage_detected": False
        }

        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan and build patient-level CBIS-DDSM dataset metadata.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--out-csv", type=str, default="data/processed/metadata.csv", help="Path to output metadata.csv")
    args = parser.parse_args()

    builder = CBISDDSMAdapter(raw_dir=args.raw_dir, output_csv=args.out_csv)
    res = builder.build_dataset_metadata()

    print("\n" + "=" * 55)
    print("DATASET INGESTION & PATIENT SPLIT SUMMARY")
    print("=" * 55)
    for k, v in res.items():
        print(f"{k:26s}: {v}")
    print("=" * 55 + "\n")


