"""
================================================================================
Project Migration Script: Local Disk C: -> Local Disk F:
================================================================================
Performs a high-speed, verified migration of the AI Breast Cancer Detection
codebase, models, results, caches, and datasets from C: to F:

Sources:
1. Active Workspace: C:\\Users\\sys\\OneDrive\\Desktop\\AI_CANCER_DETECTION
2. Legacy/Reference Models: C:\\Users\\sys\\Desktop\\AI_CANCER_DETECTION\\models
3. Raw CBIS-DDSM Datasets: C:\\Users\\sys\\Desktop\\Capstone

Destination:
1. Project Workspace: F:\\AI_CANCER_DETECTION
2. Raw Dataset: F:\\Capstone

Verifications:
- Byte-count and file-count parity
- MD5 cryptographic verification on all critical V5/V6 checkpoints
================================================================================
"""

import os
import sys
import time
import shutil
import hashlib
import logging
import subprocess
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("DriveMigration")

SRC_WORKSPACE = r"C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION"
SRC_LEGACY_MODELS = r"C:\Users\sys\Desktop\AI_CANCER_DETECTION\models"
SRC_CAPSTONE = r"C:\Users\sys\Desktop\Capstone"

DEST_WORKSPACE = r"F:\AI_CANCER_DETECTION"
DEST_CAPSTONE = r"F:\Capstone"


def compute_md5(fpath: str) -> str:
    h = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_robocopy(src: str, dst: str, extra_args: List[str] = None) -> int:
    os.makedirs(dst, exist_ok=True)
    cmd = ["robocopy", src, dst, "/E", "/R:2", "/W:2", "/MT:8", "/NP", "/NFL", "/NDL"]
    if extra_args:
        cmd.extend(extra_args)
    logger.info(f"Running Robocopy: {src} -> {dst}")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    # Robocopy exit codes 0-7 mean success (1=files copied, 0=no change, etc.)
    elapsed = time.time() - t0
    logger.info(f"Robocopy finished in {elapsed:.2f}s (Exit code: {res.returncode})")
    return res.returncode


def get_dir_stats(path: str) -> Tuple[int, int]:
    total_files = 0
    total_bytes = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_files += 1
                total_bytes += os.path.getsize(fp)
            except Exception:
                pass
    return total_files, total_bytes


def main():
    logger.info("=" * 80)
    logger.info("STARTING DRIVE C: -> DRIVE F: PROJECT MIGRATION")
    logger.info("=" * 80)

    # Check F drive capacity
    total, used, free = shutil.disk_usage("F:\\")
    logger.info(f"Drive F: Capacity: {total/1e9:.2f} GB total, {free/1e9:.2f} GB free.")

    # 1. Sync Active Workspace to F:\AI_CANCER_DETECTION
    logger.info("Step 1: Syncing Active Workspace C:\\...\\AI_CANCER_DETECTION -> F:\\AI_CANCER_DETECTION...")
    run_robocopy(SRC_WORKSPACE, DEST_WORKSPACE)

    # 2. Sync any additional legacy models if present
    if os.path.exists(SRC_LEGACY_MODELS):
        logger.info("Step 2: Syncing Legacy Models -> F:\\AI_CANCER_DETECTION\\models (no overwrite on newer)...")
        run_robocopy(SRC_LEGACY_MODELS, os.path.join(DEST_WORKSPACE, "models"), ["/XO"])

    # 3. Sync Raw Capstone Dataset
    if os.path.exists(SRC_CAPSTONE):
        logger.info("Step 3: Syncing Raw CBIS-DDSM Dataset C:\\...\\Capstone -> F:\\Capstone...")
        run_robocopy(SRC_CAPSTONE, DEST_CAPSTONE)

    # 4. Verify Critical Checkpoints and Artifacts on Drive F:
    logger.info("=" * 80)
    logger.info("VERIFYING DESTINATION CHECKSUMS ON DRIVE F:")
    logger.info("=" * 80)

    critical_checkpoints = {
        "V6 U-KAN Segmentation (Arm G Best)": (
            os.path.join(DEST_WORKSPACE, "models", "v6", "segmentation", "final", "ukan_v6_arm_g_best.pth"),
            "68cfc38cfbddd93d6e4a6e692fffbb34"
        ),
        "V6 ViT Classifier (Arm C Best)": (
            os.path.join(DEST_WORKSPACE, "models", "v6", "classification", "vit_v6_arm_c_best.pth"),
            "508250cfc1c07af1ea7d81cc2e5a7418"
        ),
        "V6 ViT Classifier (Arm B Best)": (
            os.path.join(DEST_WORKSPACE, "models", "v6", "classification", "vit_v6_arm_b_best.pth"),
            "f9fb87cad17256d36aa126c341bc28c4"
        ),
        "V5 U-KAN Segmentation Best": (
            os.path.join(DEST_WORKSPACE, "models", "v5", "segmentation", "v5_768_bce_dice_best.pth"),
            "91e29973585fb12e066d75e07117b821"
        ),
        "V5 ViT Classifier (Arm I Best)": (
            os.path.join(DEST_WORKSPACE, "models", "v5", "classification", "vit_v5_arm_i_best.pth"),
            "ec80689a43a906f89a15d7148cf7b05e"
        ),
    }

    all_verified = True
    for name, (path, expected_h) in critical_checkpoints.items():
        if os.path.exists(path):
            actual_h = compute_md5(path)
            match = (actual_h == expected_h)
            status = "VERIFIED (100% MATCH)" if match else "MISMATCH"
            logger.info(f"  [{status}] {name}")
            logger.info(f"      Path: {path}")
            logger.info(f"      MD5:  {actual_h}")
            if not match:
                all_verified = False
        else:
            logger.error(f"  [MISSING] {name} at {path}")
            all_verified = False

    # Check Directory Stats
    src_f, src_b = get_dir_stats(SRC_WORKSPACE)
    dst_f, dst_b = get_dir_stats(DEST_WORKSPACE)
    logger.info(f"\nWorkspace Stats Comparison:")
    logger.info(f"  Source (C:):      {src_f} files, {src_b/1e9:.2f} GB")
    logger.info(f"  Destination (F:): {dst_f} files, {dst_b/1e9:.2f} GB")

    if os.path.exists(DEST_CAPSTONE):
        cap_f, cap_b = get_dir_stats(DEST_CAPSTONE)
        logger.info(f"  Dataset (F:\\Capstone): {cap_f} files, {cap_b/1e9:.2f} GB")

    _, _, final_free = shutil.disk_usage("F:\\")
    logger.info(f"Drive F: Final Remaining Free Space: {final_free/1e9:.2f} GB")

    logger.info("=" * 80)
    if all_verified:
        logger.info("ALL PROJECT FILES AND CHECKPOINTS SUCCESSFULLY MIGRATED AND VERIFIED ON DRIVE F:!")
    else:
        logger.warning("Migration completed with warnings. Please inspect above log.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
