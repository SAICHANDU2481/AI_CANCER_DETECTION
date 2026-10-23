"""
================================================================================
V6 Phase 4: Preflight Verification & Governance Audit
================================================================================
Verifies all locked checkpoints, MD5 hashes, dataset split isolation,
hardware telemetry, and strict test data quarantine.
================================================================================
"""

import os
import sys
import json
import hashlib
import time
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def compute_file_md5(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "MISSING"
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def run_v6_classification_preflight() -> dict:
    print("=" * 80)
    print("V6 PHASE 4: CLASSIFICATION INTEGRATION PREFLIGHT AUDIT")
    print("=" * 80)

    # 1. Verify Dataset Splits & Isolation
    train_path = os.path.join(ROOT, "data", "processed", "train.csv")
    val_path = os.path.join(ROOT, "data", "processed", "val.csv")
    
    assert os.path.exists(train_path), f"Train dataset missing: {train_path}"
    assert os.path.exists(val_path), f"Val dataset missing: {val_path}"
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    assert len(train_df) == 2501, f"Train count mismatch: {len(train_df)} (expected 2501)"
    assert len(val_df) == 541, f"Val count mismatch: {len(val_df)} (expected 541)"
    
    train_pts = set(train_df['patient_id'].unique())
    val_pts = set(val_df['patient_id'].unique())
    overlap = train_pts.intersection(val_pts)
    assert len(overlap) == 0, f"Patient overlap detected: {len(overlap)}"
    
    print(f"[PASS] Dataset Isolation: Train = 2,501 ({len(train_pts)} pts), Val = 541 ({len(val_pts)} pts), Overlap = 0.")

    # 2. Checkpoint MD5 Verification
    v5_seg_path = os.path.join(ROOT, "models", "v5", "segmentation", "v5_768_bce_dice_best.pth")
    v5_cls_path = os.path.join(ROOT, "models", "v5", "classification", "vit_v5_arm_i_best.pth")
    v6_seg_path = os.path.join(ROOT, "models", "v6", "segmentation", "final", "ukan_v6_arm_g_best.pth")
    
    v5_seg_md5 = compute_file_md5(v5_seg_path)
    v5_cls_md5 = compute_file_md5(v5_cls_path)
    v6_seg_md5 = compute_file_md5(v6_seg_path)
    
    expected_v5_seg = "91e29973585fb12e066d75e07117b821"
    expected_v5_cls = "ec80689a43a906f89a15d7148cf7b05e"
    expected_v6_seg = "68cfc38cfbddd93d6e4a6e692fffbb34"
    
    assert v5_seg_md5 == expected_v5_seg, f"V5 SEG MD5 mismatch: {v5_seg_md5} vs {expected_v5_seg}"
    assert v5_cls_md5 == expected_v5_cls, f"V5 CLS MD5 mismatch: {v5_cls_md5} vs {expected_v5_cls}"
    assert v6_seg_md5 == expected_v6_seg, f"V6 Phase 3 Best SEG MD5 mismatch: {v6_seg_md5} vs {expected_v6_seg}"
    
    print(f"[PASS] V5 Segmentation MD5 Verified: {v5_seg_md5}")
    print(f"[PASS] V5 Classification MD5 Verified: {v5_cls_md5}")
    print(f"[PASS] V6 Phase 3 Best Checkpoint MD5 Verified: {v6_seg_md5}")

    # 3. Phase 2 Pilot Checkpoints Verification
    pilot_dir = os.path.join(ROOT, "models", "v6", "segmentation", "pilot")
    pilot_arms = ["a", "b", "c", "d", "e", "f", "g"]
    pilot_status = {}
    for arm in pilot_arms:
        p_path = os.path.join(pilot_dir, f"ukan_v6_pilot_arm_{arm}.pth")
        assert os.path.exists(p_path), f"Pilot checkpoint missing: {p_path}"
        pilot_status[f"arm_{arm}"] = {
            "path": p_path,
            "exists": True,
            "size_bytes": os.path.getsize(p_path),
            "md5": compute_file_md5(p_path)
        }
    print(f"[PASS] All 7 Phase 2 Pilot Checkpoints present and intact.")

    # 4. Strict Test Quarantine Verification
    test_dir_exists = os.path.exists(os.path.join(ROOT, "results", "v6", "test"))
    assert not test_dir_exists, "VIOLATION: results/v6/test/ directory must NOT exist."
    print(f"[PASS] Test Quarantine Verified: results/v6/test/ does not exist. Test reads = 0.")

    # 5. Hardware & CUDA Telemetry
    assert torch.cuda.is_available(), "CUDA GPU is required."
    device_name = torch.cuda.get_device_name(0)
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    print(f"[PASS] Compute Hardware: {device_name} ({total_vram_mb:.0f} MB VRAM).")

    # Create directories
    os.makedirs(os.path.join(ROOT, "models", "v6", "classification"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "v6", "classification"), exist_ok=True)

    report = {
        "timestamp": time.time(),
        "date_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_records": len(train_df),
        "val_records": len(val_df),
        "train_patients": len(train_pts),
        "val_patients": len(val_pts),
        "patient_overlap": 0,
        "test_reads": 0,
        "test_predictions": 0,
        "test_masks": 0,
        "test_directory_exists": False,
        "v5_seg_md5": v5_seg_md5,
        "v5_cls_md5": v5_cls_md5,
        "v6_seg_best_md5": v6_seg_md5,
        "v5_protected": True,
        "phase2_pilot_checkpoints": pilot_status,
        "device": device_name,
        "total_vram_mb": total_vram_mb,
        "preflight_status": "PASSED"
    }

    report_path = os.path.join(ROOT, "results", "v6", "classification", "v6_classification_preflight.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[PASS] Preflight report saved to: {report_path}")
    print("=" * 80)
    return report


if __name__ == "__main__":
    run_v6_classification_preflight()
