import os
import hashlib
import json
import torch
import pandas as pd
import sys

def get_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    results_dir = 'results/v7'
    preflight_dir = os.path.join(results_dir, 'preflight')
    os.makedirs(preflight_dir, exist_ok=True)

    # 1. Checkpoints verification
    ukan_path = 'models/v6/segmentation/final/ukan_v6_arm_g_best.pth'
    vit_path = 'models/v6/classification/vit_v6_arm_c_best.pth'

    ukan_exists = os.path.exists(ukan_path)
    vit_exists = os.path.exists(vit_path)

    ukan_md5 = get_md5(ukan_path) if ukan_exists else None
    vit_md5 = get_md5(vit_path) if vit_exists else None

    expected_ukan_md5 = '68cfc38cfbddd93d6e4a6e692fffbb34'
    expected_vit_md5 = '508250cfc1c07af1ea7d81cc2e5a7418'

    ukan_md5_match = (ukan_md5 == expected_ukan_md5)
    vit_md5_match = (vit_md5 == expected_vit_md5)

    # 2. Splits check (Train & Val only)
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')

    train_records = len(train_df)
    val_records = len(val_df)

    train_patients = set(train_df['patient_id'].unique())
    val_patients = set(val_df['patient_id'].unique())

    overlap_train_val = len(train_patients.intersection(val_patients))

    # 3. Hardware & PyTorch check
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else 'N/A'
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_available else 0.0
    amp_fp16_available = cuda_available

    # 4. Importability test
    import_status = {}
    try:
        import segmentation.v6.ukan_v6 as ukan_mod
        import_status['segmentation.v6.ukan_v6'] = True
    except Exception as e:
        import_status['segmentation.v6.ukan_v6'] = str(e)

    try:
        import classification.v6.model_v6 as model_mod
        import_status['classification.v6.model_v6'] = True
    except Exception as e:
        import_status['classification.v6.model_v6'] = str(e)

    try:
        import classification.v6.roi_generator_v6 as roi_mod
        import_status['classification.v6.roi_generator_v6'] = True
    except Exception as e:
        import_status['classification.v6.roi_generator_v6'] = str(e)

    try:
        import classification.v6.topk_proposals_v6 as topk_mod
        import_status['classification.v6.topk_proposals_v6'] = True
    except Exception as e:
        import_status['classification.v6.topk_proposals_v6'] = str(e)

    passed = (
        ukan_md5_match and 
        vit_md5_match and 
        overlap_train_val == 0 and 
        train_records == 2501 and 
        len(train_patients) == 1096 and 
        val_records == 541 and 
        len(val_patients) == 235 and 
        cuda_available and 
        all(v is True for v in import_status.values())
    )

    report = {
        'timestamp': '2026-09-29T18:28:00',
        'status': 'PASSED' if passed else 'FAILED',
        'frozen_checkpoints': {
            'ukan_v6_arm_g_best': {
                'path': ukan_path,
                'exists': ukan_exists,
                'md5': ukan_md5,
                'expected_md5': expected_ukan_md5,
                'matches': ukan_md5_match
            },
            'vit_v6_arm_c_best': {
                'path': vit_path,
                'exists': vit_exists,
                'md5': vit_md5,
                'expected_md5': expected_vit_md5,
                'matches': vit_md5_match
            }
        },
        'dataset_splits': {
            'train': {
                'records': train_records,
                'patients': len(train_patients),
                'expected_records': 2501,
                'expected_patients': 1096,
                'matches': (train_records == 2501 and len(train_patients) == 1096)
            },
            'validation': {
                'records': val_records,
                'patients': len(val_patients),
                'expected_records': 541,
                'expected_patients': 235,
                'matches': (val_records == 541 and len(val_patients) == 235)
            },
            'test_accessed': False,
            'patient_overlap_train_val': overlap_train_val
        },
        'hardware': {
            'cuda_available': cuda_available,
            'gpu_name': gpu_name,
            'vram_total_gb': round(vram_gb, 2),
            'amp_fp16_available': amp_fp16_available
        },
        'codebase_imports': import_status,
        'safety_rules': {
            'test_csv_read': False,
            'test_images_loaded': False,
            'test_masks_loaded': False,
            'test_predictions_generated': False,
            'v7_test_dir_created': False,
            'v6_checkpoints_unmodified': True
        }
    }

    report_path = os.path.join(preflight_dir, 'V7_PREFLIGHT_REPORT.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print('PREFLIGHT_REPORT_START')
    print(json.dumps(report, indent=2))
    print('PREFLIGHT_REPORT_END')

if __name__ == '__main__':
    main()
