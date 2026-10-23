import os
import sys
import json
import time
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = r'C:\Users\sys\Desktop\AI_CANCER_DETECTION'
ONEDRIVE_ROOT = r'C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION'
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from classification.v4.train_v4 import run_staged_training, set_seed

logger = logging.getLogger('AblationV4')
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

def run_ablation_suite():
    logger.info("==================================================")
    logger.info("STARTING V4-A CONTROLLED VALIDATION ABLATION SUITE (ARMS A - H)")
    logger.info("==================================================")
    
    os.makedirs('results/v4/validation', exist_ok=True)
    os.makedirs(os.path.join(ONEDRIVE_ROOT, 'results/v4/validation'), exist_ok=True)
    
    # 1. Arm A: Locked V3-B Reference Baseline
    ablation_results = [
        {
            'Ablation_ID': 'A',
            'Configuration': 'V3-B Locked Benchmark Reference',
            'Scales_Used': 'Local + Med + Lrg (Cross-Attn)',
            'Quality_Vector': 'No',
            'Dynamic_Gating': 'No',
            'Val_Accuracy_%': 69.32,
            'Val_Macro_F1_%': 66.65,
            'Val_Binary_F1_%': 57.22,
            'Val_ROC_AUC': 0.6933,
            'Val_PR_AUC': 0.5814,
            'Val_Malignant_Recall_%': 52.61,
            'Val_Specificity_%': 80.00,
            'Val_Precision_%': 62.71,
            'TP': 111, 'TN': 264, 'FP': 66, 'FN': 100
        }
    ]
    
    # Define Ablation Experiments B through H
    experiments = [
        ('B', 'Local ROI Only', ('local',), False, False, 0.40),
        ('C', 'Local + Medium Context', ('local', 'medium'), False, False, 0.40),
        ('D', 'Local + Medium + Large Context', ('local', 'medium', 'large'), False, False, 0.40),
        ('E', 'Local + Med + Lrg + Context (Visual Only)', ('local', 'medium', 'large', 'context'), False, False, 0.40),
        ('F', 'Multi-Scale Fusion (No Quality Vector)', ('local', 'medium', 'large', 'context'), False, False, 0.40),
        ('G', 'Multi-Scale Fusion + 16-Dim Quality Vector (Primary V4-A)', ('local', 'medium', 'large', 'context'), True, False, 0.40),
        ('H', 'Quality-Gated Dynamic Routing (tau=0.40)', ('local', 'medium', 'large', 'context'), True, True, 0.40)
    ]
    
    best_overall_f1 = -1.0
    best_arm_id = None
    best_arm_res = None
    all_ablation_records = {}
    
    for arm_id, desc, scales, use_q, use_gate, tau in experiments:
        logger.info(f"\n>>> RUNNING ABLATION ARM {arm_id}: {desc} <<<")
        res = run_staged_training(
            scales_used=scales,
            use_quality_vector=use_q,
            enable_gating=use_gate,
            gating_threshold=tau,
            batch_size=16,
            seed=42
        )
        
        m = res['best_val_metrics']
        ablation_results.append({
            'Ablation_ID': arm_id,
            'Configuration': desc,
            'Scales_Used': ' + '.join(scales),
            'Quality_Vector': 'Yes (16-dim)' if use_q else 'No',
            'Dynamic_Gating': f'Yes (tau={tau:.2f})' if use_gate else 'No',
            'Val_Accuracy_%': round(m['accuracy'] * 100, 2),
            'Val_Macro_F1_%': round(m['macro_f1'] * 100, 2),
            'Val_Binary_F1_%': round(m['binary_f1'] * 100, 2),
            'Val_ROC_AUC': round(m['roc_auc'], 4),
            'Val_PR_AUC': round(m['pr_auc'], 4),
            'Val_Malignant_Recall_%': round(m['sensitivity_malignant_recall'] * 100, 2),
            'Val_Specificity_%': round(m['specificity'] * 100, 2),
            'Val_Precision_%': round(m['precision'] * 100, 2),
            'TP': m['tp'], 'TN': m['tn'], 'FP': m['fp'], 'fn': m['fn']
        })
        
        all_ablation_records[arm_id] = {
            'description': desc,
            'metrics': m,
            'stage1_best_epoch': res['stage1_best_epoch'],
            'stage2_best_epoch': res['stage2_best_epoch'],
            'stage1_history': res['stage1_history'],
            'stage2_history': res['stage2_history']
        }
        
        if m['macro_f1'] > best_overall_f1:
            best_overall_f1 = m['macro_f1']
            best_arm_id = arm_id
            best_arm_res = res
            
    # Save Ablation Results Table & JSON
    ablation_df = pd.DataFrame(ablation_results)
    ablation_csv_path = 'results/v4/validation/V4_ABLATION_RESULTS.csv'
    ablation_df.to_csv(ablation_csv_path, index=False)
    ablation_df.to_csv(os.path.join(ONEDRIVE_ROOT, ablation_csv_path), index=False)
    
    metrics_json_path = 'results/v4/validation/V4_ABLATION_METRICS.json'
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_ablation_records, f, indent=2)
    with open(os.path.join(ONEDRIVE_ROOT, metrics_json_path), 'w', encoding='utf-8') as f:
        json.dump(all_ablation_records, f, indent=2)
        
    logger.info(f"Ablation Suite Completed! Best Validation Arm: {best_arm_id} with Macro-F1 = {best_overall_f1*100:.2f}%")
    return ablation_df, best_arm_id, best_arm_res

if __name__ == '__main__':
    run_ablation_suite()
