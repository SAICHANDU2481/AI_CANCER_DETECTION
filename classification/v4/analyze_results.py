import os
import sys
import json
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = r'C:\Users\sys\Desktop\AI_CANCER_DETECTION'
ONEDRIVE_ROOT = r'C:\Users\sys\OneDrive\Desktop\AI_CANCER_DETECTION'
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def analyze():
    # Load metrics json
    with open('results/v4/validation/V4_ABLATION_METRICS.json', 'r') as f:
        metrics_data = json.load(f)
        
    print("==================================================")
    print("V4-A CONTROLLED VALIDATION ABLATION SUMMARY")
    print("==================================================")
    
    arms_summary = []
    
    # Arm A reference
    arms_summary.append({
        'arm_id': 'A',
        'name': 'V3-B Locked Benchmark Reference',
        'scales': 'Local + Med + Lrg (Cross-Attn)',
        'qvec': 'No',
        'gating': 'No',
        'stage1_best_epoch': 7,
        'stage1_acc': 69.32,
        'stage1_f1': 66.65,
        'stage1_auc': 0.6933,
        'stage2_best_epoch': '-',
        'stage2_acc': '-',
        'stage2_f1': '-',
        'stage2_auc': '-',
        'overall_best_acc': 69.32,
        'overall_best_f1': 66.65,
        'overall_best_binary_f1': 57.22,
        'overall_best_auc': 0.6933,
        'overall_best_pr_auc': 0.5814,
        'overall_best_recall': 52.61,
        'overall_best_spec': 80.00,
        'overall_best_prec': 62.71,
        'tp': 111, 'tn': 264, 'fp': 66, 'fn': 100
    })
    
    for arm_id in ['B', 'C', 'D', 'E', 'F', 'G', 'H']:
        d = metrics_data[arm_id]
        desc = d['description']
        s1_ep = d['stage1_best_epoch']
        s2_ep = d['stage2_best_epoch']
        
        s1_best_rec = [h for h in d['stage1_history'] if h['epoch'] == s1_ep][0]
        s2_best_rec = [h for h in d['stage2_history'] if h['epoch'] == s2_ep][0]
        
        m = d['metrics']
        
        arms_summary.append({
            'arm_id': arm_id,
            'name': desc,
            'stage1_best_epoch': s1_ep,
            'stage1_acc': round(s1_best_rec['val_acc'] * 100, 2),
            'stage1_f1': round(s1_best_rec['val_macro_f1'] * 100, 2),
            'stage1_auc': round(s1_best_rec['val_roc_auc'], 4),
            'stage2_best_epoch': s2_ep,
            'stage2_acc': round(s2_best_rec['val_acc'] * 100, 2),
            'stage2_f1': round(s2_best_rec['val_macro_f1'] * 100, 2),
            'stage2_auc': round(s2_best_rec['val_roc_auc'], 4),
            's2_full_metrics': m
        })
        
    for s in arms_summary:
        print(f"Arm {s['arm_id']}: {s.get('name', '')}")
        print(f"  Stage 1 (Epoch {s['stage1_best_epoch']}): Acc={s['stage1_acc']}%, Macro-F1={s['stage1_f1']}%, AUC={s['stage1_auc']}")
        print(f"  Stage 2 (Epoch {s['stage2_best_epoch']}): Acc={s['stage2_acc']}%, Macro-F1={s['stage2_f1']}%, AUC={s['stage2_auc']}")

if __name__ == '__main__':
    analyze()
