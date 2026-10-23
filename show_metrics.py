"""
================================================================================
Locked V6 Held-Out Test Benchmark & Accuracy Viewer
================================================================================
Displays the locked static performance metrics in the terminal without accessing
raw test data or modifying any checkpoints.
================================================================================
"""

import os
import json

METRICS_PATH = os.path.join(os.path.dirname(__file__), "results", "v6", "test", "v6_test_metrics.json")

def display_metrics():
    if not os.path.exists(METRICS_PATH):
        print(f"Error: Metrics file not found at {METRICS_PATH}")
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    p = data.get("primary_metrics", {})
    c = data.get("cohort", {})
    sub = data.get("subgroup_metrics", {})
    ci = data.get("confidence_intervals_95", {})

    print("\n" + "=" * 80)
    print("        LOCKED V6 HELD-OUT TEST BENCHMARK -- FINAL ACCURACY & METRICS")
    print("=" * 80)
    print(f" Test Partition: {c.get('total_test_records', 526)} scans | {c.get('unique_patients', 235)} patients (Isolated)")
    print(f" Evaluation Date: {data.get('date_str', '2026-09-27')}")
    print("-" * 80)
    
    print(f"  {'Metric':<24} {'Value':<12} {'95% Confidence Interval':<28}")
    print("-" * 80)
    print(f"  {'Accuracy':<24} {p.get('accuracy', 0)*100:>6.2f}%      [{ci.get('accuracy', [0,0])[0]*100:.2f}% - {ci.get('accuracy', [0,0])[1]*100:.2f}%]")
    print(f"  {'Precision':<24} {p.get('precision', 0)*100:>6.2f}%      [{ci.get('precision', [0,0])[0]*100:.2f}% - {ci.get('precision', [0,0])[1]*100:.2f}%]")
    print(f"  {'Recall / Sensitivity':<24} {p.get('recall_sensitivity', 0)*100:>6.2f}%      [{ci.get('recall_sensitivity', [0,0])[0]*100:.2f}% - {ci.get('recall_sensitivity', [0,0])[1]*100:.2f}%]")
    print(f"  {'Specificity':<24} {p.get('specificity', 0)*100:>6.2f}%      [{ci.get('specificity', [0,0])[0]*100:.2f}% - {ci.get('specificity', [0,0])[1]*100:.2f}%]")
    print(f"  {'Binary F1-Score':<24} {p.get('f1_binary', 0)*100:>6.2f}%      [{ci.get('f1_binary', [0,0])[0]*100:.2f}% - {ci.get('f1_binary', [0,0])[1]*100:.2f}%]")
    print(f"  {'Macro F1-Score':<24} {p.get('macro_f1', 0)*100:>6.2f}%      [{ci.get('macro_f1', [0,0])[0]*100:.2f}% - {ci.get('macro_f1', [0,0])[1]*100:.2f}%]")
    print(f"  {'ROC-AUC':<24} {p.get('roc_auc', 0):>8.4f}      [{ci.get('roc_auc', [0,0])[0]:.4f} - {ci.get('roc_auc', [0,0])[1]:.4f}]")
    print(f"  {'PR-AUC':<24} {p.get('pr_auc', 0):>8.4f}      [{ci.get('pr_auc', [0,0])[0]:.4f} - {ci.get('pr_auc', [0,0])[1]:.4f}]")
    print("-" * 80)
    
    print("\n  CONFUSION MATRIX COUNTS:")
    print(f"    True Negatives (TN):  {p.get('tn', 195):<6} | False Positives (FP): {p.get('fp', 116)}")
    print(f"    False Negatives (FN): {p.get('fn', 71):<6} | True Positives (TP):  {p.get('tp', 144)}")

    if sub:
        print("\n  SUBGROUP ANALYSIS:")
        print("-" * 80)
        print(f"  {'Cohort':<24} {'Scans':<8} {'Accuracy':<12} {'Sensitivity':<14} {'ROC-AUC':<10}")
        print("-" * 80)
        mass = sub.get("mass", {})
        calc = sub.get("calcification", {})
        if mass:
            print(f"  {'Masses Only':<24} {mass.get('total', 267):<8} {mass.get('accuracy', 0)*100:>6.2f}%     {mass.get('recall_sensitivity', 0)*100:>6.2f}%        {mass.get('roc_auc', 0):.4f}")
        if calc:
            print(f"  {'Calcifications Only':<24} {calc.get('total', 259):<8} {calc.get('accuracy', 0)*100:>6.2f}%     {calc.get('recall_sensitivity', 0)*100:>6.2f}%        {calc.get('roc_auc', 0):.4f}")
    
    print("=" * 80 + "\n")

if __name__ == "__main__":
    display_metrics()
