"""
Amazon ML Challenge 2026 — Experiment Tracking Infrastructure
Standardized experiment metrics recorder producing machine-readable JSON and human-readable TSV.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime


def get_git_commit(repo_root=None):
    """Retrieves current git commit hash and branch."""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)
        commit = res.stdout.strip()
        cmd_branch = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        res_b = subprocess.run(cmd_branch, cwd=repo_root, capture_output=True, text=True)
        branch = res_b.stdout.strip()
        return f"{branch}@{commit}"
    except Exception:
        return "unknown"


def record_experiment_result(
    output_dir,
    experiment_id,
    feature_list,
    threshold,
    fallback_threshold=None,
    name_threshold=None,
    address_dropout=0.0,
    candidate_count=0,
    num_s1_entities=0,
    num_predicted_matches=0,
    val_precision=0.0,
    val_recall=0.0,
    val_f05=0.0,
    num_false_positives=0,
    num_false_negatives=0,
    empty_match_pct=0.0,
    max_matches_per_s1=0,
    runtime_seconds=0.0,
    peak_disk_gb=0.0,
    notes=""
):
    """
    Appends an experiment result to:
      1. output/experiment_results.json
      2. output/experiment_results.tsv
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "experiment_results.json")
    tsv_path = os.path.join(output_dir, "experiment_results.tsv")
    
    commit = get_git_commit(os.path.dirname(os.path.dirname(os.path.abspath(output_dir))))
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    record = {
        "timestamp": timestamp,
        "experiment_id": str(experiment_id),
        "git_commit": commit,
        "feature_count": len(feature_list) if feature_list else 0,
        "feature_list": list(feature_list) if feature_list else [],
        "threshold": float(threshold),
        "fallback_threshold": float(fallback_threshold) if fallback_threshold is not None else None,
        "name_threshold": float(name_threshold) if name_threshold is not None else None,
        "address_dropout_rate": float(address_dropout),
        "candidate_count": int(candidate_count),
        "num_s1_entities": int(num_s1_entities),
        "num_predicted_matches": int(num_predicted_matches),
        "validation_precision": round(float(val_precision), 5),
        "validation_recall": round(float(val_recall), 5),
        "validation_macro_f05": round(float(val_f05), 5),
        "false_positives": int(num_false_positives),
        "false_negatives": int(num_false_negatives),
        "empty_match_s1_pct": round(float(empty_match_pct), 2),
        "max_matches_per_s1": int(max_matches_per_s1),
        "runtime_seconds": round(float(runtime_seconds), 1),
        "peak_disk_gb": round(float(peak_disk_gb), 2),
        "notes": str(notes)
    }
    
    # 1. Update JSON list
    all_results = []
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
                if not isinstance(all_results, list):
                    all_results = []
        except Exception:
            all_results = []
            
    all_results.append(record)
    for attempt in range(5):
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2)
            break
        except PermissionError:
            time.sleep(0.05)
        
    # 2. Update TSV table
    tsv_headers = [
        "timestamp", "experiment_id", "git_commit", "threshold", "fallback_threshold",
        "name_threshold", "address_dropout_rate", "num_s1_entities", "num_predicted_matches",
        "val_precision", "val_recall", "val_macro_f05", "false_positives", "false_negatives",
        "empty_match_s1_pct", "max_matches_per_s1", "runtime_seconds", "notes"
    ]
    
    file_exists = os.path.exists(tsv_path) and os.path.getsize(tsv_path) > 0
    row = [
        str(record["timestamp"]),
        str(record["experiment_id"]),
        str(record["git_commit"]),
        str(record["threshold"]),
        str(record["fallback_threshold"] if record["fallback_threshold"] is not None else ""),
        str(record["name_threshold"] if record["name_threshold"] is not None else ""),
        str(record["address_dropout_rate"]),
        str(record["num_s1_entities"]),
        str(record["num_predicted_matches"]),
        f"{record['validation_precision']:.5f}",
        f"{record['validation_recall']:.5f}",
        f"{record['validation_macro_f05']:.5f}",
        str(record["false_positives"]),
        str(record["false_negatives"]),
        f"{record['empty_match_s1_pct']:.2f}%",
        str(record["max_matches_per_s1"]),
        f"{record['runtime_seconds']:.1f}s",
        str(record["notes"])
    ]
    
    for attempt in range(5):
        try:
            with open(tsv_path, 'a', encoding='utf-8') as f:
                if not file_exists:
                    f.write("\t".join(tsv_headers) + "\n")
                    file_exists = True
                f.write("\t".join(row) + "\n")
            break
        except PermissionError:
            time.sleep(0.05)
        
    print(f"  [Experiment Tracker] Recorded {experiment_id} results to {json_path} and {tsv_path}")
    return record
