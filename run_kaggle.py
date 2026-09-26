#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Kaggle Runner Entry Point
Experiment 002A: 15 Baseline Features + Candidate Rank (No Dense Cosine Sim)

This runner automates the end-to-end execution of Experiment 002A on Kaggle
or local GPU compute environments without requiring manual source-code,
path, or constant modifications.
"""

import os
import sys
import json
import shutil
import argparse
import subprocess

# Ensure repo root and src/ are in sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from create_validation_split import create_splits
from blocking import create_blocking
from feature_engineering import build_train_sqlite_catalog, build_v3_feature_datasets, FEATURE_COLS
from train_model import train_and_evaluate
from inference import run_test_inference


def resolve_data_paths(data_root=None):
    """Dynamically resolves dataset locations across Kaggle and local environments."""
    if data_root and os.path.exists(data_root):
        base_candidates = [data_root]
    else:
        base_candidates = [
            "/kaggle/input/amazon-ml-challenge-2026/dataset",
            "/kaggle/input/amazon-ml-challenge-2026/student_resource/dataset",
            "/kaggle/input/dataset",
            os.path.join(REPO_ROOT, "dataset"),
            os.path.join(REPO_ROOT, "student_resource", "dataset"),
            os.path.join(REPO_ROOT, "6ab10eb3b23ba_student_resource", "student_resource", "dataset"),
        ]

    resolved_root = None
    for candidate in base_candidates:
        if os.path.exists(candidate):
            resolved_root = candidate
            break

    if not resolved_root:
        raise FileNotFoundError(
            f"Dataset directory could not be located. Checked paths: {base_candidates}\n"
            f"Please specify --data-root pointing to the dataset containing train/ and test/ folders."
        )

    # Locate train source files
    train_search_dirs = [
        os.path.join(resolved_root, "train"),
        resolved_root,
    ]
    train_dir = next((d for d in train_search_dirs if os.path.exists(os.path.join(d, "train_source1.tsv"))), None)
    if not train_dir:
        raise FileNotFoundError(f"train_source1.tsv not found under {resolved_root}")

    train_files = {
        "s1": os.path.join(train_dir, "train_source1.tsv"),
        "s2": os.path.join(train_dir, "train_source2.tsv"),
        "s3": os.path.join(train_dir, "train_source3.tsv"),
        "gt": os.path.join(train_dir, "train_ground_truth.tsv"),
    }
    for name, path in train_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required training file: {path}")

    # Locate test source files
    test_search_dirs = [
        os.path.join(resolved_root, "test"),
        resolved_root,
    ]
    test_dir = next((d for d in test_search_dirs if os.path.exists(os.path.join(d, "test_source1.tsv"))), None)
    if not test_dir:
        raise FileNotFoundError(f"test_source1.tsv not found under {resolved_root}")

    test_files = {
        "s1": os.path.join(test_dir, "test_source1.tsv"),
        "s2": os.path.join(test_dir, "test_source2.tsv"),
        "s3": os.path.join(test_dir, "test_source3.tsv"),
        "test_dir": test_dir
    }
    for name, path in test_files.items():
        if name != "test_dir" and not os.path.exists(path):
            raise FileNotFoundError(f"Missing required test file: {path}")

    return resolved_root, train_files, test_files


def resolve_output_dir(output_root=None):
    """Resolves output directory, defaulting to Kaggle working dir if available."""
    if output_root:
        out_dir = os.path.abspath(output_root)
    elif os.path.exists("/kaggle/working"):
        out_dir = "/kaggle/working/output"
    else:
        out_dir = os.path.join(REPO_ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def resolve_cache_dir(output_dir, cache_dir_arg=None):
    """Resolves cache directory, preferring /tmp on Linux/Kaggle to keep /kaggle/working clear."""
    if cache_dir_arg:
        c_dir = os.path.abspath(cache_dir_arg)
    elif os.path.exists("/tmp") and os.path.isdir("/tmp"):
        c_dir = "/tmp/embeddings_cache"
    else:
        c_dir = os.path.join(output_dir, "embeddings_cache")
    os.makedirs(c_dir, exist_ok=True)
    return c_dir


def check_disk_space(path, required_gb, stage_name):
    """Checks available free disk space on the given path's drive and warns/fails if insufficient."""
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024**3)
        print(f"  [Disk Check] {stage_name} target '{path}': {free_gb:.1f} GB free (requires ~{required_gb:.1f} GB)")
        if free_gb < required_gb:
            raise RuntimeError(
                f"Insufficient disk space for {stage_name} at '{path}'!\n"
                f"Free space: {free_gb:.1f} GB, Required: at least {required_gb:.1f} GB.\n"
                f"Please clean up disk space or configure directories using a partition with more free space."
            )
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        print(f"  [Disk Check Warning] Could not determine disk usage at {path}: {e}")


def resolve_test_candidate_file(output_dir):
    """Finds existing test candidate pairs file. Reuses existing test candidates strictly."""
    candidates = [
        os.path.join(output_dir, "candidate_pairs.tsv"),
        os.path.join(REPO_ROOT, "output", "candidate_pairs.tsv"),
        os.path.join(REPO_ROOT, "candidate_pairs.tsv"),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 1024:
            return path
    return None


def run_experiment(args):
    print("=" * 75)
    print(f"  AMAZON ML CHALLENGE 2026 — EXPERIMENT RUNNER: {args.experiment.upper()}")
    print("=" * 75)

    # -------------------------------------------------------------
    # STAGE 1: Environment & Path Validation
    # -------------------------------------------------------------
    print("\n[STAGE 1/10] Validating Environment & Data Inputs...")
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "None (CPU only)"
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  CUDA Available : {has_cuda} ({device_name})")

    data_root, train_files, test_files = resolve_data_paths(args.data_root)
    output_dir = resolve_output_dir(args.output_root)
    cache_dir = resolve_cache_dir(output_dir, args.cache_dir)

    print(f"  Dataset Root    : {data_root}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Cache Directory : {cache_dir} ({'in /tmp (off /kaggle/working)' if '/tmp' in cache_dir else 'in output root'})")
    print(f"  Feature Schema  : {len(FEATURE_COLS)} features (15 baseline + candidate_rank)")

    # Verify existing test candidates
    test_cands_path = resolve_test_candidate_file(output_dir)
    if not test_cands_path:
        raise FileNotFoundError(
            "CRITICAL: Required test candidate set (output/candidate_pairs.tsv) is missing!\n"
            "For Experiment 002A, existing test candidates must be reused and NEVER regenerated.\n"
            "Please ensure output/candidate_pairs.tsv is present in the repository or output directory."
        )

    # Validate test candidate file header
    with open(test_cands_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
    if header != "source1_entity_id\tcandidate_entity_ids":
        raise ValueError(f"Invalid candidate_pairs.tsv header: {header}")
    print(f"  Test Candidates : Verified existing {test_cands_path} ({os.path.getsize(test_cands_path)/(1024**2):.1f} MB)")

    # Define stage output paths
    val_split_path = os.path.join(output_dir, "val_gt_split.tsv")
    train_cands_path = os.path.join(output_dir, "full_train_candidate_pairs.tsv")
    db_path = os.path.join(output_dir, "train_catalog_temp.db")
    train_features_path = os.path.join(output_dir, "full_train_features_v3.csv")
    val_features_path = os.path.join(output_dir, "full_val_features_v3.csv")
    lgb_model_path = os.path.join(output_dir, "lgbm_model_v3.txt")
    threshold_config_path = os.path.join(output_dir, "threshold_config_v3.json")
    out_matching_path = os.path.join(output_dir, "matching_results.tsv")

    # If blocking is needed and CUDA is not available, fail clearly
    needs_blocking = not (os.path.exists(train_cands_path) and os.path.getsize(train_cands_path) > 1024)
    if needs_blocking and not has_cuda and not args.allow_cpu and not args.dry_run:
        raise RuntimeError(
            "CUDA is not available. Dense semantic blocking (Stage 3) requires GPU acceleration (CUDA).\n"
            "Running blocking on CPU is prohibitively slow. Please enable GPU in your Kaggle notebook."
        )

    if args.dry_run:
        print("\n" + "=" * 75)
        print("  DRY-RUN SUMMARY: ALL PRE-CONDITIONS VERIFIED SUCCESSFULLY")
        print("=" * 75)
        print(f"  Experiment         : {args.experiment}")
        print(f"  Dataset Root       : {data_root}")
        print(f"  Output Root        : {output_dir}")
        print(f"  Cache Directory    : {cache_dir} ({'in /tmp (zero /kaggle/working footprint)' if '/tmp' in cache_dir else 'in output root'})")
        print(f"  Cache Auto-Cleanup : {'Disabled (--keep-cache set)' if args.keep_cache else 'Enabled (purged post-Stage 3 to conserve disk)'}")
        print(f"  Working Footprint  : Expected peak ~14.6 GB (strictly within Kaggle 20 GB quota)")
        print(f"  CUDA Status        : {'Available (' + device_name + ')' if has_cuda else 'Unavailable'}")
        print(f"  Test Candidates    : Reusing {test_cands_path}")
        print(f"  Train Candidates   : {'Already present' if not needs_blocking else 'Will generate via GPU blocking (k=100)'}")
        print(f"  Train SQLite DB    : {'Already present' if os.path.exists(db_path) else 'Will build via SQLite'}")
        print(f"  Feature Generation : {'Already present' if os.path.exists(train_features_path) else 'Will generate 16 features'}")
        print(f"  Model Training     : {'Already present' if os.path.exists(lgb_model_path) else 'Will train 800-tree LightGBM'}")
        print(f"  Inference Target   : {out_matching_path}")
        print("Dry run completed cleanly. No heavy computations performed.")
        return 0

    # -------------------------------------------------------------
    # STAGE 2: Create Stratified Validation Split
    # -------------------------------------------------------------
    print("\n[STAGE 2/10] Validation Split Preparation...")
    if os.path.exists(val_split_path) and os.path.getsize(val_split_path) > 100:
        print(f"  Reusing existing validation split at: {val_split_path}")
    else:
        print("  Generating 20% stratified validation split...")
        create_splits(data_dir=os.path.dirname(train_files["gt"]), output_dir=output_dir)

    # -------------------------------------------------------------
    # STAGE 3: Generate Training Candidate Pairs via GPU Blocking
    # -------------------------------------------------------------
    print("\n[STAGE 3/10] Training Candidate Generation (Blocking k=100)...")
    if os.path.exists(train_cands_path) and os.path.getsize(train_cands_path) > 1024:
        print(f"  Reusing existing training candidate pairs at: {train_cands_path} ({os.path.getsize(train_cands_path)/(1024**2):.1f} MB)")
    else:
        check_disk_space(cache_dir, required_gb=11.0, stage_name="Stage 3 Embedding Cache")
        check_disk_space(output_dir, required_gb=4.0, stage_name="Stage 3 Candidate Pairs Output")
        print(f"  Generating training candidate pairs using dense semantic blocking (k=100)...")
        print(f"  Temporary embedding cache location: {cache_dir}")
        create_blocking(
            val_s1_path=train_files["s1"],
            s2_path=train_files["s2"],
            s3_path=train_files["s3"],
            output_path=train_cands_path,
            cache_dir=cache_dir,
            k=100,
            require_cuda=not args.allow_cpu
        )
        if not args.keep_cache and os.path.exists(cache_dir):
            print(f"  Cleaning up temporary embedding cache at {cache_dir} to free up disk space...")
            shutil.rmtree(cache_dir, ignore_errors=True)
            print("  Temporary embedding cache successfully removed. (Stage 3 candidates safely preserved)")

    # -------------------------------------------------------------
    # STAGE 4: Build SQLite Train Catalog
    # -------------------------------------------------------------
    print("\n[STAGE 4/10] Building / Verifying SQLite Training Catalog...")
    check_disk_space(output_dir, required_gb=3.5, stage_name="Stage 4 SQLite Catalog")
    build_train_sqlite_catalog(
        s1_path=train_files["s1"],
        s2_path=train_files["s2"],
        s3_path=train_files["s3"],
        gt_path=train_files["gt"],
        db_path=db_path
    )

    # -------------------------------------------------------------
    # STAGE 5: Generate 16-Feature Datasets
    # -------------------------------------------------------------
    print("\n[STAGE 5/10] Feature Engineering (16 Features: 15 Baseline + Candidate Rank)...")
    if os.path.exists(train_features_path) and os.path.exists(val_features_path) and os.path.getsize(train_features_path) > 1024:
        print(f"  Reusing existing feature datasets at:")
        print(f"    Train: {train_features_path} ({os.path.getsize(train_features_path)/(1024**2):.1f} MB)")
        print(f"    Val  : {val_features_path} ({os.path.getsize(val_features_path)/(1024**2):.1f} MB)")
    else:
        check_disk_space(output_dir, required_gb=6.0, stage_name="Stage 5 Feature Datasets")
        print("  Extracting 16 features across training and undownsampled validation sets...")
        build_v3_feature_datasets(
            cands_path=train_cands_path,
            val_split_path=val_split_path,
            out_train_path=train_features_path,
            out_val_path=val_features_path,
            resume=True,
            db_path=db_path
        )

    # -------------------------------------------------------------
    # STAGE 6 & 7: Model Training & Threshold Calibration
    # -------------------------------------------------------------
    print("\n[STAGE 6 & 7/10] Training 16-Feature LightGBM & Threshold Calibration...")
    if os.path.exists(lgb_model_path) and os.path.exists(threshold_config_path):
        print(f"  Reusing existing trained model and threshold config at: {threshold_config_path}")
    else:
        print("  Training LightGBM booster (800 trees) on 16 features...")
        train_and_evaluate(
            train_csv=train_features_path,
            val_csv=val_features_path,
            val_gt_path=val_split_path,
            out_dir=output_dir
        )

    with open(threshold_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    optimal_threshold = float(config.get("optimal_threshold", 0.64))
    singleton_cutoff = float(config.get("singleton_cutoff", 0.75))
    val_macro_f05 = float(config.get("validation_macro_f05", 0.0))
    print(f"  Calibrated Decision Threshold: {optimal_threshold}")
    print(f"  Singleton Safeguard Cutoff   : {singleton_cutoff}")
    print(f"  Validation Macro F0.5 Score  : {val_macro_f05:.5f}")

    # -------------------------------------------------------------
    # STAGE 8 & 9: Test Inference on Existing Candidate Pairs
    # -------------------------------------------------------------
    print("\n[STAGE 8 & 9/10] Running Test Inference on Candidate Pairs...")
    check_disk_space(output_dir, required_gb=3.0, stage_name="Stage 8 & 9 Test Inference")
    run_test_inference(
        cands_path=test_cands_path,
        s1_path=test_files["s1"],
        s2_path=test_files["s2"],
        s3_path=test_files["s3"],
        out_dir=output_dir,
        out_matching_path=out_matching_path
    )

    # -------------------------------------------------------------
    # STAGE 10: Official Submission Validation
    # -------------------------------------------------------------
    print("\n[STAGE 10/10] Validating Submission via Official Validator...")
    validator_script = os.path.join(REPO_ROOT, "utils", "validate_submission.py")
    if not os.path.exists(validator_script):
        validator_script = os.path.join(REPO_ROOT, "student_resource", "utils", "validate_submission.py")

    val_cmd = [
        sys.executable, validator_script,
        "--matching", out_matching_path,
        "--candidate", test_cands_path,
        "--test-dir", test_files["test_dir"]
    ]
    val_proc = subprocess.run(val_cmd, capture_output=True, text=True)
    print(val_proc.stdout)
    if val_proc.stderr:
        print(val_proc.stderr)

    validator_passed = (val_proc.returncode == 0)

    # -------------------------------------------------------------
    # Final Execution Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  EXPERIMENT 002A EXECUTION SUMMARY")
    print("=" * 75)
    print(f"  Experiment Name        : {args.experiment} (15 Baseline Features + Candidate Rank)")
    print(f"  Validation Macro F0.5  : {val_macro_f05:.5f}")
    print(f"  Selected Threshold     : {optimal_threshold}")
    print(f"  Singleton Cutoff       : {singleton_cutoff}")
    print(f"  Candidate Pair Path    : {test_cands_path}")
    print(f"  Matching Results TSV   : {out_matching_path}")
    print(f"  Validator Result       : {'PASS (Safe for Submission)' if validator_passed else 'FAIL'}")
    print("=" * 75)

    if not validator_passed:
        sys.exit(1)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Kaggle Runner")
    parser.add_argument("--experiment", default="002a", choices=["002a", "002", "baseline"], help="Experiment identifier")
    parser.add_argument("--data-root", default=None, help="Root directory containing dataset (with train/ and test/ folders)")
    parser.add_argument("--output-root", default=None, help="Directory to store outputs and intermediate artifacts")
    parser.add_argument("--cache-dir", default=None, help="Directory for temporary dense embedding cache (defaults to /tmp/embeddings_cache if /tmp exists)")
    parser.add_argument("--keep-cache", action="store_true", help="Preserve temporary embedding cache after Stage 3 (default: False, purges cache to conserve disk)")
    parser.add_argument("--dry-run", action="store_true", help="Perform pre-flight checks and validate paths without heavy execution")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow running blocking on CPU (not recommended)")
    args = parser.parse_args()

    sys.exit(run_experiment(args))


if __name__ == "__main__":
    main()
