#!/usr/bin/env python3
"""
Fast Offline Threshold & Policy Evaluator
Amazon ML Challenge 2026 — Experiment 010 (Final)

Allows instantly applying global thresholds or country-specific thresholds
to pre-computed reusable test candidate probabilities without re-running
heavy feature extraction or model inference.
"""

import os
import sys
import json
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def apply_policy(raw_probs_path, s1_path, config_path, output_matching_path, override_threshold=None):
    print("=" * 75)
    print("  FAST THRESHOLD & POLICY EVALUATOR")
    print(f"  Probabilities Input  : {raw_probs_path}")
    print(f"  Configuration File   : {config_path}")
    print(f"  Output Matching Path : {output_matching_path}")
    print("=" * 75)

    if not os.path.exists(raw_probs_path):
        raise FileNotFoundError(f"Missing raw test probabilities file: {raw_probs_path}")

    # Load configuration
    country_thresholds = None
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            threshold = float(cfg.get('optimal_threshold', 0.63))
            country_thresholds = cfg.get('country_thresholds', None)
            singleton_cutoff = float(cfg.get('singleton_cutoff', 0.0))
    else:
        threshold = 0.63
        singleton_cutoff = 0.0

    if override_threshold is not None:
        threshold = override_threshold
        country_thresholds = None
        print(f"  [Override] Using user-specified threshold: {threshold:.2f} (ignoring config)")

    print(f"  Decision Threshold  : {threshold:.2f}")
    if country_thresholds:
        print(f"  Country Thresholds  : {country_thresholds}")

    # Load S1 country map
    print("Loading S1 query list and country metadata...")
    s1_df = pd.read_csv(s1_path, sep="\t", usecols=['entity_id', 'country'], dtype=str)
    all_s1_ids = s1_df['entity_id'].tolist()
    s1_country_map = dict(zip(s1_df['entity_id'], s1_df['country']))
    del s1_df

    # Stream raw probabilities
    print(f"Streaming probabilities from {raw_probs_path}...")
    matches_by_s1 = defaultdict(list)
    chunk_size = 1000000

    for chunk in pd.read_csv(raw_probs_path, sep="\t", chunksize=chunk_size, 
                             dtype={'source1_entity_id': str, 'candidate_entity_id': str, 'probability': float}):
        for s1_id, cand_id, p in zip(chunk['source1_entity_id'], chunk['candidate_entity_id'], chunk['probability']):
            ctry = s1_country_map.get(s1_id, '')
            t_req = country_thresholds.get(ctry, threshold) if country_thresholds else threshold
            if p >= t_req:
                matches_by_s1[s1_id].append(cand_id)

    # Write formatted output preserving exact S1 order
    print(f"Writing final matching results to {output_matching_path}...")
    total_matches = 0
    total_singletons = 0

    with open(output_matching_path, 'w', encoding='utf-8', buffering=4*1024*1024) as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ids:
            cands = matches_by_s1.get(sid, [])
            if cands:
                f_out.write(f"{sid}\t{','.join(cands)}\n")
                total_matches += len(cands)
            else:
                f_out.write(f"{sid}\t\n")
                total_singletons += 1

    print("\n" + "=" * 75)
    print("  POLICY APPLICATION COMPLETE")
    print(f"  Total Queries Written: {len(all_s1_ids):,}")
    print(f"  Matches Predicted    : {total_matches:,}")
    print(f"  Singletons Predicted : {total_singletons:,} ({total_singletons/len(all_s1_ids)*100:.2f}%)")
    print(f"  Saved To             : {output_matching_path}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Offline Policy Applier")
    parser.add_argument("--probs-file", default=None, help="Path to test_candidate_probabilities.tsv")
    parser.add_argument("--s1", default=None, help="Path to test_source1.tsv")
    parser.add_argument("--config", default=None, help="Path to threshold_config_v3.json")
    parser.add_argument("--threshold", type=float, default=None, help="Override global threshold")
    parser.add_argument("--out-matching", default=None, help="Output path for matching_results.tsv")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    out_dir = os.path.join(repo_root, "output")
    test_dir = os.path.join(repo_root, "student_resource", "dataset", "test")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(repo_root, "dataset", "test")

    probs_file = args.probs_file or os.path.join(out_dir, "test_candidate_probabilities.tsv")
    s1_path = args.s1 or os.path.join(test_dir, "test_source1.tsv")
    config_file = args.config or os.path.join(out_dir, "threshold_config_v3.json")
    out_matching = args.out_matching or os.path.join(out_dir, "augmented_matching_results.tsv")

    apply_policy(probs_file, s1_path, config_file, out_matching, override_threshold=args.threshold)
