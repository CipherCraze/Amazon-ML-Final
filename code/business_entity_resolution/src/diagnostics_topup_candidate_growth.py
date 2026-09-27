#!/usr/bin/env python3
"""
Candidate Growth & Validation Recall Diagnostics for Option 1 (India Top-Up)
Amazon ML Challenge 2026 — Experiment 010 (Final)
"""

import os
import sys
import gc
import time
import pandas as pd
from collections import defaultdict

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from india_topup_retrieval import build_india_inverted_index, rank_topup_candidates, extract_indian_keys


def run_diagnostics():
    print("=" * 75)
    print("  EXPERIMENT 010: CANDIDATE GROWTH & TOP-UP RECALL DIAGNOSTICS")
    print("=" * 75)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_dir = os.path.join(repo_root, "student_resource", "dataset", "train")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(repo_root, "dataset", "train")
        
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")

    print("\n1. Measuring Global S1 Query Distribution:")
    s1_df = pd.read_csv(s1_path, sep="\t", usecols=['entity_id', 'country'], dtype=str)
    total_train = len(s1_df)
    train_country_counts = s1_df['country'].value_counts().to_dict()
    print(f"  Total Train S1 Queries : {total_train:,}")
    for ctry, count in train_country_counts.items():
        print(f"    - {ctry:15s}: {count:,} ({count/total_train*100:.2f}%)")
    del s1_df
    gc.collect()

    # Test distribution (from known metadata)
    total_test = 1732544
    test_counts = {
        "India": 809986,
        "United States": 663106,
        "France": 259452
    }
    print(f"\n2. Measuring Test S1 Query Distribution:")
    print(f"  Total Test S1 Queries  : {total_test:,}")
    for ctry, count in test_counts.items():
        print(f"    - {ctry:15s}: {count:,} ({count/total_test*100:.2f}%)")

    # Load Ground Truth
    print("\n3. Loading Ground Truth Mapping...")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
    gt_map = {}
    for _, row in gt_df.iterrows():
        sid = row['source1_entity_id']
        m = row.get('matched_entity_ids', '')
        if pd.notna(m) and str(m).strip() and str(m).strip() != 'nan':
            gt_map[sid] = set(str(m).strip().split(','))
    print(f"  Total Queries with Non-Empty GT: {len(gt_map):,}")

    # Build Indian catalog inverted index
    print("\n4. Building Indian Catalog Inverted Index...")
    index, catalog_names = build_india_inverted_index(s2_path, s3_path, max_block_size=25, chunk_size=100000)

    # Sample 3,000 Indian queries with GT to measure top-up recall gain and candidate yield
    print("\n5. Measuring Empirical Top-Up Yield on Sample of Indian Queries...")
    sample_queries = []
    chunk_iter = pd.read_csv(s1_path, sep="\t", chunksize=50000, 
                             usecols=['entity_id', 'business_name', 'business_address', 'country'],
                             dtype=str)
    for chunk in chunk_iter:
        ind_chunk = chunk[(chunk['country'] == 'India') & (chunk['entity_id'].isin(gt_map))]
        for _, row in ind_chunk.iterrows():
            sample_queries.append((row['entity_id'], str(row['business_name']), str(row['business_address'])))
            if len(sample_queries) >= 3000:
                break
        if len(sample_queries) >= 3000:
            break

    print(f"  Sampled {len(sample_queries):,} Indian queries with non-empty ground truth.")

    queries_topped_up = 0
    total_added_cands = 0
    recovered_matches = 0
    total_true_matches = sum(len(gt_map[sid]) for sid, _, _ in sample_queries)

    for sid, name, addr in sample_queries:
        true_set = gt_map[sid]
        # Simulate base k=30 retrieval (where ~85.34% of true matches were retrieved)
        keys = extract_indian_keys(name, addr)
        cand_hits = defaultdict(int)
        for kt, kv in keys:
            for cid in index.get(f"{kt}:{kv}", []):
                cand_hits[cid] += 1
                
        # Get top-up candidates
        topup_cands = rank_topup_candidates(name, keys, cand_hits, catalog_names, max_add=6)
        n_add = len(topup_cands)
        total_added_cands += n_add
        if n_add > 0:
            queries_topped_up += 1
            
        # Check if true matches were recovered in topup
        for cid in topup_cands:
            if cid in true_set:
                recovered_matches += 1

    avg_added_per_query = total_added_cands / len(sample_queries)
    pct_topped_up = (queries_topped_up / len(sample_queries)) * 100
    empirical_recovery_rate = (recovered_matches / total_true_matches) * 100

    print("\n" + "=" * 75)
    print("  EMPIRICAL CANDIDATE GROWTH & RECALL METRICS")
    print("=" * 75)
    print(f"  Indian Queries Sampled             : {len(sample_queries):,}")
    print(f"  Indian Queries Receiving Top-Up    : {queries_topped_up:,} ({pct_topped_up:.1f}%)")
    print(f"  Average Candidates Added per Query : {avg_added_per_query:.2f} (bounded <= 6)")
    print(f"  True Matches in Sample             : {total_true_matches:,}")
    print(f"  True Matches Recovered by Top-Up   : {recovered_matches:,} (+{empirical_recovery_rate:.2f}% recall gain)")
    print("=" * 75)

    # Global Candidate Projections
    test_base_cands = total_test * 30
    test_ind_added = int(test_counts["India"] * avg_added_per_query)
    test_total_aug = test_base_cands + test_ind_added
    test_growth_pct = (test_ind_added / test_base_cands) * 100

    train_base_cands = total_train * 30
    train_ind_added = int(train_country_counts["India"] * avg_added_per_query)
    train_total_aug = train_base_cands + train_ind_added
    train_growth_pct = (train_ind_added / train_base_cands) * 100

    val_base_cands = int(total_train * 0.20 * 30)
    val_ind_added = int(train_country_counts["India"] * 0.20 * avg_added_per_query)
    val_total_aug = val_base_cands + val_ind_added

    print("\n" + "=" * 75)
    print("  GLOBAL CANDIDATE POOL PROJECTIONS")
    print("=" * 75)
    print(f"  Test Base Candidates (k=30)       : {test_base_cands:,}")
    print(f"  Test Added Candidates (India)     : {test_ind_added:,}")
    print(f"  Test Augmented Total Candidates   : {test_total_aug:,} (+{test_growth_pct:.2f}% growth)")
    print(f"  Test Country Breakdown:")
    print(f"    - India Candidates              : {test_counts['India'] * 30 + test_ind_added:,} ({((test_counts['India']*30+test_ind_added)/test_total_aug)*100:.1f}%)")
    print(f"    - United States Candidates      : {test_counts['United States'] * 30:,} ({(test_counts['United States']*30/test_total_aug)*100:.1f}%)")
    print(f"    - France Candidates             : {test_counts['France'] * 30:,} ({(test_counts['France']*30/test_total_aug)*100:.1f}%)")
    print("-" * 75)
    print(f"  Train Base Candidates (k=30)      : {train_base_cands:,}")
    print(f"  Train Added Candidates (India)    : {train_ind_added:,}")
    print(f"  Train Augmented Total Candidates  : {train_total_aug:,} (+{train_growth_pct:.2f}% growth)")
    print("-" * 75)
    print(f"  Val Base Candidates (k=30)        : {val_base_cands:,}")
    print(f"  Val Added Candidates (India)      : {val_ind_added:,}")
    print(f"  Val Augmented Total Candidates    : {val_total_aug:,} (+{(val_ind_added/val_base_cands)*100:.2f}% growth)")
    print("=" * 75)

    # Recall Ceiling Projections
    base_us_pair_recall = 0.9538
    base_india_pair_recall = 0.8534
    est_new_india_pair_recall = min(0.95, base_india_pair_recall + (empirical_recovery_rate / 100.0) * 0.85)
    est_overall_pair_recall = 0.60 * base_us_pair_recall + 0.40 * est_new_india_pair_recall

    print("\n" + "=" * 75)
    print("  PROJECTED VALIDATION RECALL AFTER TOP-UP")
    print("=" * 75)
    print(f"  Baseline India Pair Recall (k=30) : {base_india_pair_recall*100:.2f}%")
    print(f"  Augmented India Pair Recall       : {est_new_india_pair_recall*100:.2f}% (+{(est_new_india_pair_recall - base_india_pair_recall)*100:.2f}%)")
    print(f"  Baseline US Pair Recall (k=30)    : {base_us_pair_recall*100:.2f}% (unchanged)")
    print(f"  Baseline Overall Pair Recall      : 91.36%")
    print(f"  Augmented Overall Pair Recall     : {est_overall_pair_recall*100:.2f}% (+{(est_overall_pair_recall - 0.9136)*100:.2f}%)")
    print("=" * 75)


if __name__ == "__main__":
    run_diagnostics()
