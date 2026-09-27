import time
import os
import numpy as np
import pandas as pd
import pyarrow.csv as pv

def run_step2_candidate_recall():
    print("=" * 70)
    print("STEP 2: VALIDATION CANDIDATE RECALL CEILING DIAGNOSTIC")
    print("=" * 70)

    t0 = time.time()
    
    # 1. Load ground truth validation split
    val_gt_path = 'output/val_gt_split.tsv'
    print(f"Loading {val_gt_path}...")
    val_gt_df = pd.read_csv(val_gt_path, sep='\t', dtype=str)
    total_val_entities = len(val_gt_df)
    
    # Map S1 to its ground truth matches
    gt_pairs = set() # set of (s1_id, matched_id)
    s1_gt_map = {} # s1_id -> set of matched_ids
    singletons = set()
    
    for _, row in val_gt_df.iterrows():
        sid = row['source1_entity_id']
        matches_val = row.get('matched_entity_ids', '')
        if pd.notna(matches_val) and str(matches_val).strip() and str(matches_val).strip() != 'nan':
            m_list = [m.strip() for m in str(matches_val).split(',') if m.strip()]
            s1_gt_map[sid] = set(m_list)
            for mid in m_list:
                gt_pairs.add((sid, mid))
        else:
            s1_gt_map[sid] = set()
            singletons.add(sid)
            
    total_gt_pairs = len(gt_pairs)
    s1_with_gt = set(s1_gt_map.keys()) - singletons
    num_s1_with_gt = len(s1_with_gt)
    
    print(f"Total validation S1 entities: {total_val_entities:,}")
    print(f"Validation singletons (no match in catalog): {len(singletons):,} ({len(singletons)/total_val_entities*100:.2f}%)")
    print(f"Validation S1 entities with ground truth: {num_s1_with_gt:,} ({num_s1_with_gt/total_val_entities*100:.2f}%)")
    print(f"Total validation ground-truth pairs: {total_gt_pairs:,} (avg {total_gt_pairs/num_s1_with_gt:.2f} pairs/entity)")

    # 2. Load country mapping for validation S1 entities
    print("\nLoading country mapping from train_source1.tsv...")
    s1_train_path = 'student_resource/dataset/train/train_source1.tsv'
    s1_country_df = pd.read_csv(s1_train_path, sep='\t', usecols=['entity_id', 'country'], dtype=str)
    s1_country_map = dict(zip(s1_country_df['entity_id'], s1_country_df['country']))
    del s1_country_df
    
    # 3. Load validation candidates from full_val_features_v3.csv
    val_csv_path = 'output/full_val_features_v3.csv'
    print(f"\nLoading candidates from {val_csv_path} via PyArrow...")
    t_read = time.time()
    tab = pv.read_csv(val_csv_path, convert_options=pv.ConvertOptions(
        include_columns=['source1_entity_id', 'candidate_entity_id', 'label', 'candidate_rank']
    ))
    df_cands = tab.to_pandas()
    del tab
    print(f"Loaded {len(df_cands):,} candidate pairs in {time.time() - t_read:.2f}s")
    
    # Filter candidates with label == 1 or check intersection with gt_pairs
    # Check label == 1 vs ground truth matching
    val_positives = df_cands[df_cands['label'] == 1]
    print(f"Total positive candidates in full_val_features_v3.csv (label==1): {len(val_positives):,}")
    
    # Candidate pairs present
    # Group by source1_entity_id
    cand_pairs_set = set(zip(val_positives['source1_entity_id'], val_positives['candidate_entity_id']))
    
    # Overlap with gt_pairs
    retrieved_gt_pairs = gt_pairs.intersection(cand_pairs_set)
    pair_recall = len(retrieved_gt_pairs) / total_gt_pairs
    pair_misses = total_gt_pairs - len(retrieved_gt_pairs)
    
    # Entity-level recall
    retrieved_s1_set = set(val_positives['source1_entity_id'])
    retrieved_s1_hits = s1_with_gt.intersection(retrieved_s1_set)
    entity_recall = len(retrieved_s1_hits) / num_s1_with_gt
    entity_misses = num_s1_with_gt - len(retrieved_s1_hits)
    
    print("\n" + "=" * 50)
    print("GLOBAL CANDIDATE RECALL CEILING RESULTS")
    print("=" * 50)
    print(f"Total validation S1 entities:            {total_val_entities:,}")
    print(f"Total validation ground-truth pairs:      {total_gt_pairs:,}")
    print(f"Validation S1 entities with GT:           {num_s1_with_gt:,}")
    print(f"Entity-level candidate recall:           {entity_recall*100:.4f}% ({len(retrieved_s1_hits):,} / {num_s1_with_gt:,})")
    print(f"Entity-level misses:                     {entity_misses:,} ({entity_misses/num_s1_with_gt*100:.4f}%)")
    print(f"Pair-level candidate recall:             {pair_recall*100:.4f}% ({len(retrieved_gt_pairs):,} / {total_gt_pairs:,})")
    print(f"Pair-level misses (Retrieval-limited FN): {pair_misses:,} ({pair_misses/total_gt_pairs*100:.4f}%)")
    
    # Candidate rank distribution for retrieved true pairs
    print("\n" + "=" * 50)
    print("CANDIDATE RANK DISTRIBUTION OF TRUE PAIRS (label==1)")
    print("=" * 50)
    ranks = val_positives['candidate_rank'].values
    rank_bins = [
        ("Rank 1", ranks == 1),
        ("Rank 2", ranks == 2),
        ("Rank 3-5", (ranks >= 3) & (ranks <= 5)),
        ("Rank 6-10", (ranks >= 6) & (ranks <= 10)),
        ("Rank 11-20", (ranks >= 11) & (ranks <= 20)),
        ("Rank 21-30", (ranks >= 21) & (ranks <= 30)),
        ("Rank 31+", ranks > 30),
    ]
    for label, mask in rank_bins:
        cnt = np.sum(mask)
        pct = cnt / len(ranks) * 100
        cum_cnt = np.sum(ranks <= (1 if label == "Rank 1" else (2 if label == "Rank 2" else (5 if label == "Rank 3-5" else (10 if label == "Rank 6-10" else (20 if label == "Rank 11-20" else 30))))))
        print(f"  {label:<12}: {cnt:>8,} ({pct:>6.2f}%) | Cumulative Top: {cum_cnt:>8,} ({cum_cnt/len(ranks)*100:>6.2f}%)")
    
    # Country breakdown
    print("\n" + "=" * 50)
    print("COUNTRY BREAKDOWN OF VALIDATION CANDIDATE RECALL")
    print("=" * 50)
    
    # Map country to entities and pairs
    country_stats = {}
    for sid in s1_with_gt:
        country = s1_country_map.get(sid, 'UNKNOWN')
        if country not in country_stats:
            country_stats[country] = {
                's1_count': 0,
                's1_hits': 0,
                'gt_pairs': 0,
                'retrieved_pairs': 0,
                'ranks': []
            }
        country_stats[country]['s1_count'] += 1
        if sid in retrieved_s1_hits:
            country_stats[country]['s1_hits'] += 1
            
        gt_m = s1_gt_map[sid]
        country_stats[country]['gt_pairs'] += len(gt_m)
        
    # Add retrieved pairs and ranks
    val_pos_sid = val_positives['source1_entity_id'].values
    val_pos_cid = val_positives['candidate_entity_id'].values
    val_pos_rank = val_positives['candidate_rank'].values
    
    for sid, cid, rk in zip(val_pos_sid, val_pos_cid, val_pos_rank):
        if (sid, cid) in gt_pairs:
            country = s1_country_map.get(sid, 'UNKNOWN')
            if country in country_stats:
                country_stats[country]['retrieved_pairs'] += 1
                country_stats[country]['ranks'].append(rk)
                
    header_str = f"{'Country':<10} | {'S1 w/ GT':>10} | {'Ent Recall':>11} | {'GT Pairs':>10} | {'Pair Recall':>11} | {'Pair Miss':>10} | {'Top-1 %':>8} | {'Top-5 %':>8}"
    print(header_str)
    print("-" * len(header_str))
    
    for c, stats in sorted(country_stats.items(), key=lambda x: x[1]['s1_count'], reverse=True):
        ent_rec = stats['s1_hits'] / stats['s1_count'] if stats['s1_count'] > 0 else 0
        p_rec = stats['retrieved_pairs'] / stats['gt_pairs'] if stats['gt_pairs'] > 0 else 0
        p_miss = stats['gt_pairs'] - stats['retrieved_pairs']
        rnks = np.array(stats['ranks'])
        top1_pct = (np.sum(rnks == 1) / len(rnks) * 100) if len(rnks) > 0 else 0
        top5_pct = (np.sum(rnks <= 5) / len(rnks) * 100) if len(rnks) > 0 else 0
        print(f"{c:<10} | {stats['s1_count']:>10,} | {ent_rec*100:>10.2f}% | {stats['gt_pairs']:>10,} | {p_rec*100:>10.2f}% | {p_miss:>10,} | {top1_pct:>7.2f}% | {top5_pct:>7.2f}%")
        
    print(f"\nCompleted in {time.time() - t0:.2f}s")

if __name__ == '__main__':
    run_step2_candidate_recall()
