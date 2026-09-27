import time
import os
import gc
import json
import numpy as np
import pandas as pd
import pyarrow.csv as pv

def compute_f05_single(pred_set, true_set):
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    if len(pred_set) == 0:
        return 0.0
    tp = len(pred_set.intersection(true_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def run_step4_and_5_diagnostics():
    print("=" * 70)
    print("STEP 4: VALIDATION ERROR ANALYSIS & STEP 5: OFFLINE POLICY OPTIMIZATION")
    print("=" * 70)
    
    t0 = time.time()
    
    # 1. Load ground truth
    val_gt_path = 'output/val_gt_split.tsv'
    print(f"Loading {val_gt_path}...")
    val_gt_df = pd.read_csv(val_gt_path, sep='\t', dtype=str)
    total_val_entities = len(val_gt_df)
    
    gt_map = {}
    singleton_entities = set()
    total_gt_pairs = 0
    gt_pairs_set = set()
    
    for _, row in val_gt_df.iterrows():
        sid = row['source1_entity_id']
        matches_val = row.get('matched_entity_ids', '')
        if pd.notna(matches_val) and str(matches_val).strip() and str(matches_val).strip() != 'nan':
            m_set = set(m.strip() for m in str(matches_val).split(',') if m.strip())
            gt_map[sid] = m_set
            total_gt_pairs += len(m_set)
            for cid in m_set:
                gt_pairs_set.add((sid, cid))
        else:
            gt_map[sid] = set()
            singleton_entities.add(sid)
            
    print(f"Total entities: {total_val_entities:,}")
    print(f"Total ground truth pairs: {total_gt_pairs:,}")
    print(f"Total singletons: {len(singleton_entities):,}")
    
    # 2. Load probabilities
    print("\nLoading saved validation probabilities...")
    val_probs_xgb = np.load('output/val_probs_xgb_v3.npy')
    val_probs_lgb = np.load('output/val_probs_lgb_v3.npy')
    print(f"Loaded XGB probs: {val_probs_xgb.shape}, LGB probs: {val_probs_lgb.shape}")
    
    # 3. Load validation features needed for error segmentation
    # Columns: ['source1_entity_id', 'candidate_entity_id', 'label', 'candidate_rank', 'name_ratio', 'is_addr_missing', 'street_name_sim', 'country_match']
    val_csv_path = 'output/full_val_features_v3.csv'
    print(f"\nLoading feature columns from {val_csv_path} via PyArrow...")
    t_read = time.time()
    cols = ['source1_entity_id', 'candidate_entity_id', 'label', 'candidate_rank', 'name_ratio', 'is_addr_missing', 'street_name_sim', 'country_match']
    tab = pv.read_csv(val_csv_path, convert_options=pv.ConvertOptions(include_columns=cols))
    df_val = tab.to_pandas()
    del tab
    print(f"Loaded {len(df_val):,} rows in {time.time() - t_read:.2f}s")
    
    # Add probabilities
    df_val['p_xgb'] = val_probs_xgb
    df_val['p_lgb'] = val_probs_lgb
    
    # Current production policy: XGBoost, threshold = 0.63, singleton cutoff = 0.75
    # Wait, let's also evaluate with and without the singleton cutoff!
    
    # 4. H1 vs H2 FALSE NEGATIVES CLASSIFICATION
    print("\n" + "=" * 50)
    print("H. FALSE NEGATIVE TAXONOMY (H1: Retrieval vs H2: Model/Rank)")
    print("=" * 50)
    
    # Filter true pairs in candidates (label == 1)
    val_positives = df_val[df_val['label'] == 1].copy()
    retrieved_true_pairs_count = len(val_positives)
    
    # H1: Retrieval-limited FNs (never retrieved in candidate pool)
    h1_count = total_gt_pairs - retrieved_true_pairs_count
    h1_pct = h1_count / total_gt_pairs * 100
    
    # Model predictions under baseline policy:
    # First, let's see how many retrieved true pairs were rejected by p < 0.63 or singleton guard
    # Group by source1_entity_id to determine max probability per S1
    print("Grouping candidate probabilities by S1 entity...")
    t_grp = time.time()
    
    # Group by S1 for max_p and candidate lists
    s1_grouped = {} # s1_id -> list of (cid, p_xgb, p_lgb, rk, label, name_ratio, is_addr_missing, street_name_sim, country_match)
    
    # We can iterate through df_val
    s1_vals = df_val['source1_entity_id'].values
    cid_vals = df_val['candidate_entity_id'].values
    p_xgb_vals = df_val['p_xgb'].values
    p_lgb_vals = df_val['p_lgb'].values
    rk_vals = df_val['candidate_rank'].values
    lbl_vals = df_val['label'].values
    nr_vals = df_val['name_ratio'].values
    am_vals = df_val['is_addr_missing'].values
    ss_vals = df_val['street_name_sim'].values
    cm_vals = df_val['country_match'].values
    
    for i in range(len(df_val)):
        sid = s1_vals[i]
        if sid not in s1_grouped:
            s1_grouped[sid] = []
        s1_grouped[sid].append((cid_vals[i], p_xgb_vals[i], p_lgb_vals[i], rk_vals[i], lbl_vals[i], nr_vals[i], am_vals[i], ss_vals[i], cm_vals[i]))
        
    print(f"Grouped into {len(s1_grouped):,} unique S1 queries in {time.time() - t_grp:.2f}s")
    
    # Baseline 002A policy: XGB, thr = 0.63, singleton cutoff = 0.75
    # Let's count True Positives (TP), False Positives (FP), and False Negatives (FN) under baseline policy
    accepted_pairs = set()
    accepted_pairs_no_guard = set() # thr = 0.63 only
    
    s1_max_p = {}
    for sid, cands in s1_grouped.items():
        max_p = max(c[1] for c in cands)
        s1_max_p[sid] = max_p
        
        # With singleton guard:
        if max_p >= 0.75:
            for c in cands:
                if c[1] >= 0.63:
                    accepted_pairs.add((sid, c[0]))
                    
        # Without singleton guard:
        for c in cands:
            if c[1] >= 0.63:
                accepted_pairs_no_guard.add((sid, c[0]))
                
    # Evaluate False Negatives under baseline 002A policy (with 0.75 singleton guard)
    tp_pairs = gt_pairs_set.intersection(accepted_pairs)
    total_fn_pairs = gt_pairs_set - accepted_pairs
    h2_pairs = set(zip(val_positives['source1_entity_id'], val_positives['candidate_entity_id'])) - accepted_pairs
    
    print(f"Total Ground Truth Pairs:                    {total_gt_pairs:,} (100.0%)")
    print(f"True Positives (Predicted Correctly):        {len(tp_pairs):,} ({len(tp_pairs)/total_gt_pairs*100:.2f}%)")
    print(f"Total False Negatives (Missed Pairs):        {len(total_fn_pairs):,} ({len(total_fn_pairs)/total_gt_pairs*100:.2f}%)")
    print(f"  --> H1. Retrieval-Limited (Never in pool):  {h1_count:,} ({h1_pct:.2f}% of all GT pairs | {h1_count/len(total_fn_pairs)*100:.2f}% of FNs)")
    print(f"  --> H2. Model/Rank-Limited (In pool, missed): {len(h2_pairs):,} ({len(h2_pairs)/total_gt_pairs*100:.2f}% of all GT pairs | {len(h2_pairs)/len(total_fn_pairs)*100:.2f}% of FNs)")
    
    # Also evaluate without the 0.75 singleton guard:
    tp_pairs_ng = gt_pairs_set.intersection(accepted_pairs_no_guard)
    total_fn_pairs_ng = gt_pairs_set - accepted_pairs_no_guard
    h2_pairs_ng = set(zip(val_positives['source1_entity_id'], val_positives['candidate_entity_id'])) - accepted_pairs_no_guard
    print(f"\n[Comparison without 0.75 singleton guard (pure thr=0.63)]:")
    print(f"  TP: {len(tp_pairs_ng):,}, Total FN: {len(total_fn_pairs_ng):,}")
    print(f"  H1: {h1_count:,} ({h1_count/len(total_fn_pairs_ng)*100:.2f}% of FNs)")
    print(f"  H2: {len(h2_pairs_ng):,} ({len(h2_pairs_ng)/len(total_fn_pairs_ng)*100:.2f}% of FNs)")
    killed_by_guard = len(h2_pairs) - len(h2_pairs_ng)
    print(f"  --> Number of true pairs killed SPECIFICALLY by 0.75 singleton guard: {killed_by_guard:,}!")
    
    # 5. ERROR SEGMENTATION OF H2 (MODEL-LIMITED FALSE NEGATIVES)
    print("\n" + "=" * 50)
    print("DETAILED SEGMENTATION OF H2 FALSE NEGATIVES")
    print("=" * 50)
    
    # Extract the rows of H2
    # In val_positives: label == 1, but (sid, cid) not in accepted_pairs
    val_positives['is_h2_fn'] = [(s, c) in h2_pairs for s, c in zip(val_positives['source1_entity_id'], val_positives['candidate_entity_id'])]
    h2_df = val_positives[val_positives['is_h2_fn']].copy()
    print(f"Total H2 candidate rows: {len(h2_df):,}")
    
    # A. Candidate Rank Segmentation
    print("\n--- A. Candidate Rank Distribution of H2 Misses ---")
    rk_bins = [
        ("Rank 1", h2_df['candidate_rank'] == 1),
        ("Rank 2", h2_df['candidate_rank'] == 2),
        ("Rank 3-5", (h2_df['candidate_rank'] >= 3) & (h2_df['candidate_rank'] <= 5)),
        ("Rank 6-10", (h2_df['candidate_rank'] >= 6) & (h2_df['candidate_rank'] <= 10)),
        ("Rank 11-30", (h2_df['candidate_rank'] >= 11) & (h2_df['candidate_rank'] <= 30)),
    ]
    for lbl, m in rk_bins:
        cnt = m.sum()
        print(f"  {lbl:<12}: {cnt:>8,} ({cnt/len(h2_df)*100:>6.2f}%)")
        
    # B. Name Similarity Bins
    print("\n--- B. Name Similarity Bins (name_ratio) of H2 Misses ---")
    nr_bins = [
        ("< 0.50", h2_df['name_ratio'] < 0.50),
        ("0.50 - 0.70", (h2_df['name_ratio'] >= 0.50) & (h2_df['name_ratio'] < 0.70)),
        ("0.70 - 0.80", (h2_df['name_ratio'] >= 0.70) & (h2_df['name_ratio'] < 0.80)),
        ("0.80 - 0.90", (h2_df['name_ratio'] >= 0.80) & (h2_df['name_ratio'] < 0.90)),
        ("0.90 - 0.95", (h2_df['name_ratio'] >= 0.90) & (h2_df['name_ratio'] < 0.95)),
        ("0.95+", h2_df['name_ratio'] >= 0.95),
    ]
    for lbl, m in nr_bins:
        cnt = m.sum()
        print(f"  {lbl:<14}: {cnt:>8,} ({cnt/len(h2_df)*100:>6.2f}%)")
        
    # C. Address Availability
    print("\n--- C. Address Availability of H2 Misses ---")
    # is_addr_missing == 1.0 means either S1 or candidate address is missing
    addr_miss_cnt = (h2_df['is_addr_missing'] == 1.0).sum()
    addr_pres_cnt = (h2_df['is_addr_missing'] == 0.0).sum()
    print(f"  Address Missing (either S1 or cand): {addr_miss_cnt:>8,} ({addr_miss_cnt/len(h2_df)*100:>6.2f}%)")
    print(f"  Address Present (both available):     {addr_pres_cnt:>8,} ({addr_pres_cnt/len(h2_df)*100:>6.2f}%)")
    
    # E. Country Agreement
    print("\n--- E. Country Agreement of H2 Misses ---")
    cm_same = (h2_df['country_match'] == 1.0).sum()
    cm_diff = (h2_df['country_match'] == 0.0).sum()
    print(f"  Same Country:      {cm_same:>8,} ({cm_same/len(h2_df)*100:>6.2f}%)")
    print(f"  Different/Missing: {cm_diff:>8,} ({cm_diff/len(h2_df)*100:>6.2f}%)")
    
    # F. Model Probability Bins
    print("\n--- F. XGBoost Model Probability Bins of H2 Misses ---")
    p_bins = [
        ("< 0.40", h2_df['p_xgb'] < 0.40),
        ("0.40 - 0.50", (h2_df['p_xgb'] >= 0.40) & (h2_df['p_xgb'] < 0.50)),
        ("0.50 - 0.60", (h2_df['p_xgb'] >= 0.50) & (h2_df['p_xgb'] < 0.60)),
        ("0.60 - 0.63", (h2_df['p_xgb'] >= 0.60) & (h2_df['p_xgb'] < 0.63)),
        ("0.63 - 0.75 (Killed by singleton guard!)", (h2_df['p_xgb'] >= 0.63) & (h2_df['p_xgb'] < 0.75)),
        (">= 0.75 (Rejected by other rule/outranked)", h2_df['p_xgb'] >= 0.75),
    ]
    for lbl, m in p_bins:
        cnt = m.sum()
        print(f"  {lbl:<45}: {cnt:>8,} ({cnt/len(h2_df)*100:>6.2f}%)")
        
    # G. Country Breakdown of H2 Misses
    print("\n--- Country Breakdown of H2 Misses ---")
    s1_train_path = 'student_resource/dataset/train/train_source1.tsv'
    s1_country_df = pd.read_csv(s1_train_path, sep='\t', usecols=['entity_id', 'country'], dtype=str)
    s1_country_map = dict(zip(s1_country_df['entity_id'], s1_country_df['country']))
    h2_df['country'] = [s1_country_map.get(s, 'UNKNOWN') for s in h2_df['source1_entity_id']]
    for c, cnt in h2_df['country'].value_counts().items():
        print(f"  {c:<10}: {cnt:>8,} ({cnt/len(h2_df)*100:>6.2f}%)")
        
    del df_val, val_positives, h2_df
    gc.collect()
    
    # 6. STEP 5: OFFLINE POLICY OPTIMIZATION
    print("\n" + "=" * 50)
    print("STEP 5: OFFLINE POLICY OPTIMIZATION")
    print("=" * 50)
    
    def evaluate_policy(prob_key, threshold, singleton_cutoff=None, margin_thr=None, rank_cutoff=None):
        score_sum = 0.0
        for sid in val_gt_df['source1_entity_id']:
            cands = s1_grouped.get(sid, [])
            if not cands:
                if sid in singleton_entities:
                    score_sum += 1.0
                continue
                
            # Filter by rank cutoff if specified
            if rank_cutoff is not None:
                filtered_cands = [c for c in cands if c[3] <= rank_cutoff]
            else:
                filtered_cands = cands
                
            if not filtered_cands:
                if sid in singleton_entities:
                    score_sum += 1.0
                continue
                
            # Probabilities
            p_idx = 1 if prob_key == 'xgb' else (2 if prob_key == 'lgb' else -1)
            if prob_key == 'ens':
                probs = [0.5 * c[1] + 0.5 * c[2] for c in filtered_cands]
            elif isinstance(prob_key, float): # alpha blend: alpha*xgb + (1-alpha)*lgb
                probs = [prob_key * c[1] + (1.0 - prob_key) * c[2] for c in filtered_cands]
            else:
                probs = [c[p_idx] for c in filtered_cands]
                
            max_p = max(probs) if probs else 0.0
            
            # Singleton cutoff check
            if singleton_cutoff is not None and max_p < singleton_cutoff:
                pred_set = set()
            else:
                # Margin check
                if margin_thr is not None and len(probs) >= 2:
                    sorted_p = sorted(probs, reverse=True)
                    if sorted_p[0] - sorted_p[1] < margin_thr:
                        pred_set = set()
                    else:
                        pred_set = set(c[0] for c, p in zip(filtered_cands, probs) if p >= threshold)
                else:
                    pred_set = set(c[0] for c, p in zip(filtered_cands, probs) if p >= threshold)
                    
            if pred_set:
                score_sum += compute_f05_single(pred_set, gt_map[sid])
            else:
                if sid in singleton_entities:
                    score_sum += 1.0
                    
        return score_sum / total_val_entities

    # A. Global Threshold Sweep for XGBoost (WITHOUT singleton guard vs WITH singleton guard)
    print("\n--- A. Global Threshold Sweep (XGBoost) ---")
    print(f"{'Threshold':>10} | {'Pure (No Guard)':>16} | {'With Guard (0.75)':>18} | {'With Guard (thr)':>18}")
    print("-" * 70)
    for t in [0.50, 0.55, 0.60, 0.63, 0.65, 0.70, 0.75, 0.80]:
        f_no_guard = evaluate_policy('xgb', t, singleton_cutoff=None)
        f_guard_75 = evaluate_policy('xgb', t, singleton_cutoff=0.75)
        f_guard_self = evaluate_policy('xgb', t, singleton_cutoff=t)
        print(f"{t:>10.2f} | {f_no_guard:>16.5f} | {f_guard_75:>18.5f} | {f_guard_self:>18.5f}")
        
    # B. Alpha Ensemble Sweep
    print("\n--- B. Alpha Ensemble Blend Sweep (alpha * XGB + (1-alpha) * LGB) ---")
    print(f"{'Alpha (XGB)':>12} | {'Optimal Threshold':>18} | {'Macro F0.5':>12}")
    print("-" * 50)
    best_alpha = 1.0
    best_ens_f05 = -1.0
    best_ens_t = 0.63
    
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Quick search around 0.60 - 0.66
        best_t_for_alpha = 0.63
        best_f_for_alpha = -1.0
        for t in [0.60, 0.62, 0.63, 0.64, 0.65]:
            f = evaluate_policy(alpha, t, singleton_cutoff=None)
            if f > best_f_for_alpha:
                best_f_for_alpha = f
                best_t_for_alpha = t
        marker = " <<<" if best_f_for_alpha > best_ens_f05 else ""
        if best_f_for_alpha > best_ens_f05:
            best_ens_f05 = best_f_for_alpha
            best_alpha = alpha
            best_ens_t = best_t_for_alpha
        print(f"{alpha:>12.2f} | {best_t_for_alpha:>18.2f} | {best_f_for_alpha:>12.5f}{marker}")
        
    # C. Country-Specific Threshold Evaluation
    print("\n--- C. Country-Specific Threshold Search ---")
    # Evaluate US and India separately
    us_sids = [sid for sid in val_gt_df['source1_entity_id'] if s1_country_map.get(sid) == 'US']
    in_sids = [sid for sid in val_gt_df['source1_entity_id'] if s1_country_map.get(sid) == 'India']
    
    for c_name, c_sids in [('US', us_sids), ('India', in_sids)]:
        print(f"\nScanning thresholds for {c_name} (Entities: {len(c_sids):,})...")
        best_c_t = 0.63
        best_c_f = -1.0
        for t in [0.50, 0.55, 0.60, 0.63, 0.65, 0.70, 0.75]:
            score_sum = 0.0
            for sid in c_sids:
                cands = s1_grouped.get(sid, [])
                pred_set = set(c[0] for c in cands if c[1] >= t)
                if pred_set:
                    score_sum += compute_f05_single(pred_set, gt_map[sid])
                else:
                    if sid in singleton_entities:
                        score_sum += 1.0
            macro = score_sum / len(c_sids)
            marker = " <<< MAX" if macro > best_c_f else ""
            if macro > best_c_f:
                best_c_f = macro
                best_c_t = t
            print(f"  {c_name} Threshold {t:.2f}: Macro F0.5 = {macro:.5f}{marker}")
            
    # D. Margin-Aware Policy Search
    print("\n--- D. Margin-Aware Policy Search (p1 >= thr AND margin >= m_thr) ---")
    for m_thr in [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]:
        f = evaluate_policy('xgb', 0.63, singleton_cutoff=None, margin_thr=m_thr)
        print(f"  Margin Threshold {m_thr:.2f} (with base thr 0.63): Macro F0.5 = {f:.5f}")
        
    # E. Rank-Aware Policy
    print("\n--- E. Rank Cutoff Policy Search ---")
    for rk in [1, 2, 3, 5, 10, 20, 30]:
        f = evaluate_policy('xgb', 0.63, singleton_cutoff=None, rank_cutoff=rk)
        print(f"  Max Candidate Rank <= {rk:>2}: Macro F0.5 = {f:.5f}")
        
    print(f"\nStep 4 & 5 Complete in {time.time() - t0:.2f}s!")

if __name__ == '__main__':
    run_step4_and_5_diagnostics()
