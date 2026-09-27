import time
import os
import gc
import random
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import sys
sys.path.append('code/business_entity_resolution/src')
from preprocess import normalize_text, normalize_address

def run_test_equivalent_recall_diagnostic(sample_per_country=300):
    print("=" * 75, flush=True)
    print("TEST-EQUIVALENT VALIDATION RECALL CEILING DIAGNOSTIC", flush=True)
    print("Requirement: candidate_rank <= 30 AND dense_cosine_sim >= 0.50", flush=True)
    print("=" * 75, flush=True)
    
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    # 1. Load validation ground truth
    val_gt_path = 'output/val_gt_split.tsv'
    print(f"Loading {val_gt_path}...", flush=True)
    val_gt_df = pd.read_csv(val_gt_path, sep='\t', dtype=str)
    
    gt_map = {}
    for _, row in val_gt_df.iterrows():
        sid = row['source1_entity_id']
        matches_val = row.get('matched_entity_ids', '')
        if pd.notna(matches_val) and str(matches_val).strip() and str(matches_val).strip() != 'nan':
            gt_map[sid] = set(m.strip() for m in str(matches_val).split(',') if m.strip())
            
    # Load country mapping fast using chunks
    s1_train_path = 'student_resource/dataset/train/train_source1.tsv'
    print(f"Loading country mapping from {s1_train_path}...", flush=True)
    s1_country_df = pd.read_csv(s1_train_path, sep='\t', usecols=['entity_id', 'country'], dtype=str)
    s1_country_map = dict(zip(s1_country_df['entity_id'], s1_country_df['country']))
    del s1_country_df
    gc.collect()
    
    us_sids = [sid for sid in gt_map if s1_country_map.get(sid) == 'US']
    in_sids = [sid for sid in gt_map if s1_country_map.get(sid) == 'India']
    print(f"Total Validation S1 with GT: {len(gt_map):,} (US: {len(us_sids):,}, India: {len(in_sids):,})", flush=True)
    
    # Stratified sample
    sampled_us = random.sample(us_sids, min(sample_per_country, len(us_sids)))
    sampled_in = random.sample(in_sids, min(sample_per_country, len(in_sids)))
    sampled_sids = set(sampled_us + sampled_in)
    print(f"Sampled {len(sampled_sids):,} queries ({len(sampled_us)} US, {len(sampled_in)} India).", flush=True)
    
    # Load raw text for only sampled S1
    print("Loading raw text for sampled S1 queries...", flush=True)
    s1_text_map = {}
    for chunk in pd.read_csv(s1_train_path, sep='\t', chunksize=250000, dtype=str):
        matched = chunk[chunk['entity_id'].isin(sampled_sids)]
        for _, row in matched.iterrows():
            s1_text_map[row['entity_id']] = (row['business_name'] or "", row['business_address'] or "")
        if len(s1_text_map) == len(sampled_sids):
            break
            
    # 2. Extract candidate pairs from full_val_features_v3.csv for sampled queries
    val_csv_path = 'output/full_val_features_v3.csv'
    print(f"Streaming {val_csv_path} for sampled queries...", flush=True)
    sampled_rows = []
    chunk_size = 500000
    for chunk in pd.read_csv(val_csv_path, chunksize=chunk_size, usecols=['source1_entity_id', 'candidate_entity_id', 'candidate_rank', 'label'], dtype=str):
        matched = chunk[chunk['source1_entity_id'].isin(sampled_sids)]
        if len(matched) > 0:
            sampled_rows.append(matched)
            
    df_sampled = pd.concat(sampled_rows, ignore_index=True)
    del sampled_rows
    gc.collect()
    df_sampled['candidate_rank'] = df_sampled['candidate_rank'].astype(float).astype(int)
    df_sampled['label'] = df_sampled['label'].astype(int)
    print(f"Found {len(df_sampled):,} candidate rows for the {len(sampled_sids):,} sampled queries.", flush=True)
    
    # 3. Retrieve raw text for needed candidate catalog entities
    needed_cand_ids = set(df_sampled['candidate_entity_id'])
    print(f"Retrieving raw text for {len(needed_cand_ids):,} candidate catalog entities...", flush=True)
    
    cand_text_map = {}
    for s_path in ['student_resource/dataset/train/train_source2.tsv', 'student_resource/dataset/train/train_source3.tsv']:
        print(f"  Scanning {s_path}...", flush=True)
        for chunk in pd.read_csv(s_path, sep='\t', chunksize=300000, usecols=['entity_id', 'business_name', 'business_address'], dtype=str):
            matched = chunk[chunk['entity_id'].isin(needed_cand_ids)]
            for _, row in matched.iterrows():
                cid = row['entity_id']
                if cid not in cand_text_map:
                    cand_text_map[cid] = (row['business_name'] or "", row['business_address'] or "")
            if len(cand_text_map) == len(needed_cand_ids):
                break
        if len(cand_text_map) == len(needed_cand_ids):
            break
            
    print(f"Loaded {len(cand_text_map):,} catalog entity texts.", flush=True)
    
    # 4. Format strings with official multilingual-e5 prefixes
    print("Formatting text features with official prefixes ('query: ' and 'passage: ')...", flush=True)
    s1_prepped = {}
    for sid in sampled_sids:
        name, addr = s1_text_map.get(sid, ("", ""))
        norm_n = normalize_text(name)
        norm_a = normalize_address(addr)[0]
        s1_prepped[sid] = f"query: {norm_n} {norm_a}".strip()
        
    cand_prepped = {}
    for cid in needed_cand_ids:
        name, addr = cand_text_map.get(cid, ("", ""))
        norm_n = normalize_text(name)
        norm_a = normalize_address(addr)[0]
        cand_prepped[cid] = f"passage: {norm_n} {norm_a}".strip()
        
    # 5. Encode using SentenceTransformer
    print("\nLoading SentenceTransformer model 'intfloat/multilingual-e5-small'...", flush=True)
    t_enc_start = time.time()
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    model.max_seq_length = 64
    
    print(f"Encoding {len(s1_prepped):,} S1 queries...", flush=True)
    s1_id_list = list(s1_prepped.keys())
    s1_texts = [s1_prepped[sid] for sid in s1_id_list]
    s1_embs = model.encode(s1_texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    s1_emb_dict = dict(zip(s1_id_list, s1_embs))
    del s1_texts, s1_embs
    gc.collect()
    
    print(f"Encoding {len(cand_prepped):,} candidate passages...", flush=True)
    cand_id_list = list(cand_prepped.keys())
    cand_texts = [cand_prepped[cid] for cid in cand_id_list]
    cand_embs = model.encode(cand_texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    cand_emb_dict = dict(zip(cand_id_list, cand_embs))
    del cand_texts, cand_embs
    gc.collect()
    print(f"Encoding complete in {time.time() - t_enc_start:.2f}s!", flush=True)
    
    # 6. Compute exact dense_cosine_sim
    print("Computing dot-product cosine similarities for all candidate pairs...", flush=True)
    cosines = []
    for sid, cid in zip(df_sampled['source1_entity_id'], df_sampled['candidate_entity_id']):
        e_s1 = s1_emb_dict.get(sid)
        e_c = cand_emb_dict.get(cid)
        if e_s1 is not None and e_c is not None:
            sim = float(np.dot(e_s1, e_c))
        else:
            sim = 0.0
        cosines.append(sim)
        
    df_sampled['dense_cosine_sim'] = np.array(cosines, dtype=np.float32)
    df_sampled['country'] = [s1_country_map.get(sid, 'UNKNOWN') for sid in df_sampled['source1_entity_id']]
    
    # 7. Analyze Cosine Similarity Distribution for True Pairs vs False Pairs
    true_pairs = df_sampled[df_sampled['label'] == 1]
    false_pairs = df_sampled[df_sampled['label'] == 0]
    
    print("\n" + "=" * 50, flush=True)
    print("DENSE COSINE SIMILARITY DISTRIBUTION", flush=True)
    print("=" * 50, flush=True)
    print(f"True Pairs (label=1, N={len(true_pairs):,}):", flush=True)
    print(f"  Mean Cosine:   {true_pairs['dense_cosine_sim'].mean():.4f}", flush=True)
    print(f"  Median Cosine: {true_pairs['dense_cosine_sim'].median():.4f}", flush=True)
    print(f"  Min Cosine:    {true_pairs['dense_cosine_sim'].min():.4f}", flush=True)
    print(f"  Max Cosine:    {true_pairs['dense_cosine_sim'].max():.4f}", flush=True)
    print(f"  Cosine >= 0.50: {(true_pairs['dense_cosine_sim'] >= 0.50).mean()*100:.2f}% ({(true_pairs['dense_cosine_sim'] >= 0.50).sum():,} / {len(true_pairs):,})", flush=True)
    print(f"  Cosine < 0.50:  {(true_pairs['dense_cosine_sim'] < 0.50).mean()*100:.2f}% ({(true_pairs['dense_cosine_sim'] < 0.50).sum():,} / {len(true_pairs):,})", flush=True)
    
    print(f"\nFalse Candidate Pairs (label=0, N={len(false_pairs):,}):", flush=True)
    print(f"  Mean Cosine:   {false_pairs['dense_cosine_sim'].mean():.4f}", flush=True)
    print(f"  Median Cosine: {false_pairs['dense_cosine_sim'].median():.4f}", flush=True)
    print(f"  Cosine >= 0.50: {(false_pairs['dense_cosine_sim'] >= 0.50).mean()*100:.2f}%", flush=True)
    
    print("\nCosine Brackets for True Pairs (label=1):", flush=True)
    brackets = [
        ("< 0.50 (Filtered out by test rule!)", true_pairs['dense_cosine_sim'] < 0.50),
        ("0.50 - 0.60", (true_pairs['dense_cosine_sim'] >= 0.50) & (true_pairs['dense_cosine_sim'] < 0.60)),
        ("0.60 - 0.70", (true_pairs['dense_cosine_sim'] >= 0.60) & (true_pairs['dense_cosine_sim'] < 0.70)),
        ("0.70 - 0.80", (true_pairs['dense_cosine_sim'] >= 0.70) & (true_pairs['dense_cosine_sim'] < 0.80)),
        ("0.80 - 0.90", (true_pairs['dense_cosine_sim'] >= 0.80) & (true_pairs['dense_cosine_sim'] < 0.90)),
        ("0.90 - 1.00", true_pairs['dense_cosine_sim'] >= 0.90),
    ]
    for lbl, m in brackets:
        cnt = m.sum()
        print(f"  {lbl:<38}: {cnt:>6,} ({cnt/len(true_pairs)*100:>6.2f}%)", flush=True)
        
    # 8. RECOMPUTE TEST-EQUIVALENT RECALL
    print("\n" + "=" * 75, flush=True)
    print("RECALL COMPARISON: EXISTING (k<=30) VS TEST-EQUIVALENT (k<=30 & cosine>=0.50)", flush=True)
    print("=" * 75, flush=True)
    
    for c_name, sub_sids in [('US', sampled_us), ('India', sampled_in), ('OVERALL', list(sampled_sids))]:
        sub_df = df_sampled[df_sampled['source1_entity_id'].isin(sub_sids)]
        
        tot_gt_pairs = sum(len(gt_map[sid]) for sid in sub_sids)
        tot_s1_with_gt = len(sub_sids)
        
        # 1. Pool A: rank <= 30
        pool_a_true = sub_df[sub_df['label'] == 1]
        pool_a_retrieved_pairs = len(pool_a_true)
        pool_a_pair_recall = pool_a_retrieved_pairs / tot_gt_pairs if tot_gt_pairs > 0 else 0
        pool_a_s1_hits = pool_a_true['source1_entity_id'].nunique()
        pool_a_ent_recall = pool_a_s1_hits / tot_s1_with_gt
        
        # 2. Pool B: rank <= 30 AND dense_cosine_sim >= 0.50
        pool_b_true = pool_a_true[pool_a_true['dense_cosine_sim'] >= 0.50]
        pool_b_retrieved_pairs = len(pool_b_true)
        pool_b_pair_recall = pool_b_retrieved_pairs / tot_gt_pairs if tot_gt_pairs > 0 else 0
        pool_b_s1_hits = pool_b_true['source1_entity_id'].nunique()
        pool_b_ent_recall = pool_b_s1_hits / tot_s1_with_gt
        
        pair_loss = pool_a_retrieved_pairs - pool_b_retrieved_pairs
        pair_loss_pct = (pool_a_pair_recall - pool_b_pair_recall) * 100
        
        print(f"\n--- {c_name} (S1 Entities: {tot_s1_with_gt:,}, Total GT Pairs: {tot_gt_pairs:,}) ---", flush=True)
        print(f"  [Pool A: k<=30 raw]          Pair Recall: {pool_a_pair_recall*100:>6.2f}% ({pool_a_retrieved_pairs:,} / {tot_gt_pairs:,}) | Entity Recall: {pool_a_ent_recall*100:>6.2f}% ({pool_a_s1_hits:,} / {tot_s1_with_gt:,})", flush=True)
        print(f"  [Pool B: k<=30 & cos>=0.50]   Pair Recall: {pool_b_pair_recall*100:>6.2f}% ({pool_b_retrieved_pairs:,} / {tot_gt_pairs:,}) | Entity Recall: {pool_b_ent_recall*100:>6.2f}% ({pool_b_s1_hits:,} / {tot_s1_with_gt:,})", flush=True)
        print(f"  --> Recall Dropped by cosine>=0.50: -{pair_loss_pct:.2f}% (-{pair_loss:,} true pairs lost)", flush=True)
        
    print(f"\nCompleted in {time.time() - t0:.2f}s!", flush=True)

if __name__ == '__main__':
    run_test_equivalent_recall_diagnostic(sample_per_country=300)
