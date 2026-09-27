import time
import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from rapidfuzz import fuzz

import sys
sys.path.append('code/business_entity_resolution/src')
from preprocess import normalize_text, get_compact_signature, get_acronym, decompose_address
from inference import extract_digits, fast_16_features, FEATURE_COLS

def run_france_proxy_and_test_confidence():
    print("=" * 70)
    print("FRANCE PROXY DIAGNOSTIC & TEST CONFIDENCE / MARGIN ANALYSIS")
    print("=" * 70)
    
    random.seed(42)
    np.random.seed(42)
    
    # 1. Load test S1
    test_s1_path = 'student_resource/dataset/test/test_source1.tsv'
    print(f"Loading {test_s1_path}...")
    df_s1 = pd.read_csv(test_s1_path, sep='\t', dtype=str)
    
    france_s1 = df_s1[df_s1['country'] == 'France']
    us_s1 = df_s1[df_s1['country'] == 'US']
    india_s1 = df_s1[df_s1['country'] == 'India']
    
    print(f"France test S1 count: {len(france_s1):,}")
    print(f"US test S1 count:     {len(us_s1):,}")
    print(f"India test S1 count:  {len(india_s1):,}")
    
    # Sample 40 France entities for detailed manual inspection
    france_sample = france_s1.sample(n=40, random_state=42)
    france_sample_ids = set(france_sample['entity_id'])
    
    # Also sample 500 from each for confidence statistics (p1, p2, margin)
    sample_size = 500
    s_fr = france_s1.sample(n=sample_size, random_state=101)
    s_us = us_s1.sample(n=sample_size, random_state=101)
    s_in = india_s1.sample(n=sample_size, random_state=101)
    
    all_sample_s1 = pd.concat([france_sample, s_fr, s_us, s_in]).drop_duplicates(subset=['entity_id'])
    all_s1_dict = {row['entity_id']: row for _, row in all_sample_s1.iterrows()}
    all_s1_ids = set(all_s1_dict.keys())
    print(f"Total sampled S1 entities for confidence/France proxy: {len(all_s1_ids):,}")
    
    # 2. Extract candidates for sampled S1 entities from candidate_pairs.tsv
    cand_pairs_path = 'output/candidate_pairs.tsv'
    print(f"Reading candidates from {cand_pairs_path} for sampled queries...")
    sampled_cands_map = {} # s1_id -> list of (cand_id, rank)
    needed_cand_ids = set()
    
    with open(cand_pairs_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx = line.find('\t')
            if idx == -1:
                continue
            sid = line[:idx]
            if sid in all_s1_ids:
                cstr = line[idx+1:]
                c_list = [c.strip() for c in cstr.split(',') if c.strip()]
                sampled_cands_map[sid] = [(c, r) for r, c in enumerate(c_list, start=1)]
                for c in c_list:
                    needed_cand_ids.add(c)
                    
    print(f"Found candidates for {len(sampled_cands_map):,} queries. Total needed catalog entities: {len(needed_cand_ids):,}")
    
    # 3. Retrieve catalog entity metadata from test_source2.tsv and test_source3.tsv
    catalog_dict = {}
    for s_path in ['student_resource/dataset/test/test_source2.tsv', 'student_resource/dataset/test/test_source3.tsv']:
        print(f"Scanning {s_path} for {len(needed_cand_ids) - len(catalog_dict):,} missing entities...")
        with open(s_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')
            eid_idx = header.index('entity_id')
            name_idx = header.index('business_name')
            addr_idx = header.index('business_address')
            ctry_idx = header.index('country')
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > eid_idx:
                    eid = parts[eid_idx]
                    if eid in needed_cand_ids and eid not in catalog_dict:
                        catalog_dict[eid] = {
                            'entity_id': eid,
                            'business_name': parts[name_idx] if len(parts) > name_idx else "",
                            'business_address': parts[addr_idx] if len(parts) > addr_idx else "",
                            'country': parts[ctry_idx] if len(parts) > ctry_idx else ""
                        }
                        if len(catalog_dict) == len(needed_cand_ids):
                            break
                            
    print(f"Loaded {len(catalog_dict):,} catalog entities into memory.")
    
    # 4. Preprocess helper
    def prep_rec(d):
        name = d.get('business_name', '')
        addr = d.get('business_address', '')
        country = d.get('country', '')
        n_name = normalize_text(name)
        c_comp = get_compact_signature(name)
        c_acro = get_acronym(name)
        c_addr, s_num, s_name, c_cs, c_pc = decompose_address(addr)
        d_str = extract_digits(f"{n_name} {addr}")
        return (n_name, c_comp, c_acro, c_addr, s_num, s_name, c_cs, d_str, country or "")
        
    s1_prepped = {sid: prep_rec(row) for sid, row in all_s1_dict.items()}
    cat_prepped = {cid: prep_rec(row) for cid, row in catalog_dict.items()}
    
    # 5. Load XGBoost model
    xgb_model_path = 'output/xgb_model_v3.json'
    print(f"Loading XGBoost model from {xgb_model_path}...")
    bst = xgb.Booster()
    bst.load_model(xgb_model_path)
    
    # 6. Score candidate pairs for sampled queries
    # Collect features
    all_pairs = []
    pairs_meta = []
    
    for sid, c_list in sampled_cands_map.items():
        s_rec = s1_prepped[sid]
        for cid, rk in c_list:
            c_rec = cat_prepped.get(cid, ("", "", "", "", "", "", "", "", ""))
            feats = fast_16_features(s_rec, c_rec, cid, rank_idx=rk)
            all_pairs.append(feats)
            pairs_meta.append((sid, cid, rk))
            
    X_mat = np.array(all_pairs, dtype=np.float32)
    dmat = xgb.DMatrix(X_mat, feature_names=FEATURE_COLS)
    probs = bst.predict(dmat)
    
    # Group predictions by s1_id
    preds_by_s1 = {}
    for (sid, cid, rk), p in zip(pairs_meta, probs):
        if sid not in preds_by_s1:
            preds_by_s1[sid] = []
        preds_by_s1[sid].append((cid, rk, float(p)))
        
    for sid in preds_by_s1:
        preds_by_s1[sid].sort(key=lambda x: x[2], reverse=True)
        
    # 7. Compute Confidence Statistics (p1, p2, margin = p1 - p2) per country
    print("\n" + "=" * 50)
    print("TEST CONFIDENCE STATISTICS (p1, p2, margin) BY COUNTRY")
    print("=" * 50)
    
    for c_name, sample_df in [('US', s_us), ('India', s_in), ('France', s_fr)]:
        p1_list = []
        p2_list = []
        margin_list = []
        c_sids = set(sample_df['entity_id'])
        
        for sid in c_sids:
            cands = preds_by_s1.get(sid, [])
            if len(cands) == 0:
                p1_list.append(0.0)
                p2_list.append(0.0)
                margin_list.append(0.0)
            elif len(cands) == 1:
                p1_list.append(cands[0][2])
                p2_list.append(0.0)
                margin_list.append(cands[0][2])
            else:
                p1 = cands[0][2]
                p2 = cands[1][2]
                p1_list.append(p1)
                p2_list.append(p2)
                margin_list.append(p1 - p2)
                
        p1_arr = np.array(p1_list)
        p2_arr = np.array(p2_list)
        m_arr = np.array(margin_list)
        
        print(f"\n--- {c_name} (Sample size: {len(p1_arr)}) ---")
        print(f"  Mean p1:         {np.mean(p1_arr):.4f} | Median p1:     {np.median(p1_arr):.4f}")
        print(f"  Mean p2:         {np.mean(p2_arr):.4f} | Median p2:     {np.median(p2_arr):.4f}")
        print(f"  Mean Margin:     {np.mean(m_arr):.4f} | Median Margin: {np.median(m_arr):.4f}")
        print(f"  p1 >= 0.75:      {np.mean(p1_arr >= 0.75)*100:.2f}% (passes singleton guard)")
        print(f"  0.63 <= p1 < 0.75: {np.mean((p1_arr >= 0.63) & (p1_arr < 0.75))*100:.2f}% (killed by singleton guard!)")
        print(f"  p1 < 0.63:       {np.mean(p1_arr < 0.63)*100:.2f}% (below threshold)")
        
    # 8. Detailed France Proxy Manual Inspection
    print("\n" + "=" * 70)
    print("FRANCE PROXY DIAGNOSTIC: INSPECTION OF 40 RANDOMLY SAMPLED ENTITIES")
    print("=" * 70)
    
    france_results = []
    for sid in france_sample['entity_id']:
        s_data = all_s1_dict[sid]
        s_name = s_data.get('business_name', '')
        s_addr = s_data.get('business_address', '')
        cands = preds_by_s1.get(sid, [])
        
        # Check candidates
        plausible_cands = []
        top_cand_info = []
        for cid, rk, p in cands[:5]:
            c_data = catalog_dict.get(cid, {})
            c_name = c_data.get('business_name', '')
            c_addr = c_data.get('business_address', '')
            
            # Simple lexical score for reference
            n_ratio = fuzz.ratio(str(s_name).lower(), str(c_name).lower())
            tok_set = fuzz.token_set_ratio(str(s_name).lower(), str(c_name).lower())
            
            # Plausibility check:
            # 1. High name similarity (ratio >= 80 or token_set >= 90)
            # 2. Or model p >= 0.60
            is_plausible = (tok_set >= 85 and n_ratio >= 65) or p >= 0.60
            if is_plausible:
                plausible_cands.append((cid, c_name, c_addr, p, tok_set))
                
            top_cand_info.append({
                'cid': cid, 'rank': rk, 'p': p, 'name': c_name, 'addr': c_addr, 'tok_set': tok_set
            })
            
        has_plausible = len(plausible_cands) > 0
        france_results.append({
            'sid': sid,
            'name': s_name,
            'addr': s_addr,
            'cand_count': len(cands),
            'top_p': cands[0][2] if cands else 0.0,
            'has_plausible': has_plausible,
            'plausible_cands': plausible_cands,
            'top_candidates': top_cand_info
        })
        
    plausible_count = sum(1 for r in france_results if r['has_plausible'])
    print(f"\nSummary of France Manual Proxy:")
    print(f"  Total Sampled France Queries: {len(france_results)}")
    print(f"  Queries with >=1 Plausible Candidate in Top-5: {plausible_count} ({plausible_count/len(france_results)*100:.1f}%)")
    print(f"  Queries with Top p >= 0.63: {sum(1 for r in france_results if r['top_p'] >= 0.63)} ({sum(1 for r in france_results if r['top_p'] >= 0.63)/len(france_results)*100:.1f}%)")
    print(f"  Queries with Top p >= 0.75: {sum(1 for r in france_results if r['top_p'] >= 0.75)} ({sum(1 for r in france_results if r['top_p'] >= 0.75)/len(france_results)*100:.1f}%)")
    
    # Print 10 concrete examples of France queries and candidates
    print("\n--- 10 Concrete Examples of France Test Queries & Retrieved Candidates ---")
    for i, r in enumerate(france_results[:10], start=1):
        print(f"\n[Example {i}] Query {r['sid']}")
        print(f"  Query Name:    '{r['name']}'")
        print(f"  Query Address: '{r['addr']}'")
        print(f"  Total Candidates in pool: {r['cand_count']}")
        print(f"  Plausible Candidate Present: {'YES' if r['has_plausible'] else 'NO'}")
        for c in r['top_candidates'][:3]:
            print(f"    Cand {c['cid']} (Rank {c['rank']}): p={c['p']:.4f}, tok_set={c['tok_set']} | '{c['name']}' | '{c['addr']}'")

if __name__ == '__main__':
    run_france_proxy_and_test_confidence()
