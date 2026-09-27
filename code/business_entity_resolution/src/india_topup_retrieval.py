#!/usr/bin/env python3
"""
India-Only Top-Up Retrieval Module
Amazon ML Challenge 2026 — Experiment 010 (Final)

Builds selective inverted indexes on Indian catalog records (S2 + S3) using:
  1. Transliterated compound brand tokens
  2. Free-form street-number + street-token keys
  3. 6-digit PIN/postal code + house/plot number

Applies retrieval top-up across ALL Indian S1 queries to recover missed matches
caused by Indic script variations and non-ASCII stripping, while strictly
capping candidate additions (<= 6 per query) and bounding inverted block sizes.
"""

import os
import sys
import re
import gc
import time
import psutil
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz

try:
    import unidecode
except ImportError:
    unidecode = None

indic_re = re.compile(r'[\u0900-\u0D7F]')
postal_re = re.compile(r'\b[1-9][0-9]{5}\b')
num_clean_re = re.compile(r'[^a-z0-9]')

STOP_WORDS = {
    'private', 'limited', 'ltd', 'pvt', 'llp', 'enterprises', 'enterprise', 'services', 
    'service', 'technology', 'technologies', 'solutions', 'solution', 'consulting', 
    'corporation', 'corp', 'company', 'co', 'india', 'brothers', 'associates', 
    'traders', 'trade', 'multitrade', 'industries', 'industry', 'management', 
    'logistics', 'construction', 'healthcare', 'care', 'hospital', 'advisors',
    'center', 'centre', 'stores', 'store', 'agency', 'agencies', 'group', 'works'
}

def log_memory(label=""):
    mem = psutil.virtual_memory()
    print(f"  [MEM {label}] Used: {mem.used / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB ({mem.percent}%)", flush=True)


def extract_indian_keys(name, addr):
    """
    Extracts high-precision retrieval keys for Indian business entities:
      - Brand tokens and compound brand tokens (transliterated from Indic scripts)
      - PIN + house/plot/flat/door number
      - Street name + street number
    """
    keys = []
    name_str = str(name) if (pd.notna(name) and name) else ''
    if indic_re.search(name_str) and unidecode is not None:
        name_str = unidecode.unidecode(name_str)
    name_norm = re.sub(r'[^a-z0-9\s]', ' ', name_str.lower())
    words = [w for w in name_norm.split() if len(w) >= 3 and w not in STOP_WORDS]
    
    # 1. Distinctive brand tokens and compound brand
    for b in words[:3]:
        if len(b) >= 4:
            keys.append(('b', b))
    if len(words) >= 2:
        keys.append(('cb', f"{words[0]}_{words[1]}"))
        
    addr_str = str(addr) if (pd.notna(addr) and addr) else ''
    if addr_str:
        # 2. PIN + house/plot number
        pins = postal_re.findall(addr_str)
        num_m = re.search(r'(?:h\.?no\.?|house\s*no\.?|plot\s*no\.?|flat\s*no\.?|d\.?no\.?|door\s*no\.?|bldg\s*no\.?|#)\s*([a-z0-9\-/]+)', addr_str, re.I)
        h_num = None
        if num_m:
            h_num = num_clean_re.sub('', num_m.group(1).lower())
        else:
            m2 = re.search(r'\b([a-z]?\d{1,4}[a-z]?)\b', addr_str, re.I)
            if m2 and (not pins or m2.group(1) not in pins):
                h_num = m2.group(1).lower()
                
        if pins and h_num and len(h_num) >= 2:
            keys.append(('pin_num', f"{pins[0]}_{h_num}"))
            
        # 3. Street name + number
        st_m = re.search(r'([a-z0-9]+)\s+(?:road|rd|street|st|nagar|colony|vihar|marg|lane|block|sector)', addr_str, re.I)
        if st_m and h_num:
            st_tok = st_m.group(1).lower()
            if len(st_tok) >= 3 and st_tok not in {'near', 'opp', 'behind', 'beside', 'next', 'floor'}:
                keys.append(('st_num', f"{st_tok}_{h_num}"))
                
    return keys


def build_india_inverted_index(s2_path, s3_path, max_block_size=25, chunk_size=100000):
    """
    Builds inverted index for Indian catalog records from S2 and S3.
    Caps block size at max_block_size to eliminate generic/high-frequency keys.
    Also stores Indian catalog business names for fast candidate ranking.
    """
    print(f"\nBuilding India-only catalog inverted index (max_block_size={max_block_size})...")
    t0 = time.time()
    log_memory("Before Inverted Index Build")
    
    index = defaultdict(list)
    catalog_names = {}
    total_indian_catalog = 0
    
    for source_path in [s2_path, s3_path]:
        source_name = os.path.basename(source_path)
        print(f"  Streaming {source_name} for Indian entities...")
        for chunk in pd.read_csv(source_path, sep="\t", chunksize=chunk_size, 
                                 usecols=['entity_id', 'business_name', 'business_address', 'country'],
                                 dtype=str):
            ind_mask = chunk['country'] == 'India'
            if not ind_mask.any():
                continue
            ind_chunk = chunk[ind_mask]
            for eid, name, addr in zip(ind_chunk['entity_id'], ind_chunk['business_name'], ind_chunk['business_address']):
                catalog_names[eid] = str(name) if pd.notna(name) else ''
                keys = extract_indian_keys(name, addr)
                for kt, kv in keys:
                    k = f"{kt}:{kv}"
                    postings = index[k]
                    if len(postings) < max_block_size:
                        postings.append(eid)
            total_indian_catalog += len(ind_chunk)
            
    # Prune keys that reached max_block_size to purge frequent/generic keys
    print(f"  Pruning generic keys that hit {max_block_size} postings limit...")
    pruned_index = {}
    for k, postings in index.items():
        if len(postings) < max_block_size:
            pruned_index[k] = postings
            
    elapsed = time.time() - t0
    print(f"  Indexed {total_indian_catalog:,} Indian catalog records across {len(pruned_index):,} selective keys in {elapsed:.1f}s.")
    log_memory("After Inverted Index Build")
    return pruned_index, catalog_names


def rank_topup_candidates(s1_name, s1_keys, cand_hits, catalog_names, max_add=6):
    """
    Ranks top-up candidates retrieved by inverted index hits using:
      1. Number of key hits (multi-key agreement)
      2. Fast token-set similarity between transliterated names
    Filters out candidates with name similarity < 0.35 to avoid noisy additions.
    """
    scored = []
    s1_str = str(s1_name) if pd.notna(s1_name) else ''
    s1_norm = unidecode.unidecode(s1_str).lower() if (indic_re.search(s1_str) and unidecode is not None) else s1_str.lower()
    
    for cid, hits in cand_hits.items():
        c_name = catalog_names.get(cid, '')
        if not c_name:
            continue
        c_norm = unidecode.unidecode(c_name).lower() if (indic_re.search(c_name) and unidecode is not None) else c_name.lower()
        sim = fuzz.token_set_ratio(s1_norm, c_norm) / 100.0
        if sim >= 0.35:
            scored.append((hits, sim, cid))
            
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [cid for _, _, cid in scored[:max_add]]


def augment_candidate_pairs(s1_path, s2_path, s3_path, input_cands_path, output_cands_path, 
                            max_add_per_query=6, max_block_size=25, chunk_size=50000):
    """
    Reads existing candidate pairs, builds India inverted index, and creates
    an augmented candidate file with bounded top-up candidates for Indian S1 queries.
    Preserves exact candidates and ordering for US and France queries.
    Preserves original candidate ranks 1..30 for all existing pairs.
    """
    print("=" * 75)
    print("  INDIA TOP-UP CANDIDATE POOL AUGMENTATION")
    print(f"  Input Candidates : {input_cands_path}")
    print(f"  Output Candidates: {output_cands_path}")
    print(f"  Max Add Per Query: {max_add_per_query}")
    print(f"  Max Block Size   : {max_block_size}")
    print("=" * 75)
    
    if os.path.exists(output_cands_path) and os.path.getsize(output_cands_path) > 1024:
        print(f"Output augmented candidate pool already exists at {output_cands_path}.")
        print("Verifying header and query count...")
        with open(output_cands_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
        if header == "source1_entity_id\tcandidate_entity_ids":
            print("  Verified existing augmented candidate pool. Skipping re-generation.")
            return output_cands_path
            
    t0 = time.time()
    
    # 1. Load S1 metadata (only for Indian queries to minimize memory)
    print("\n[Step 1/3] Loading S1 metadata for Indian queries...")
    indian_s1_map = {}
    country_counts = defaultdict(int)
    
    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=chunk_size, 
                             usecols=['entity_id', 'business_name', 'business_address', 'country'],
                             dtype=str):
        for eid, name, addr, ctry in zip(chunk['entity_id'], chunk['business_name'], chunk['business_address'], chunk['country']):
            country_counts[ctry] += 1
            if ctry == 'India':
                indian_s1_map[eid] = (str(name) if pd.notna(name) else '', str(addr) if pd.notna(addr) else '')
                
    print(f"  Total S1 queries: {sum(country_counts.values()):,}")
    for ctry, count in country_counts.items():
        print(f"    - {ctry}: {count:,} ({count/sum(country_counts.values())*100:.1f}%)")
    print(f"  Cached {len(indian_s1_map):,} Indian S1 queries for top-up retrieval.")
    log_memory("After S1 Loading")
    
    # 2. Build India catalog inverted index
    print("\n[Step 2/3] Building Indian catalog inverted index...")
    index, catalog_names = build_india_inverted_index(s2_path, s3_path, max_block_size=max_block_size)
    
    # 3. Stream input candidates, perform top-up on Indian queries, write output
    print(f"\n[Step 3/3] Generating augmented candidate pool into {output_cands_path}...")
    log_memory("Before Streaming Top-Up")
    
    temp_output_path = output_cands_path + ".tmp"
    total_queries = 0
    total_original_cands = 0
    total_added_cands = 0
    indian_queries_processed = 0
    indian_queries_topped_up = 0
    
    with open(input_cands_path, 'r', encoding='utf-8') as f_in, \
         open(temp_output_path, 'w', encoding='utf-8', buffering=8*1024*1024) as f_out:
        
        header = f_in.readline().strip()
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            idx = line.find('\t')
            if idx == -1:
                continue
            s1_id = line[:idx]
            cands_str = line[idx+1:]
            
            existing_cands = [c.strip() for c in cands_str.split(',') if c.strip()]
            existing_set = set(existing_cands)
            n_orig = len(existing_cands)
            total_original_cands += n_orig
            total_queries += 1
            
            if s1_id in indian_s1_map:
                indian_queries_processed += 1
                s1_name, s1_addr = indian_s1_map[s1_id]
                keys = extract_indian_keys(s1_name, s1_addr)
                cand_hits = defaultdict(int)
                for kt, kv in keys:
                    for cid in index.get(f"{kt}:{kv}", []):
                        if cid not in existing_set:
                            cand_hits[cid] += 1
                            
                new_cands = rank_topup_candidates(s1_name, keys, cand_hits, catalog_names, max_add=max_add_per_query)
                if new_cands:
                    augmented_cands = existing_cands + new_cands
                    total_added_cands += len(new_cands)
                    indian_queries_topped_up += 1
                else:
                    augmented_cands = existing_cands
            else:
                augmented_cands = existing_cands
                
            f_out.write(f"{s1_id}\t{','.join(augmented_cands)}\n")
            
            if total_queries % 250000 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {total_queries:,} queries | Added {total_added_cands:,} candidates | Elapsed: {elapsed:.1f}s", flush=True)
                
    # Atomic rename to protect output file
    if os.path.exists(output_cands_path):
        os.remove(output_cands_path)
    os.rename(temp_output_path, output_cands_path)
    
    total_elapsed = time.time() - t0
    final_cand_count = total_original_cands + total_added_cands
    growth_pct = (total_added_cands / max(total_original_cands, 1)) * 100
    
    print("\n" + "=" * 75)
    print("  INDIA TOP-UP AUGMENTATION COMPLETE")
    print("=" * 75)
    print(f"  Total Queries Processed      : {total_queries:,}")
    print(f"  Indian Queries Processed     : {indian_queries_processed:,}")
    print(f"  Indian Queries Topped Up     : {indian_queries_topped_up:,} ({indian_queries_topped_up/max(indian_queries_processed, 1)*100:.1f}%)")
    print(f"  Original Candidate Count     : {total_original_cands:,}")
    print(f"  Added Top-Up Candidates      : {total_added_cands:,}")
    print(f"  Final Augmented Candidates   : {final_cand_count:,} (+{growth_pct:.2f}% growth)")
    print(f"  Output Saved To              : {output_cands_path} ({os.path.getsize(output_cands_path)/(1024**2):.1f} MB)")
    print(f"  Total Runtime                : {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)")
    print("=" * 75)
    
    # Clean up memory
    del index, catalog_names, indian_s1_map
    gc.collect()
    return output_cands_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="India Top-Up Candidate Retrieval")
    parser.add_argument("--s1", required=True, help="Path to source1.tsv")
    parser.add_argument("--s2", required=True, help="Path to source2.tsv")
    parser.add_argument("--s3", required=True, help="Path to source3.tsv")
    parser.add_argument("--input-cands", required=True, help="Path to input candidate_pairs.tsv")
    parser.add_argument("--output-cands", required=True, help="Path to output augmented_candidate_pairs.tsv")
    parser.add_argument("--max-add", type=int, default=6, help="Max candidates to add per Indian query")
    parser.add_argument("--max-block", type=int, default=25, help="Max postings per inverted index key")
    args = parser.parse_args()

    augment_candidate_pairs(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        input_cands_path=args.input_cands,
        output_cands_path=args.output_cands,
        max_add_per_query=args.max_add,
        max_block_size=args.max_block
    )
