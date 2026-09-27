import os
import re
import gc
import random
import time
import psutil
import sqlite3
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz

from preprocess import (
    normalize_text, get_compact_signature, get_acronym, decompose_address,
    compute_name_indic_translit_sim, compute_postal_match
)

FEATURE_COLS = [
    'name_ratio', 'name_token_sort', 'name_token_set', 'name_partial',
    'name_compact_match', 'name_acronym_match', 'is_addr_missing',
    'street_num_match', 'street_name_sim', 'city_state_sim',
    'addr_token_sort', 'digits_match', 'country_match',
    'is_dba_pattern', 'source_origin',
    'candidate_rank',
    'dense_cosine_sim',
    'name_indic_translit_sim',
    'postal_match'
]
assert len(FEATURE_COLS) == 19, f"Expected 19 features, got {len(FEATURE_COLS)}"

digits_re = re.compile(r'\d+')
def extract_digits(text):
    if not text:
        return ""
    return "".join(digits_re.findall(str(text)))

def log_memory(label=""):
    mem = psutil.virtual_memory()
    print(f"  [MEM {label}] Used: {mem.used / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB ({mem.percent}%)", flush=True)

def get_cpu_usage():
    per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
    usage_str = " ".join([f"{int(c)}%" for c in per_cpu])
    return f"CPU Cores: [{usage_str}]"


# Persistent worker read-only connection
_WORKER_CONN = None

def get_worker_conn(db_path):
    global _WORKER_CONN
    if _WORKER_CONN is None:
        abs_path = os.path.abspath(db_path).replace('\\', '/')
        uri = f"file:{abs_path}?mode=ro&immutable=1"
        _WORKER_CONN = sqlite3.connect(uri, uri=True, timeout=60.0)
        _WORKER_CONN.execute("PRAGMA query_only = ON;")
    return _WORKER_CONN


def fetch_by_ids(cur, table, id_list, cols, batch_size=900):
    results = []
    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i+batch_size]
        placeholders = ','.join(['?'] * len(batch))
        cur.execute(f"SELECT {cols} FROM {table} WHERE entity_id IN ({placeholders})", batch)
        results.extend(cur.fetchall())
    return results


def fetch_gt_by_ids(cur, id_list, batch_size=900):
    gt_map = {}
    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i+batch_size]
        placeholders = ','.join(['?'] * len(batch))
        cur.execute(f"SELECT source1_entity_id, matched_entity_ids FROM ground_truth WHERE source1_entity_id IN ({placeholders})", batch)
        for sid, matches in cur.fetchall():
            if matches:
                gt_map[sid] = set(matches.split(','))
    return gt_map


def fast_19_features(s1_rec, c_rec, cand_id, dense_score=0.0, rank_idx=1):
    """
    Computes full 19-feature vector for entity pair:
      1-15. Baseline lexical & address features
      16.   candidate_rank (1..60)
      17.   dense_cosine_sim (from recovered e5-small, default 0.0)
      18.   name_indic_translit_sim (transliteration-aware Indic name similarity)
      19.   postal_match (exact / prefix postal agreement)
    """
    s1_name, s1_comp, s1_acro, s1_addr, s1_num, s1_street, s1_cs, s1_d, s1_country = s1_rec[:9]
    s1_raw_name = s1_rec[9] if len(s1_rec) > 9 else s1_name
    s1_pc = s1_rec[10] if len(s1_rec) > 10 else ""
    
    c_name, c_comp, c_acro, c_addr, c_num, c_street, c_cs, c_d, c_country = c_rec[:9]
    c_raw_name = c_rec[9] if len(c_rec) > 9 else c_name
    c_pc = c_rec[10] if len(c_rec) > 10 else ""
    
    # 1-4. Name similarities
    if s1_name and c_name:
        name_ratio = fuzz.ratio(s1_name, c_name) / 100.0
        name_token_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
        name_token_set = fuzz.token_set_ratio(s1_name, c_name) / 100.0
        name_partial = fuzz.partial_ratio(s1_name, c_name) / 100.0
    else:
        name_ratio = name_token_sort = name_token_set = name_partial = 0.0
        
    # 5. Compact signature match
    if s1_comp and c_comp and (s1_comp == c_comp or (len(s1_comp)>=6 and s1_comp in c_comp) or (len(c_comp)>=6 and c_comp in s1_comp)):
        name_compact_match = 1.0
    else:
        name_compact_match = 0.0
        
    # 6. Acronym match
    if (s1_acro and s1_acro == c_comp) or (c_acro and c_acro == s1_comp):
        name_acronym_match = 1.0
    else:
        name_acronym_match = 0.0
        
    # 7. is_addr_missing
    is_addr_missing = 1.0 if (not s1_addr or not c_addr) else 0.0
    
    # 8. street_num_match
    if s1_num and c_num:
        street_num_match = 1.0 if s1_num == c_num else 0.0
    elif not s1_num and not c_num:
        street_num_match = 0.5
    else:
        street_num_match = 0.5
        
    # 9. street_name_sim
    if s1_street and c_street:
        street_name_sim = fuzz.ratio(s1_street, c_street) / 100.0
    elif not s1_street and not c_street:
        street_name_sim = 0.5
    else:
        street_name_sim = 0.0
        
    # 10. city_state_sim
    if s1_cs and c_cs:
        city_state_sim = fuzz.token_set_ratio(s1_cs, c_cs) / 100.0
    elif not s1_cs and not c_cs:
        city_state_sim = 0.5
    else:
        city_state_sim = 0.0
        
    # 11. addr_token_sort
    if s1_addr and c_addr:
        addr_token_sort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
    else:
        addr_token_sort = 0.0
        
    # 12. digits_match
    if s1_d and c_d:
        digits_match = 1.0 if s1_d == c_d else 0.0
    elif not s1_d and not c_d:
        digits_match = 1.0
    else:
        digits_match = 0.0
        
    # 13. country_match
    country_match = 1.0 if (s1_country and c_country and s1_country == c_country) else 0.0
    
    # 14. is_dba_pattern
    is_dba_pattern = 1.0 if (street_name_sim >= 0.85 and street_num_match == 1.0 and country_match == 1.0) else 0.0
    
    # 15. source_origin
    source_origin = 1.0 if cand_id.startswith('S2-') else 0.0
    
    # 16. candidate_rank (raw integer rank 1..60 as float32)
    rank_val = float(rank_idx)
    assert 1.0 <= rank_val <= 60.0, f"Candidate rank {rank_val} out of expected [1, 60] range"
    
    # 17. dense_cosine_sim
    dense_val = float(dense_score) if dense_score is not None else 0.0
    dense_val = max(0.0, min(1.0, dense_val))
    
    # 18. name_indic_translit_sim
    name_indic_sim = compute_name_indic_translit_sim(s1_raw_name, c_raw_name)
    
    # 19. postal_match
    postal_sim = compute_postal_match(s1_pc or s1_addr, c_pc or c_addr)
    
    res = (name_ratio, name_token_sort, name_token_set, name_partial, name_compact_match, name_acronym_match,
           is_addr_missing, street_num_match, street_name_sim, city_state_sim, addr_token_sort, digits_match,
           country_match, is_dba_pattern, source_origin, rank_val, dense_val, name_indic_sim, postal_sim)
    assert len(res) == 19, f"Expected 19 features, got {len(res)}"
    return res


def fast_17_features(s1_rec, c_rec, cand_id, dense_score=0.0, rank_idx=1):
    return fast_19_features(s1_rec, c_rec, cand_id, dense_score=dense_score, rank_idx=rank_idx)[:17]


def fast_16_features(s1_rec, c_rec, cand_id, rank_idx=1):
    """Computes the 16-feature vector (15 baseline + candidate_rank)."""
    f19 = fast_19_features(s1_rec, c_rec, cand_id, dense_score=0.0, rank_idx=rank_idx)
    return f19[:16]


def fast_15_features(s1_rec, c_rec, cand_id):
    """Backwards-compatible wrapper returning the first 15 features."""
    return fast_19_features(s1_rec, c_rec, cand_id, dense_score=0.0, rank_idx=1)[:15]



def process_chunk(args):
    """
    Worker function:
    1. Fetches ground truth using pure in-memory batched SELECT IN (?, ...).
    2. Selects candidate pairs (Train: up to 8 negatives/pos; Val: top 30 undownsampled).
    3. Fetches candidate and query attributes using pure in-memory batched SELECT IN (?, ...).
    4. Pre-normalizes names and addresses once.
    5. Computes the 16 features (15 baseline + candidate_rank) and formats directly as CSV strings.
    """
    chunk_records, db_path = args
    
    s1_needed = [r[0] for r in chunk_records]
    if not s1_needed:
        return "", "", 0, 0, 0, 0
        
    conn = get_worker_conn(db_path)
    cur = conn.cursor()
    
    gt_map = fetch_gt_by_ids(cur, s1_needed, batch_size=900)
            
    pairs_train = []
    pairs_val = []
    needed_cand_ids = set()
    
    for rec in chunk_records:
        s1_id, cands_str, is_val = rec[:3]
        row_scores = rec[3] if len(rec) > 3 else None
            
        if not cands_str or cands_str == 'nan':
            continue
        true_matches = gt_map.get(s1_id, set())
        raw_cands = [c.strip() for c in cands_str.split(',') if c.strip()]
        
        if is_val:
            for rank_idx, cid in enumerate(raw_cands, start=1):
                assert 1 <= rank_idx <= 60, f"Validation candidate rank {rank_idx} is out of expected [1, 60] range"
                is_match = 1 if cid in true_matches else 0
                dense_score = float(row_scores[rank_idx - 1]) if (row_scores is not None and rank_idx <= len(row_scores)) else 0.0
                pairs_val.append((s1_id, cid, is_match, rank_idx, dense_score))
                needed_cand_ids.add(cid)
        else:
            for rank_idx, cid in enumerate(raw_cands, start=1):
                assert 1 <= rank_idx <= 60, f"Training candidate rank {rank_idx} is out of expected [1, 60] range"
                is_match = 1 if cid in true_matches else 0
                dense_score = float(row_scores[rank_idx - 1]) if (row_scores is not None and rank_idx <= len(row_scores)) else 0.0
                pairs_train.append((s1_id, cid, is_match, rank_idx, dense_score))
                needed_cand_ids.add(cid)
                    
    if not pairs_train and not pairs_val:
        return "", "", 0, 0, 0, 0
        
    needed_cand_list = list(needed_cand_ids)
    cat_rows = fetch_by_ids(cur, "catalog", needed_cand_list, "entity_id, business_name, business_address, country", batch_size=900)
    
    catalog_dict = {}
    for eid, name, addr, country in cat_rows:
        n_name = normalize_text(name)
        c_comp = get_compact_signature(name)
        c_acro = get_acronym(name)
        c_addr, s_num, s_name, c_cs, c_pc = decompose_address(addr)
        d_str = extract_digits(f"{n_name} {addr}")
        catalog_dict[eid] = (n_name, c_comp, c_acro, c_addr, s_num, s_name, c_cs, d_str, country or "", str(name) if pd.notna(name) else "", c_pc or "")
    del cat_rows
    
    s1_rows = fetch_by_ids(cur, "s1_catalog", s1_needed, "entity_id, business_name, business_address, country", batch_size=900)
    
    s1_dict = {}
    for eid, name, addr, country in s1_rows:
        n_name = normalize_text(name)
        c_comp = get_compact_signature(name)
        c_acro = get_acronym(name)
        c_addr, s_num, s_name, c_cs, c_pc = decompose_address(addr)
        d_str = extract_digits(f"{n_name} {addr}")
        s1_dict[eid] = (n_name, c_comp, c_acro, c_addr, s_num, s_name, c_cs, d_str, country or "", str(name) if pd.notna(name) else "", c_pc or "")
    del s1_rows
    
    empty_rec = ("", "", "", "", "", "", "", "", "", "", "")
    train_lines = []
    val_lines = []
    train_pos = train_neg = val_pos = val_neg = 0
    
    for s1_id, cand_id, label, r_idx, dense_s in pairs_train:
        s1_rec = s1_dict.get(s1_id, empty_rec)
        c_rec = catalog_dict.get(cand_id, empty_rec)
        f = fast_19_features(s1_rec, c_rec, cand_id, dense_score=dense_s, rank_idx=r_idx)
        assert len(f) == 19, f"Feature vector length must be 19, got {len(f)}"
        train_lines.append(f"{f[0]:.4f},{f[1]:.4f},{f[2]:.4f},{f[3]:.4f},{f[4]:.1f},{f[5]:.1f},"
                           f"{f[6]:.1f},{f[7]:.1f},{f[8]:.4f},{f[9]:.4f},{f[10]:.4f},{f[11]:.1f},"
                           f"{f[12]:.1f},{f[13]:.1f},{f[14]:.1f},{f[15]:.1f},{f[16]:.4f},{f[17]:.4f},{f[18]:.1f},"
                           f"{s1_id},{cand_id},{label}\n")
        if label == 1:
            train_pos += 1
        else:
            train_neg += 1
            
    for s1_id, cand_id, label, r_idx, dense_s in pairs_val:
        s1_rec = s1_dict.get(s1_id, empty_rec)
        c_rec = catalog_dict.get(cand_id, empty_rec)
        f = fast_19_features(s1_rec, c_rec, cand_id, dense_score=dense_s, rank_idx=r_idx)
        assert len(f) == 19, f"Feature vector length must be 19, got {len(f)}"
        val_lines.append(f"{f[0]:.4f},{f[1]:.4f},{f[2]:.4f},{f[3]:.4f},{f[4]:.1f},{f[5]:.1f},"
                         f"{f[6]:.1f},{f[7]:.1f},{f[8]:.4f},{f[9]:.4f},{f[10]:.4f},{f[11]:.1f},"
                         f"{f[12]:.1f},{f[13]:.1f},{f[14]:.1f},{f[15]:.1f},{f[16]:.4f},{f[17]:.4f},{f[18]:.1f},"
                         f"{s1_id},{cand_id},{label}\n")
        if label == 1:
            val_pos += 1
        else:
            val_neg += 1
            
    return "".join(train_lines), "".join(val_lines), train_pos, train_neg, val_pos, val_neg


def find_last_processed_query(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 1024:
        return None
    with open(csv_path, 'rb') as f:
        f.seek(max(0, os.path.getsize(csv_path) - 20000))
        lines = f.read().decode('utf-8', errors='ignore').strip().split('\n')
        for line in reversed(lines):
            parts = line.strip().split(',')
            if len(parts) >= 18:
                return parts[-3]
    return None


def tsv_chunk_generator(tsv_path, chunk_size, val_set, last_s1_id=None, max_chunks=None, scores_mmap=None):
    with open(tsv_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        resumed = False if last_s1_id else True
        skipped = 0
        
        batch = []
        chunks_count = 0
        row_counter = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx = line.find('\t')
            if idx == -1:
                continue
            s1_id = line[:idx]
            cands_str = line[idx+1:]
            
            if not resumed:
                skipped += 1
                row_counter += 1
                if s1_id == last_s1_id:
                    resumed = True
                    print(f"Checkpoint matched! Resuming immediately after query #{skipped:,} ({last_s1_id})...")
                continue
                
            is_val = s1_id in val_set
            if scores_mmap is not None:
                cands = [c.strip() for c in cands_str.split(',') if c.strip()]
                n_scores = scores_mmap.shape[1] if len(scores_mmap.shape) > 1 else 0
                avail = min(len(cands), n_scores)
                row_scores = list(scores_mmap[row_counter][:avail])
                if len(cands) > avail:
                    row_scores.extend([0.0] * (len(cands) - avail))
                batch.append((s1_id, cands_str, is_val, row_scores))
            else:
                batch.append((s1_id, cands_str, is_val))
            row_counter += 1
            if len(batch) >= chunk_size:
                yield batch, skipped + chunks_count * chunk_size
                batch = []
                chunks_count += 1
                if max_chunks is not None and chunks_count >= max_chunks:
                    return
        if batch and (max_chunks is None or chunks_count < max_chunks):
            yield batch, skipped + chunks_count * chunk_size


def build_train_sqlite_catalog(s1_path, s2_path, s3_path, gt_path, db_path):
    """Builds persistent SQLite catalog tables for multiprocessing feature extraction."""
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        try:
            cur.execute("SELECT count(*) FROM catalog")
            cat_count = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM s1_catalog")
            s1_count = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM ground_truth")
            gt_count = cur.fetchone()[0]
            if cat_count > 0 and s1_count > 0 and gt_count > 0:
                print(f"  Found verified SQLite train catalog ({cat_count:,} catalog records, {s1_count:,} S1 records, {gt_count:,} GT records)!")
                conn.close()
                return
        except Exception:
            pass
        conn.close()

    print(f"\nBuilding SQLite catalog for training at {db_path}...")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    cur = conn.cursor()

    cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='catalog'")
    if cur.fetchone()[0] == 0:
        print("  Importing S2 and S3 catalogs into SQLite...")
        for source_file in [s2_path, s3_path]:
            for chunk in tqdm(pd.read_csv(source_file, sep="\t", chunksize=250000, usecols=['entity_id', 'business_name', 'business_address', 'country']), desc=f"Importing {os.path.basename(source_file)}"):
                chunk.to_sql("catalog", conn, if_exists="append", index=False)
                conn.commit()
        print("  Indexing catalog table on entity_id...")
        conn.execute("CREATE INDEX idx_entity_id ON catalog(entity_id)")
        conn.commit()

    cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='s1_catalog'")
    if cur.fetchone()[0] == 0:
        print("  Importing S1 catalog into SQLite...")
        for chunk in tqdm(pd.read_csv(s1_path, sep="\t", chunksize=250000, usecols=['entity_id', 'business_name', 'business_address', 'country']), desc="Importing S1"):
            chunk.to_sql("s1_catalog", conn, if_exists="append", index=False)
            conn.commit()
        print("  Indexing s1_catalog table on entity_id...")
        conn.execute("CREATE INDEX idx_s1_entity_id ON s1_catalog(entity_id)")
        conn.commit()

    cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='ground_truth'")
    if cur.fetchone()[0] == 0:
        print("  Importing Ground Truth into SQLite...")
        for chunk in tqdm(pd.read_csv(gt_path, sep="\t", chunksize=250000, usecols=['source1_entity_id', 'matched_entity_ids']), desc="Importing GT"):
            chunk.to_sql("ground_truth", conn, if_exists="append", index=False)
            conn.commit()
        print("  Indexing ground_truth table on source1_entity_id...")
        conn.execute("CREATE INDEX idx_gt_s1 ON ground_truth(source1_entity_id)")
        conn.commit()

    conn.commit()
    conn.close()
    print("  SQLite train database ready!")


def build_v3_feature_datasets(cands_path, val_split_path, out_train_path, out_val_path, resume=True, max_chunks=None, scores_path=None, db_path=None):
    log_memory("Feature Extraction V3 Start")
    
    if db_path is None:
        db_path = os.path.join(os.path.dirname(out_train_path), "train_catalog_temp.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite train DB not found at: {db_path}")
        
    print(f"Loading validation set IDs from: {val_split_path}...")
    val_df = pd.read_csv(val_split_path, sep="\t", usecols=['source1_entity_id'])
    val_set = set(val_df['source1_entity_id'])
    print(f"  Loaded {len(val_set):,} validation IDs into memory ({len(val_set)/2206821*100:.1f}% split).")
    del val_df
    gc.collect()
    
    # Check for dense scores file (best_scores.npy or candidate_dense_scores.npy)
    scores_mmap = None
    candidate_scores_paths = [
        scores_path,
        os.path.join(os.path.dirname(cands_path), "embeddings_cache", "best_scores.npy"),
        os.path.join(os.path.dirname(cands_path), "best_scores.npy"),
        os.path.join(os.path.dirname(cands_path), "candidate_dense_scores.npy")
    ]
    for sp in candidate_scores_paths:
        if sp and os.path.exists(sp):
            print(f"Loading cached dense scores from: {sp} (mmap)...")
            scores_mmap = np.load(sp, mmap_mode='r')
            print(f"  Loaded scores cache: shape={scores_mmap.shape}, dtype={scores_mmap.dtype}")
            break
            
    if scores_mmap is None:
        print("No dense scores cache found. Defaulting dense_cosine_sim to 0.0.")
    
    header = ",".join(FEATURE_COLS) + ",source1_entity_id,candidate_entity_id,label\n"
              
    last_s1 = find_last_processed_query(out_train_path) if resume else None
    
    if last_s1:
        print(f"Detected checkpoint at query: {last_s1}!")
        file_mode = 'a'
        print(f"  Current Train File: {os.path.getsize(out_train_path)/(1024**2):.1f} MB")
        print(f"  Current Val File: {os.path.getsize(out_val_path)/(1024**2):.1f} MB")
    else:
        file_mode = 'w'
        for p in [out_train_path, out_val_path]:
            if os.path.exists(p):
                os.remove(p)
                
    num_cores = min(6, max(1, os.cpu_count() - 2))
    chunk_size = 1500
    MAX_IN_FLIGHT = 8
    
    print(f"\nStarting Zero-Lock In-Memory ProcessPoolExecutor with {num_cores} workers (chunk_size={chunk_size})...", flush=True)
    print("Initial " + get_cpu_usage(), flush=True)
    
    tot_train_pos = tot_train_neg = tot_val_pos = tot_val_neg = 0
    completed_chunks = 0
    start_time = time.time()
    
    with open(out_train_path, file_mode, encoding='utf-8', buffering=2*1024*1024) as f_train, \
         open(out_val_path, file_mode, encoding='utf-8', buffering=2*1024*1024) as f_val:
         
        if file_mode == 'w':
            f_train.write(header)
            f_val.write(header)
            
        with ProcessPoolExecutor(max_workers=num_cores, max_tasks_per_child=50) as executor:
            futures = set()
            
            def harvest_completed(futures_set):
                nonlocal completed_chunks, tot_train_pos, tot_train_neg, tot_val_pos, tot_val_neg
                done, remaining = wait(futures_set, return_when=FIRST_COMPLETED)
                for fut in done:
                    train_csv, val_csv, t_pos, t_neg, v_pos, v_neg = fut.result()
                    if train_csv:
                        f_train.write(train_csv)
                    if val_csv:
                        f_val.write(val_csv)
                    tot_train_pos += t_pos
                    tot_train_neg += t_neg
                    tot_val_pos += v_pos
                    tot_val_neg += v_neg
                    completed_chunks += 1
                    
                    if completed_chunks % 50 == 0:
                        elapsed = time.time() - start_time
                        queries_done = completed_chunks * chunk_size
                        qps = queries_done / max(elapsed, 0.001)
                        total_remaining_queries = 2206821 - 1687499
                        rem_sec = max(0, (total_remaining_queries - queries_done) / max(qps, 1))
                        overall_done = 1687499 + queries_done
                        print(f"\n[Progress] {completed_chunks:,} chunks ({overall_done:,} / 2,206,821 queries: {overall_done/2206821*100:.1f}%) in {elapsed:.1f}s | Speed: {qps:,.0f} q/s | ETA: {rem_sec:.0f}s", flush=True)
                        print(f"  {get_cpu_usage()}", flush=True)
                        print(f"  Session New Train Rows: {tot_train_pos + tot_train_neg:,} | Val Rows: {tot_val_pos + tot_val_neg:,} [100% UNDOWNSAMPLED]", flush=True)
                        log_memory(f"Chunk {completed_chunks}")
                        
                return remaining

            for batch, est_offset in tsv_chunk_generator(cands_path, chunk_size, val_set, last_s1_id=last_s1, max_chunks=max_chunks, scores_mmap=scores_mmap):
                while len(futures) >= MAX_IN_FLIGHT:
                    futures = harvest_completed(futures)
                    
                mem = psutil.virtual_memory()
                while mem.percent > 85.0:
                    print(f"  [System Memory Alert: {mem.percent}%] Pausing 2s to allow external apps to release RAM...", flush=True)
                    time.sleep(2.0)
                    mem = psutil.virtual_memory()
                    
                futures.add(executor.submit(process_chunk, (batch, db_path)))
                
            while futures:
                futures = harvest_completed(futures)
                
    total_elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"17-Feature Extraction Complete! (Session finished in {total_elapsed:.1f}s / {total_elapsed/60:.1f}m)")
    print(f"  Train Dataset: {out_train_path} ({os.path.getsize(out_train_path)/(1024**2):.1f} MB)")
    print(f"  Validation Dataset (100% Undownsampled): {out_val_path} ({os.path.getsize(out_val_path)/(1024**2):.1f} MB)")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp
    mp.freeze_support()
    
    parser = argparse.ArgumentParser(description="Feature Engineering (16 features)")
    parser.add_argument("--cands", default=None, help="Path to full_train_candidate_pairs.tsv")
    parser.add_argument("--val-split", default=None, help="Path to val_gt_split.tsv")
    parser.add_argument("--out-train", default=None, help="Path to output full_train_features_v3.csv")
    parser.add_argument("--out-val", default=None, help="Path to output full_val_features_v3.csv")
    parser.add_argument("--db-path", default=None, help="Path to train_catalog_temp.db")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from last checkpoint")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    out_dir = os.path.join(repo_root, "output")
    os.makedirs(out_dir, exist_ok=True)
    
    cands_path = args.cands or os.path.join(out_dir, "full_train_candidate_pairs.tsv")
    val_split_path = args.val_split or os.path.join(out_dir, "val_gt_split.tsv")
    out_train_path = args.out_train or os.path.join(out_dir, "full_train_features_v3.csv")
    out_val_path = args.out_val or os.path.join(out_dir, "full_val_features_v3.csv")
    
    build_v3_feature_datasets(cands_path, val_split_path, out_train_path, out_val_path, resume=args.resume, db_path=args.db_path)
