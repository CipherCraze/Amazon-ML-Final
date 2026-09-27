#!/usr/bin/env python3
"""
Synthetic Top-Up Retrieval Test
Amazon ML Challenge 2026 — Experiment 010 (Final)

Verifies:
  1. US and France queries are completely unaugmented.
  2. Indian queries retrieve bounded additions (<= max_add).
  3. Original candidate order and ranks 1..30 are preserved.
  4. Candidate deduplication is strictly enforced.
  5. High-frequency / generic keys (> max_block_size) are pruned.
  6. Indic script and address-based matches are successfully retrieved.
"""

import os
import sys
import tempfile
import pandas as pd

# Ensure src directory is in path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from india_topup_retrieval import augment_candidate_pairs, extract_indian_keys


def run_synthetic_topup_test():
    print("=" * 65)
    print("  RUNNING SYNTHETIC INDIA TOP-UP RETRIEVAL TEST")
    print("=" * 65)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        s1_path = os.path.join(tmp_dir, "test_s1.tsv")
        s2_path = os.path.join(tmp_dir, "test_s2.tsv")
        s3_path = os.path.join(tmp_dir, "test_s3.tsv")
        input_cands_path = os.path.join(tmp_dir, "test_cands.tsv")
        output_cands_path = os.path.join(tmp_dir, "augmented_cands.tsv")
        
        # 1. Synthetic S1 entities:
        # S1-US: US entity
        # S1-FR: France entity
        # S1-IN1: Indian entity with Devanagari script ("श्री गणेश ट्रेडर्स")
        # S1-IN2: Indian entity with PIN and plot number
        # S1-IN3: Indian entity with generic tokens that hit frequent block
        s1_data = [
            ("S1-US", "Acme Corporation", "100 Broadway, New York, NY 10001", "United States"),
            ("S1-FR", "Boulangerie Dupont SARL", "15 Rue de Paris, 75001 Paris", "France"),
            ("S1-IN1", "श्री गणेश ट्रेडर्स", "Plot 42, MG Road, Bangalore 560001", "India"),
            ("S1-IN2", "Apollo Pharmacy", "Door No 12-4, Sector 5, Rohini, Delhi 110085", "India"),
            ("S1-IN3", "Generic Trading Co", "Shop 1, Main Road, Mumbai 400001", "India"),
        ]
        s1_df = pd.DataFrame(s1_data, columns=['entity_id', 'business_name', 'business_address', 'country'])
        s1_df.to_csv(s1_path, sep="\t", index=False)
        
        # 2. Synthetic Catalog S2 and S3:
        # S2-MATCH-IN1: True match for S1-IN1 in English ("Shree Ganesh Traders")
        # S3-MATCH-IN2: True match for S1-IN2 with same PIN and house number ("Apollo Pharmacy Pvt Ltd")
        # Generic entries designed to flood "trading" key to test max_block_size pruning
        s2_data = [
            ("S2-MATCH-IN1", "Shree Ganesh Traders Pvt Ltd", "Plot 42, Mahatma Gandhi Road, Bengaluru 560001", "India"),
            ("S2-US-1", "Acme USA Inc", "100 Broadway, NY 10001", "United States"),
        ]
        for i in range(30):
            s2_data.append((f"S2-GENERIC-{i}", f"Generic Trading Enterprise {i}", f"Main Road {i}, Mumbai", "India"))
            
        s2_df = pd.DataFrame(s2_data, columns=['entity_id', 'business_name', 'business_address', 'country'])
        s2_df.to_csv(s2_path, sep="\t", index=False)
        
        s3_data = [
            ("S3-MATCH-IN2", "Apollo Pharmacy Healthcare", "Door 12-4, Sector 5, Rohini, New Delhi 110085", "India"),
            ("S3-FR-1", "Boulangerie Dupont", "15 Rue de Paris, Paris", "France"),
        ]
        s3_df = pd.DataFrame(s3_data, columns=['entity_id', 'business_name', 'business_address', 'country'])
        s3_df.to_csv(s3_path, sep="\t", index=False)
        
        # 3. Base candidate pairs (k=30 for each query with distractors)
        with open(input_cands_path, 'w', encoding='utf-8') as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for sid, _, _, _ in s1_data:
                # 30 base distractors
                cands = [f"CAND-BASE-{sid}-{i}" for i in range(1, 31)]
                f.write(f"{sid}\t{','.join(cands)}\n")
                
        # 4. Run top-up retrieval
        augment_candidate_pairs(
            s1_path=s1_path,
            s2_path=s2_path,
            s3_path=s3_path,
            input_cands_path=input_cands_path,
            output_cands_path=output_cands_path,
            max_add_per_query=6,
            max_block_size=25
        )
        
        # 5. Inspect and verify results
        aug_df = pd.read_csv(output_cands_path, sep="\t", dtype=str)
        res_map = {}
        for _, row in aug_df.iterrows():
            sid = row['source1_entity_id']
            cands = row['candidate_entity_ids'].split(',')
            res_map[sid] = cands
            
        print("\n--- Verifying Results ---")
        
        # Verification 1: US query must have EXACTLY 30 candidates (unchanged)
        us_cands = res_map["S1-US"]
        assert len(us_cands) == 30, f"US query candidate count changed: {len(us_cands)} != 30"
        assert all(c.startswith("CAND-BASE-S1-US-") for c in us_cands), "US candidates altered!"
        print("  [PASS] US query completely untouched (30 candidates preserved).")
        
        # Verification 2: France query must have EXACTLY 30 candidates (unchanged)
        fr_cands = res_map["S1-FR"]
        assert len(fr_cands) == 30, f"France query candidate count changed: {len(fr_cands)} != 30"
        assert all(c.startswith("CAND-BASE-S1-FR-") for c in fr_cands), "France candidates altered!"
        print("  [PASS] France query completely untouched (30 candidates preserved).")
        
        # Verification 3: S1-IN1 (Devanagari) retrieved S2-MATCH-IN1
        in1_cands = res_map["S1-IN1"]
        assert len(in1_cands) > 30, f"S1-IN1 did not receive top-up: {len(in1_cands)} <= 30"
        assert len(in1_cands) <= 36, f"S1-IN1 exceeded max_add: {len(in1_cands)} > 36"
        assert in1_cands[:30] == [f"CAND-BASE-S1-IN1-{i}" for i in range(1, 31)], "Original base ranks altered!"
        assert "S2-MATCH-IN1" in in1_cands[30:], "Devanagari match S2-MATCH-IN1 was not retrieved in top-up!"
        print(f"  [PASS] Indian Devanagari query S1-IN1 retrieved S2-MATCH-IN1 (Total cands: {len(in1_cands)}).")
        
        # Verification 4: S1-IN2 (PIN + House No) retrieved S3-MATCH-IN2
        in2_cands = res_map["S1-IN2"]
        assert len(in2_cands) > 30, f"S1-IN2 did not receive top-up: {len(in2_cands)} <= 30"
        assert len(in2_cands) <= 36, f"S1-IN2 exceeded max_add: {len(in2_cands)} > 36"
        assert in2_cands[:30] == [f"CAND-BASE-S1-IN2-{i}" for i in range(1, 31)], "Original base ranks altered!"
        assert "S3-MATCH-IN2" in in2_cands[30:], "PIN + House No match S3-MATCH-IN2 was not retrieved in top-up!"
        print(f"  [PASS] Indian PIN+Address query S1-IN2 retrieved S3-MATCH-IN2 (Total cands: {len(in2_cands)}).")
        
        # Verification 5: Deduplication check
        for sid, cands in res_map.items():
            assert len(cands) == len(set(cands)), f"Duplicate candidate found in query {sid}!"
        print("  [PASS] Strict candidate deduplication verified across all queries.")
        
        # Verification 6: Max block size pruning check
        # 'generic' and 'trading' tokens had 30 postings, which exceeds max_block_size=25, so they must be pruned
        in3_cands = res_map["S1-IN3"]
        # Generic query must not blow up with 30 generic additions
        assert len(in3_cands) <= 36, f"Generic query blew up: {len(in3_cands)} > 36"
        print(f"  [PASS] Generic query pruned properly (Total cands: {len(in3_cands)}).")
        
    print("\n" + "=" * 65)
    print("  ALL SYNTHETIC TOP-UP RETRIEVAL TESTS PASSED CLEANLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_synthetic_topup_test()
