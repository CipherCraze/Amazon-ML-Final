#!/usr/bin/env python3
"""
Unit and Smoke Tests for 19-Feature Schema
Amazon ML Challenge 2026 — Experiment 010 (Final)
"""

import os
import sys
import numpy as np

# Ensure src directory is in path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import feature_engineering as fe
import inference as inf
import train_model as tm
from preprocess import (
    normalize_text, get_compact_signature, get_acronym, decompose_address,
    compute_name_indic_translit_sim, compute_postal_match
)


def test_schema_parity():
    print("=" * 60)
    print("1. Testing Schema Parity across Modules (19 Features)")
    print("=" * 60)
    
    assert len(fe.FEATURE_COLS) == 19, f"fe.FEATURE_COLS length {len(fe.FEATURE_COLS)} != 19"
    assert len(inf.FEATURE_COLS) == 19, f"inf.FEATURE_COLS length {len(inf.FEATURE_COLS)} != 19"
    assert len(tm.FEATURE_COLS) == 19, f"tm.FEATURE_COLS length {len(tm.FEATURE_COLS)} != 19"
    
    assert fe.FEATURE_COLS == inf.FEATURE_COLS, "fe.FEATURE_COLS != inf.FEATURE_COLS"
    assert fe.FEATURE_COLS == tm.FEATURE_COLS, "fe.FEATURE_COLS != tm.FEATURE_COLS"
    
    print(f"All 3 modules share the exact same 19 features:")
    for idx, name in enumerate(fe.FEATURE_COLS, start=1):
        print(f"  {idx:2d}. {name} (dtype: float32)")
        assert name in tm.COL_DTYPES, f"{name} missing from train_model.COL_DTYPES"
        assert tm.COL_DTYPES[name] == np.float32, f"{name} dtype is not float32"
        
    print(" Schema parity verified successfully!\n")


def test_feature_computation():
    print("=" * 60)
    print("2. Testing Feature Computation & Value Correctness")
    print("=" * 60)
    
    # Synthetic records with Indic name and postal codes
    s1_name = "श्री गणेश ट्रेडर्स"  # "Shri Ganesh Traders" in Devanagari
    s1_addr = "Plot 42, MG Road, Bangalore 560001"
    s1_country = "India"
    
    s1_norm = normalize_text(s1_name)
    s1_comp = get_compact_signature(s1_name)
    s1_acro = get_acronym(s1_name)
    s1_c_addr, s1_num, s1_street, s1_cs, s1_pc = decompose_address(s1_addr)
    s1_d = fe.extract_digits(f"{s1_norm} {s1_addr}")
    s1_rec = (s1_norm, s1_comp, s1_acro, s1_c_addr, s1_num, s1_street, s1_cs, s1_d, s1_country, s1_name, s1_pc)
    
    cand_name = "Shree Ganesh Traders Pvt Ltd"
    cand_addr = "No 42 Mahatma Gandhi Rd, Bengaluru, Karnataka 560001"
    cand_country = "India"
    
    c_norm = normalize_text(cand_name)
    c_comp = get_compact_signature(cand_name)
    c_acro = get_acronym(cand_name)
    c_addr, c_num, c_street, c_cs, c_pc = decompose_address(cand_addr)
    c_d = fe.extract_digits(f"{c_norm} {cand_addr}")
    c_rec = (c_norm, c_comp, c_acro, c_addr, c_num, c_street, c_cs, c_d, cand_country, cand_name, c_pc)
    
    # Test feature vector generation for ranks 1, 30, and top-up ranks 31, 36
    for test_rank in [1, 15, 30, 31, 36]:
        test_score = 0.8954 if test_rank <= 30 else 0.0
        f_fe = fe.fast_19_features(s1_rec, c_rec, "S2-12345", dense_score=test_score, rank_idx=test_rank)
        f_inf = inf.fast_19_features(s1_rec, c_rec, "S2-12345", dense_score=test_score, rank_idx=test_rank)
        
        assert len(f_fe) == 19, f"Feature count {len(f_fe)} != 19"
        assert len(f_inf) == 19, f"Inference feature count {len(f_inf)} != 19"
        assert f_fe == f_inf, f"fe output != inf output for rank {test_rank}"
        
        # Verify candidate_rank at index 15
        assert f_fe[15] == float(test_rank), f"Candidate rank mismatch: {f_fe[15]} vs {test_rank}"
        # Verify dense_cosine_sim at index 16
        assert abs(f_fe[16] - test_score) < 1e-6, f"Dense score mismatch: {f_fe[16]} vs {test_score}"
        
        # Verify Indic transliteration sim at index 17
        indic_sim = f_fe[17]
        assert indic_sim > 0.60, f"Indic transliteration similarity too low: {indic_sim}"
        
        # Verify postal_match at index 18 (exact match -> 1.0)
        postal_sim = f_fe[18]
        assert postal_sim == 1.0, f"Postal match should be 1.0, got: {postal_sim}"
        
    print(f"Sample 19-feature vector generated (Rank 31 Top-Up Candidate):")
    for name, val in zip(fe.FEATURE_COLS, f_fe):
        print(f"  {name:>25s}: {val:.4f}")
        
    print(" Feature computation verified successfully!\n")


def test_indic_and_postal_functions():
    print("=" * 60)
    print("3. Testing Indic Transliteration and Postal Matching")
    print("=" * 60)
    
    # 1. Indic transliteration
    sim_ganesh = compute_name_indic_translit_sim("गणेश ट्रेडर्स", "Ganesh Traders")
    print(f"  Transliteration Sim (Devanagari Ganesh vs Latin Ganesh): {sim_ganesh:.4f}")
    assert sim_ganesh >= 0.70, f"Expected transliteration sim >= 0.70, got {sim_ganesh}"
    
    sim_reliance = compute_name_indic_translit_sim("रिलायंस इंडस्ट्रीज", "Reliance Industries Limited")
    print(f"  Transliteration Sim (Devanagari Reliance vs Latin Reliance): {sim_reliance:.4f}")
    assert sim_reliance >= 0.50, f"Expected transliteration sim >= 0.50, got {sim_reliance}"
    
    # 2. Postal matching
    p_exact = compute_postal_match("123 Main St, New Delhi 110001", "Room 4, Connaught Place, New Delhi 110001")
    p_prefix = compute_postal_match("Sector 15, Rohini 110085", "Sector 3, Rohini 110089")
    p_diff = compute_postal_match("Mumbai 400001", "Kolkata 700001")
    p_missing = compute_postal_match("MG Road, Bangalore", "Koramangala, Bangalore")
    
    print(f"  Postal Exact   (110001 vs 110001): {p_exact} (Expected 1.0)")
    print(f"  Postal Prefix  (110085 vs 110089): {p_prefix} (Expected 0.5)")
    print(f"  Postal Diff    (400001 vs 700001): {p_diff} (Expected 0.0)")
    print(f"  Postal Missing (None   vs None  ): {p_missing} (Expected 0.2)")
    
    assert p_exact == 1.0
    assert p_prefix == 0.5
    assert p_diff == 0.0
    assert p_missing == 0.2
    print(" Indic and postal logic verified successfully!\n")


def test_rank_bounds_and_assertions():
    print("=" * 60)
    print("4. Testing Bounds Assertions (Rank 1..60 & Dense Scores)")
    print("=" * 60)
    
    empty_rec = ("", "", "", "", "", "", "", "", "", "", "")
    
    # Valid bounds: 1 to 60
    for valid_rank in [1, 30, 31, 36, 60]:
        fe.fast_19_features(empty_rec, empty_rec, "cand_1", dense_score=0.5, rank_idx=valid_rank)
        inf.fast_19_features(empty_rec, empty_rec, "cand_1", dense_score=0.5, rank_idx=valid_rank)
    
    # Invalid bounds: 0 and 61 must raise AssertionError
    rank_zero_failed = False
    try:
        fe.fast_19_features(empty_rec, empty_rec, "cand_1", dense_score=0.5, rank_idx=0)
    except AssertionError:
        rank_zero_failed = True
    assert rank_zero_failed, "Rank 0 did not trigger AssertionError!"
    
    rank_61_failed = False
    try:
        fe.fast_19_features(empty_rec, empty_rec, "cand_1", dense_score=0.5, rank_idx=61)
    except AssertionError:
        rank_61_failed = True
    assert rank_61_failed, "Rank 61 did not trigger AssertionError!"
    
    print(" Rank bound assertions verified (strictly 1..60 enforced)!\n")


def test_model_data_structures():
    print("=" * 60)
    print("5. Testing Tree Model Compatibility (XGBoost & LightGBM)")
    print("=" * 60)
    
    import lightgbm as lgb
    import xgboost as xgb
    
    # 20 samples with 19 features
    rng = np.random.RandomState(42)
    X = rng.rand(20, 19).astype(np.float32)
    # Integer ranks 1..60 in column 15
    X[:, 15] = rng.randint(1, 61, size=20).astype(np.float32)
    y = rng.randint(0, 2, size=20).astype(np.int8)
    
    # XGBoost DMatrix with 19 feature names
    dmat = xgb.DMatrix(X, label=y, feature_names=fe.FEATURE_COLS)
    assert dmat.num_col() == 19, f"DMatrix cols {dmat.num_col()} != 19"
    assert dmat.num_row() == 20, f"DMatrix rows {dmat.num_row()} != 20"
    
    # LightGBM Dataset with 19 feature names
    ds = lgb.Dataset(X, label=y, feature_name=fe.FEATURE_COLS, free_raw_data=False)
    ds.construct()
    assert len(ds.feature_name) == 19, f"LGBM dataset cols {len(ds.feature_name)} != 19"
    
    print(" Both XGBoost and LightGBM construct cleanly with the 19-feature schema!\n")


if __name__ == "__main__":
    test_schema_parity()
    test_feature_computation()
    test_indic_and_postal_functions()
    test_rank_bounds_and_assertions()
    test_model_data_structures()
    print("=" * 60)
    print("ALL 19-FEATURE SMOKE TESTS PASSED CLEANLY!")
    print("=" * 60)
