"""
Unit tests for Experiment 004: Address-Aware Features and Controlled Address Dropout.
"""

import os
import sys
import numpy as np

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import feature_engineering as fe
import train_model as tm
from preprocess import normalize_text, get_compact_signature, get_acronym, decompose_address


def test_18_feature_extraction():
    print("=" * 60)
    print("1. Testing 18-Feature Extraction (Exp 004: JW + addr_both_present)")
    print("=" * 60)
    
    assert len(fe.FEATURE_COLS_18) == 18, f"FEATURE_COLS_18 length != 18: {len(fe.FEATURE_COLS_18)}"
    assert fe.FEATURE_COLS_18[-2] == 'name_jaro_winkler'
    assert fe.FEATURE_COLS_18[-1] == 'addr_both_present'
    
    # Case 1: Both have addresses
    s1_rec = ("acme global corp", "acmeglobal", "agcorp", "123 main st seattle", "123", "main st", "seattle", "123", "US")
    cand_rec = ("acme global corporation", "acmeglobal", "agcorp", "123 main street seattle", "123", "main st", "seattle", "123", "US")
    
    f18 = fe.fast_18_features(s1_rec, cand_rec, "S2-001", rank_idx=1)
    assert len(f18) == 18, f"Feature count != 18: {len(f18)}"
    name_jw = f18[16]
    addr_both = f18[17]
    assert 0.90 <= name_jw <= 1.0, f"Expected high Jaro-Winkler, got {name_jw}"
    assert addr_both == 1.0, f"Expected addr_both_present == 1.0, got {addr_both}"
    assert f18[6] == 0.0, f"Expected is_addr_missing == 0.0, got {f18[6]}"
    
    # Case 2: Candidate has missing address (e.g. Straight Edge Grill)
    cand_no_addr = ("acme global corporation", "acmeglobal", "agcorp", "", "", "", "", "", "US")
    f18_no_addr = fe.fast_18_features(s1_rec, cand_no_addr, "S2-002", rank_idx=1)
    assert f18_no_addr[16] == name_jw, "Name similarity should be identical regardless of address"
    assert f18_no_addr[17] == 0.0, f"Expected addr_both_present == 0.0, got {f18_no_addr[17]}"
    assert f18_no_addr[6] == 1.0, f"Expected is_addr_missing == 1.0, got {f18_no_addr[6]}"
    
    # Case 3: Conflicting address
    cand_conf_addr = ("acme global corporation", "acmeglobal", "agcorp", "999 south blvd dallas", "999", "south blvd", "dallas", "999", "US")
    f18_conf = fe.fast_18_features(s1_rec, cand_conf_addr, "S2-003", rank_idx=1)
    assert f18_conf[17] == 1.0, "Conflicting address has BOTH present"
    assert f18_conf[6] == 0.0, "Conflicting address is NOT missing"
    assert f18_conf[10] < 0.50, f"Conflicting address has lower token sort similarity: {f18_conf[10]}"
    
    print("  18-Feature extraction and address validity semantics verified successfully!\n")


def test_address_dropout_augmentation():
    print("=" * 60)
    print("2. Testing Controlled Address Dropout Augmentation")
    print("=" * 60)
    
    num_samples = 1000
    rng = np.random.RandomState(42)
    X = rng.rand(num_samples, 18).astype(np.float32)
    y = np.zeros(num_samples, dtype=np.int8)
    
    # Half positive, half negative
    y[:500] = 1
    # For positive samples, set is_addr_missing=0.0 and addr_both_present=1.0 initially
    is_missing_idx = fe.FEATURE_COLS_18.index('is_addr_missing')
    both_present_idx = fe.FEATURE_COLS_18.index('addr_both_present')
    addr_sort_idx = fe.FEATURE_COLS_18.index('addr_token_sort')
    
    X[:500, is_missing_idx] = 0.0
    X[:500, both_present_idx] = 1.0
    X[:500, addr_sort_idx] = 0.95
    
    # Test rate 0.00 (no dropout)
    X_d0 = tm.apply_address_dropout(X, y, fe.FEATURE_COLS_18, dropout_rate=0.0, seed=42)
    assert np.array_equal(X, X_d0), "Dropout 0.00 modified the matrix!"
    
    # Test rates 0.05, 0.10, 0.15
    for rate in [0.05, 0.10, 0.15]:
        X_drop = tm.apply_address_dropout(X, y, fe.FEATURE_COLS_18, dropout_rate=rate, seed=42)
        
        # Negatives must NEVER be touched
        assert np.array_equal(X[500:], X_drop[500:]), f"Negatives were mutated at rate {rate}!"
        
        # Check dropped samples
        dropped_mask = (X_drop[:500, is_missing_idx] == 1.0)
        num_dropped = dropped_mask.sum()
        expected_dropped = int(500 * rate)
        assert num_dropped == expected_dropped, f"Expected {expected_dropped} dropped, got {num_dropped}"
        
        # Verify dropped features
        dropped_rows = X_drop[:500][dropped_mask]
        assert np.all(dropped_rows[:, is_missing_idx] == 1.0), "is_addr_missing not set to 1.0"
        assert np.all(dropped_rows[:, both_present_idx] == 0.0), "addr_both_present not set to 0.0"
        assert np.all(dropped_rows[:, addr_sort_idx] == 0.0), "addr_token_sort not zeroed"
        
        # Determinism check: Same seed produces identical result
        X_drop2 = tm.apply_address_dropout(X, y, fe.FEATURE_COLS_18, dropout_rate=rate, seed=42)
        assert np.array_equal(X_drop, X_drop2), "Address dropout was non-deterministic under same seed!"
        
    print("  Address dropout augmentation verified across all rates (0.00, 0.05, 0.10, 0.15)!\n")


if __name__ == "__main__":
    test_18_feature_extraction()
    test_address_dropout_augmentation()
    print("=" * 60)
    print("ALL EXPERIMENT 004 TESTS PASSED CLEANLY!")
    print("=" * 60)
