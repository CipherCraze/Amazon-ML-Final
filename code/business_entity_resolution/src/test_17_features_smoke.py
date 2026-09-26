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
from preprocess import normalize_text, get_compact_signature, get_acronym, decompose_address


def test_schema_parity():
    print("=" * 60)
    print("1. Testing Schema Parity across Modules")
    print("=" * 60)
    
    # 1. Feature list equality
    assert len(fe.FEATURE_COLS) == 17, f"fe.FEATURE_COLS length {len(fe.FEATURE_COLS)} != 17"
    assert len(inf.FEATURE_COLS) == 17, f"inf.FEATURE_COLS length {len(inf.FEATURE_COLS)} != 17"
    assert len(tm.FEATURE_COLS) == 17, f"tm.FEATURE_COLS length {len(tm.FEATURE_COLS)} != 17"
    
    assert fe.FEATURE_COLS == inf.FEATURE_COLS, "fe.FEATURE_COLS != inf.FEATURE_COLS"
    assert fe.FEATURE_COLS == tm.FEATURE_COLS, "fe.FEATURE_COLS != tm.FEATURE_COLS"
    
    print(f"All 3 modules share the exact same 17 features:")
    for idx, name in enumerate(fe.FEATURE_COLS, start=1):
        print(f"  {idx:2d}. {name} (dtype: float32)")
        assert name in tm.COL_DTYPES, f"{name} missing from train_model.COL_DTYPES"
        assert tm.COL_DTYPES[name] == np.float32, f"{name} dtype is not float32"
        
    print(" Schema parity verified successfully!\n")


def test_feature_computation():
    print("=" * 60)
    print("2. Testing Feature Computation & Value Correctness")
    print("=" * 60)
    
    # Synthetic records
    s1_name = "Acme Global Solutions Inc"
    s1_addr = "123 Main Street, Suite 400, Seattle, WA 98101"
    s1_country = "US"
    
    s1_norm = normalize_text(s1_name)
    s1_comp = get_compact_signature(s1_name)
    s1_acro = get_acronym(s1_name)
    s1_c_addr, s1_num, s1_street, s1_cs, s1_pc = decompose_address(s1_addr)
    s1_d = fe.extract_digits(f"{s1_norm} {s1_addr}")
    s1_rec = (s1_norm, s1_comp, s1_acro, s1_c_addr, s1_num, s1_street, s1_cs, s1_d, s1_country)
    
    cand_name = "Acme Global Solutions"
    cand_addr = "123 Main St, Seattle, WA 98101"
    cand_country = "US"
    
    c_norm = normalize_text(cand_name)
    c_comp = get_compact_signature(cand_name)
    c_acro = get_acronym(cand_name)
    c_addr, c_num, c_street, c_cs, c_pc = decompose_address(cand_addr)
    c_d = fe.extract_digits(f"{c_norm} {cand_addr}")
    c_rec = (c_norm, c_comp, c_acro, c_addr, c_num, c_street, c_cs, c_d, cand_country)
    
    # Test feature vector generation for ranks 1 to 30
    for test_rank in [1, 2, 15, 30]:
        test_score = 0.8954
        f_fe = fe.fast_17_features(s1_rec, c_rec, "S2-12345", dense_score=test_score, rank_idx=test_rank)
        f_inf = inf.fast_17_features(s1_rec, c_rec, "S2-12345", dense_score=test_score, rank_idx=test_rank)
        
        assert len(f_fe) == 17, f"Feature count {len(f_fe)} != 17"
        assert len(f_inf) == 17, f"Inference feature count {len(f_inf)} != 17"
        assert f_fe == f_inf, f"fe output != inf output for rank {test_rank}"
        
        # Verify dense_cosine_sim at index 15
        assert abs(f_fe[15] - test_score) < 1e-6, f"Dense score mismatch: {f_fe[15]} vs {test_score}"
        # Verify candidate_rank at index 16
        assert f_fe[16] == float(test_rank), f"Candidate rank mismatch: {f_fe[16]} vs {test_rank}"
        
        # Verify backward compatibility with fast_15_features
        f15_fe = fe.fast_15_features(s1_rec, c_rec, "S2-12345")
        assert len(f15_fe) == 15, f"fast_15_features length {len(f15_fe)} != 15"
        assert f15_fe == f_fe[:15], "First 15 features differ between fast_15 and fast_17"
        
    print(f"Sample 17-feature vector generated:")
    for name, val in zip(fe.FEATURE_COLS, f_fe):
        print(f"  {name:>20s}: {val:.4f}")
        
    print(" Feature computation verified successfully!\n")


def test_rank_bounds_and_assertions():
    print("=" * 60)
    print("3. Testing Bounds Assertions (Rank 1..30 & Dense Scores)")
    print("=" * 60)
    
    empty_rec = ("", "", "", "", "", "", "", "", "")
    
    # Valid bounds
    fe.fast_17_features(empty_rec, empty_rec, "cand_1", dense_score=0.5, rank_idx=1)
    fe.fast_17_features(empty_rec, empty_rec, "cand_1", dense_score=0.5, rank_idx=30)
    
    # Invalid bounds must raise AssertionError
    rank_zero_failed = False
    try:
        fe.fast_17_features(empty_rec, empty_rec, "cand_1", dense_score=0.5, rank_idx=0)
    except AssertionError:
        rank_zero_failed = True
    assert rank_zero_failed, "Rank 0 did not trigger AssertionError!"
    
    rank_31_failed = False
    try:
        fe.fast_17_features(empty_rec, empty_rec, "cand_1", dense_score=0.5, rank_idx=31)
    except AssertionError:
        rank_31_failed = True
    assert rank_31_failed, "Rank 31 did not trigger AssertionError!"
    
    print(" Rank bound assertions verified (strictly 1..30 enforced)!\n")


def test_model_data_structures():
    print("=" * 60)
    print("4. Testing Tree Model Compatibility (XGBoost & LightGBM)")
    print("=" * 60)
    
    import lightgbm as lgb
    import xgboost as xgb
    
    # 20 samples with 17 features
    rng = np.random.RandomState(42)
    X = rng.rand(20, 17).astype(np.float32)
    # Integer ranks 1..30 in column 16
    X[:, 16] = rng.randint(1, 31, size=20).astype(np.float32)
    y = rng.randint(0, 2, size=20).astype(np.int8)
    
    # XGBoost DMatrix with 17 feature names
    dmat = xgb.DMatrix(X, label=y, feature_names=fe.FEATURE_COLS)
    assert dmat.num_col() == 17, f"DMatrix cols {dmat.num_col()} != 17"
    assert dmat.num_row() == 20, f"DMatrix rows {dmat.num_row()} != 20"
    
    # LightGBM Dataset with 17 feature names
    ds = lgb.Dataset(X, label=y, feature_name=fe.FEATURE_COLS, free_raw_data=False)
    ds.construct()
    assert len(ds.feature_name) == 17, f"LGBM dataset cols {len(ds.feature_name)} != 17"
    
    print(" Both XGBoost and LightGBM construct cleanly with the 17-feature schema!\n")


if __name__ == "__main__":
    test_schema_parity()
    test_feature_computation()
    test_rank_bounds_and_assertions()
    test_model_data_structures()
    print("=" * 60)
    print("ALL 17-FEATURE SMOKE TESTS PASSED CLEANLY!")
    print("=" * 60)
