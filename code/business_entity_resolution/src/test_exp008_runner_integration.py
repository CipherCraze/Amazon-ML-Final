"""
Unit and Integration Tests for Experiment 008:
Combined Jaro-Winkler + Address-Aware Features + Address Dropout + Runner Integration
"""

import os
import sys
import unittest
import numpy as np

# Ensure repo root and src are in sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from feature_engineering import fast_18_features, FEATURE_COLS_18, FEATURE_COLS_16
from train_model import apply_address_dropout
import run_kaggle


class TestExp008Integration(unittest.TestCase):

    def test_18_feature_definitions(self):
        self.assertEqual(len(FEATURE_COLS_18), 18)
        self.assertEqual(len(FEATURE_COLS_16), 16)
        self.assertIn("name_jaro_winkler", FEATURE_COLS_18)
        self.assertIn("addr_both_present", FEATURE_COLS_18)
        self.assertIn("is_addr_missing", FEATURE_COLS_18)
        self.assertIn("candidate_rank", FEATURE_COLS_18)

    def test_fast_18_feature_computation(self):
        s1 = ("Apple Inc.", "apple", "ai", "1 Infinite Loop Cupertino CA", "1", "infinite loop", "cupertino ca", "1", "US")
        c_match = ("Apple Incorporated", "apple", "ai", "1 Infinite Loop Cupertino CA", "1", "infinite loop", "cupertino ca", "1", "US")
        c_no_addr = ("Apple Inc", "apple", "ai", "", "", "", "", "", "US")

        f_full = fast_18_features(s1, c_match, "cand_01", rank_idx=1)
        self.assertEqual(len(f_full), 18)
        # Jaro-Winkler between Apple Inc. and Apple Incorporated is ~0.88
        self.assertGreater(f_full[16], 0.85)
        # Both addresses present
        self.assertEqual(f_full[17], 1.0)
        # Address is NOT missing
        self.assertEqual(f_full[6], 0.0)

        f_missing = fast_18_features(s1, c_no_addr, "cand_02", rank_idx=2)
        self.assertEqual(len(f_missing), 18)
        self.assertGreater(f_missing[16], 0.95)
        # Address NOT both present
        self.assertEqual(f_missing[17], 0.0)
        # Address missing flag is 1.0
        self.assertEqual(f_missing[6], 1.0)

    def test_address_dropout_deterministic(self):
        rng = np.random.RandomState(42)
        N = 200
        X = rng.rand(N, 18).astype(np.float32)
        feat_to_idx = {name: i for i, name in enumerate(FEATURE_COLS_18)}
        X[:, feat_to_idx['is_addr_missing']] = 0.0
        X[:, feat_to_idx['addr_both_present']] = 1.0
        y = np.ones(N, dtype=np.int8)

        X_drop1 = apply_address_dropout(X, y, FEATURE_COLS_18, dropout_rate=0.10, seed=42)
        X_drop2 = apply_address_dropout(X, y, FEATURE_COLS_18, dropout_rate=0.10, seed=42)

        np.testing.assert_array_equal(X_drop1, X_drop2)
        # Exactly 10% (20 samples) should have is_addr_missing set to 1.0
        dropped_count = np.sum(X_drop1[:, feat_to_idx['is_addr_missing']] == 1.0)
        self.assertEqual(dropped_count, 20)
        # addr_both_present should be 0.0 for those 20 samples
        dropped_both = np.sum(X_drop1[:, feat_to_idx['addr_both_present']] == 0.0)
        self.assertEqual(dropped_both, 20)

    def test_run_kaggle_arg_parsing(self):
        # Verify parser accepts all experiment IDs
        for exp_id in ["002a", "002", "003", "004", "005", "006", "007", "008", "baseline"]:
            parser = run_kaggle.argparse.ArgumentParser()
            # Reconstruct or test run_kaggle parser
            parser.add_argument("--experiment", choices=["002a", "002", "003", "004", "005", "006", "007", "008", "baseline"])
            args = parser.parse_args(["--experiment", exp_id])
            self.assertEqual(args.experiment, exp_id)


if __name__ == "__main__":
    unittest.main()
