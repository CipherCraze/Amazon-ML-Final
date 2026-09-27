import time
import os
import numpy as np
import pandas as pd

def run_test_prediction_diagnostics():
    print("=" * 70)
    print("STEP 3: TEST PREDICTION STATISTICS BY COUNTRY")
    print("=" * 70)
    
    t0 = time.time()
    
    test_s1_path = 'student_resource/dataset/test/test_source1.tsv'
    matching_results_path = 'output/matching_results.tsv'
    
    print(f"Loading {test_s1_path}...")
    s1_df = pd.read_csv(test_s1_path, sep='\t', usecols=['entity_id', 'country'], dtype=str)
    print(f"Loaded {len(s1_df):,} test S1 entities.")
    
    print(f"Loading {matching_results_path}...")
    matches_df = pd.read_csv(matching_results_path, sep='\t', dtype=str)
    print(f"Loaded {len(matches_df):,} test prediction rows.")
    
    # Merge on entity_id / source1_entity_id
    merged = pd.merge(s1_df, matches_df, left_on='entity_id', right_on='source1_entity_id', how='left')
    del s1_df, matches_df
    
    # Analyze match counts
    # matched_entity_ids can be NaN, empty, or comma-separated
    def count_matches(val):
        if pd.isna(val) or not str(val).strip() or str(val).strip() == 'nan':
            return 0
        return len(str(val).split(','))
        
    merged['match_count'] = merged['matched_entity_ids'].apply(count_matches)
    
    # Print overall and per country
    countries = list(merged['country'].unique())
    # Sort with US, India, France first
    priority = ['US', 'India', 'France']
    sorted_countries = [c for c in priority if c in countries] + [c for c in countries if c not in priority]
    
    header = f"{'Country':<10} | {'S1 Count':>10} | {'% of Test':>9} | {'Empty/Sing %':>12} | {'Multi %':>9} | {'Avg Matches':>11} | {'Med':>4} | {'Max':>4}"
    print("\n" + header)
    print("-" * len(header))
    
    for c in sorted_countries:
        subset = merged[merged['country'] == c] if pd.notna(c) else merged[merged['country'].isna()]
        cnt = len(subset)
        pct_test = cnt / len(merged) * 100
        empty_cnt = (subset['match_count'] == 0).sum()
        empty_rate = empty_cnt / cnt * 100
        single_cnt = (subset['match_count'] == 1).sum()
        multi_cnt = (subset['match_count'] > 1).sum()
        multi_rate = multi_cnt / cnt * 100
        avg_m = subset['match_count'].mean()
        med_m = subset['match_count'].median()
        max_m = subset['match_count'].max()
        c_label = str(c) if pd.notna(c) else "MISSING"
        print(f"{c_label:<10} | {cnt:>10,} | {pct_test:>8.2f}% | {empty_rate:>11.2f}% | {multi_rate:>8.2f}% | {avg_m:>11.2f} | {int(med_m):>4} | {max_m:>4}")
        
    # TOTAL
    tot_cnt = len(merged)
    tot_empty = (merged['match_count'] == 0).sum()
    tot_multi = (merged['match_count'] > 1).sum()
    tot_avg = merged['match_count'].mean()
    tot_med = merged['match_count'].median()
    tot_max = merged['match_count'].max()
    print("-" * len(header))
    print(f"{'OVERALL':<10} | {tot_cnt:>10,} | {'100.00%':>9} | {tot_empty/tot_cnt*100:>11.2f}% | {tot_multi/tot_cnt*100:>8.2f}% | {tot_avg:>11.2f} | {int(tot_med):>4} | {tot_max:>4}")
    
    # Detailed match count frequency distribution
    print("\n" + "=" * 50)
    print("PREDICTED MATCH COUNT FREQUENCY PER COUNTRY")
    print("=" * 50)
    for c in sorted_countries:
        subset = merged[merged['country'] == c]
        vc = subset['match_count'].value_counts()
        print(f"\n--- {c} (Total: {len(subset):,}) ---")
        for k in range(0, 6):
            c_k = vc.get(k, 0)
            print(f"  Match count {k}: {c_k:>8,} ({c_k/len(subset)*100:>6.2f}%)")
        c_gt5 = (subset['match_count'] > 5).sum()
        print(f"  Match count >5: {c_gt5:>8,} ({c_gt5/len(subset)*100:>6.2f}%)")
        
    print(f"\nDone in {time.time() - t0:.2f}s")

if __name__ == '__main__':
    run_test_prediction_diagnostics()
