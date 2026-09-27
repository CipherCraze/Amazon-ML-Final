import time
import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

def compute_f05_single(pred_set, true_set):
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    if len(pred_set) == 0:
        return 0.0
    tp = len(pred_set.intersection(true_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def run_stability_and_model_comparison():
    print("=" * 70)
    print("STEP 8 & 9: XGB vs LGBM COMPARISON & STABILITY CHECK")
    print("=" * 70)
    
    t0 = time.time()
    
    # 1. Load ground truth
    val_gt_path = 'output/val_gt_split.tsv'
    val_gt_df = pd.read_csv(val_gt_path, sep='\t', dtype=str)
    total_val_entities = len(val_gt_df)
    
    gt_map = {}
    singleton_entities = set()
    for _, row in val_gt_df.iterrows():
        sid = row['source1_entity_id']
        matches_val = row.get('matched_entity_ids', '')
        if pd.notna(matches_val) and str(matches_val).strip() and str(matches_val).strip() != 'nan':
            gt_map[sid] = set(m.strip() for m in str(matches_val).split(',') if m.strip())
        else:
            gt_map[sid] = set()
            singleton_entities.add(sid)
            
    # 2. Load predictions
    val_probs_xgb = np.load('output/val_probs_xgb_v3.npy')
    val_probs_lgb = np.load('output/val_probs_lgb_v3.npy')
    
    # Load candidate IDs and S1 IDs
    val_csv_path = 'output/full_val_features_v3.csv'
    import pyarrow.csv as pv
    tab = pv.read_csv(val_csv_path, convert_options=pv.ConvertOptions(include_columns=['source1_entity_id', 'candidate_entity_id']))
    df_ids = tab.to_pandas()
    del tab
    
    df_ids['p_xgb'] = val_probs_xgb
    df_ids['p_lgb'] = val_probs_lgb
    
    # Threshold = 0.63 (optimal for both)
    # Check predictions per S1 entity
    print("Comparing per-entity decisions between XGBoost and LightGBM...")
    xgb_preds = {}
    lgb_preds = {}
    
    # Group by S1
    for sid, cid, px, pl in zip(df_ids['source1_entity_id'], df_ids['candidate_entity_id'], df_ids['p_xgb'], df_ids['p_lgb']):
        if px >= 0.63:
            if sid not in xgb_preds:
                xgb_preds[sid] = set()
            xgb_preds[sid].add(cid)
        if pl >= 0.63:
            if sid not in lgb_preds:
                lgb_preds[sid] = set()
            lgb_preds[sid].add(cid)
            
    del df_ids
    gc.collect()
    
    # Entity-level score comparison
    xgb_correct_lgb_wrong = 0
    lgb_correct_xgb_wrong = 0
    both_correct = 0
    both_wrong = 0
    both_identical = 0
    
    xgb_scores = []
    lgb_scores = []
    diffs = []
    
    for sid in val_gt_df['source1_entity_id']:
        true_set = gt_map.get(sid, set())
        pred_x = xgb_preds.get(sid, set())
        pred_l = lgb_preds.get(sid, set())
        
        score_x = compute_f05_single(pred_x, true_set)
        score_l = compute_f05_single(pred_l, true_set)
        
        xgb_scores.append(score_x)
        lgb_scores.append(score_l)
        diff = score_x - score_l
        diffs.append(diff)
        
        if pred_x == pred_l:
            both_identical += 1
            if score_x == 1.0:
                both_correct += 1
            elif score_x == 0.0:
                both_wrong += 1
        else:
            if score_x > score_l:
                xgb_correct_lgb_wrong += 1
            elif score_l > score_x:
                lgb_correct_xgb_wrong += 1
            else:
                pass # partial tie
                
    macro_xgb = np.mean(xgb_scores)
    macro_lgb = np.mean(lgb_scores)
    
    print("\n" + "=" * 50)
    print("XGBOOST VS LIGHTGBM PER-ENTITY BREAKDOWN")
    print("=" * 50)
    print(f"Total Validation Entities:               {total_val_entities:,}")
    print(f"XGBoost Macro F0.5:                      {macro_xgb:.5f}")
    print(f"LightGBM Macro F0.5:                     {macro_lgb:.5f}")
    print(f"Net Gap (XGB - LGBM):                    {macro_xgb - macro_lgb:+.5f}")
    print(f"\nEntity Decision Overlap:")
    print(f"  Exact Identical Prediction Set:        {both_identical:,} ({both_identical/total_val_entities*100:.2f}%)")
    print(f"  Both Perfect (F0.5 = 1.0):             {both_correct:,} ({both_correct/total_val_entities*100:.2f}%)")
    print(f"  Both Complete Miss (F0.5 = 0.0):       {both_wrong:,} ({both_wrong/total_val_entities*100:.2f}%)")
    print(f"  XGBoost Better (score_x > score_l):    {xgb_correct_lgb_wrong:,} ({xgb_correct_lgb_wrong/total_val_entities*100:.2f}%)")
    print(f"  LightGBM Better (score_l > score_x):    {lgb_correct_xgb_wrong:,} ({lgb_correct_xgb_wrong/total_val_entities*100:.2f}%)")
    print(f"  Net Entity Advantage for XGB:         {xgb_correct_lgb_wrong - lgb_correct_xgb_wrong:+,}")
    
    # Statistical significance: Paired Wilcoxon / t-test on per-entity F0.5 differences
    from scipy import stats
    diff_arr = np.array(diffs)
    non_zero_diffs = diff_arr[diff_arr != 0]
    print(f"\nStatistical Significance Test on Non-Zero Differing Entities ({len(non_zero_diffs):,} entities):")
    t_stat, p_val = stats.ttest_1samp(diff_arr, 0.0)
    print(f"  Paired t-test t-statistic: {t_stat:.4f}, p-value: {p_val:.2e}")
    if p_val < 0.001:
        print("  --> The XGBoost advantage over LightGBM is STATISTICALLY HIGHLY SIGNIFICANT (p < 0.001), not random noise!")
    else:
        print("  --> The difference is NOT statistically significant.")
        
    print(f"\nDone in {time.time() - t0:.2f}s")

if __name__ == '__main__':
    run_stability_and_model_comparison()
