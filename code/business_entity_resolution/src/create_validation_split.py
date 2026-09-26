import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

def create_splits(data_dir, output_dir, val_size=0.2, random_state=42):
    print("Loading datasets for validation split...")
    gt_candidates = [
        os.path.join(data_dir, "train", "train_ground_truth.tsv"),
        os.path.join(data_dir, "train_ground_truth.tsv"),
    ]
    gt_path = next((p for p in gt_candidates if os.path.exists(p)), gt_candidates[0])
    
    s1_candidates = [
        os.path.join(data_dir, "train", "train_source1.tsv"),
        os.path.join(data_dir, "train_source1.tsv"),
    ]
    s1_path = next((p for p in s1_candidates if os.path.exists(p)), s1_candidates[0])
    
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"train_ground_truth.tsv not found under: {data_dir}")
    if not os.path.exists(s1_path):
        raise FileNotFoundError(f"train_source1.tsv not found under: {data_dir}")
        
    gt_df = pd.read_csv(gt_path, sep="\t")
    s1_df = pd.read_csv(s1_path, sep="\t")
    
    # Merge to get country for stratification
    merged = pd.merge(gt_df, s1_df[['entity_id', 'country']], left_on='source1_entity_id', right_on='entity_id', how='left')
    
    # Determine if it has matches
    merged['has_match'] = merged['matched_entity_ids'].notna() & (merged['matched_entity_ids'] != '')
    
    # Create stratification column
    merged['stratify_col'] = merged['country'].astype(str) + "_" + merged['has_match'].astype(str)
    
    print(f"Total entities: {len(merged)}")
    print("Stratification distribution:")
    print(merged['stratify_col'].value_counts())
    
    # Stratified split
    train_split, val_split = train_test_split(
        merged, 
        test_size=val_size, 
        random_state=random_state, 
        stratify=merged['stratify_col']
    )
    
    print(f"\nTrain size: {len(train_split)}, Val size: {len(val_split)}")
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    # Keep only the original columns
    train_gt = train_split[['source1_entity_id', 'matched_entity_ids']]
    val_gt = val_split[['source1_entity_id', 'matched_entity_ids']]
    
    train_out = os.path.join(output_dir, "train_gt_split.tsv")
    val_out = os.path.join(output_dir, "val_gt_split.tsv")
    train_gt.to_csv(train_out, sep="\t", index=False)
    val_gt.to_csv(val_out, sep="\t", index=False)
    print(f"Saved splits to {output_dir}")
    return train_out, val_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create stratified validation split")
    parser.add_argument("--data-dir", default=None, help="Path to dataset directory containing train data")
    parser.add_argument("--output-dir", default=None, help="Path to output directory for splits")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if args.data_dir:
        data_dir = args.data_dir
    else:
        cand_dirs = [
            os.path.join(base_dir, "dataset"),
            os.path.join(base_dir, "student_resource", "dataset"),
            os.path.join(base_dir, "6ab10eb3b23ba_student_resource", "student_resource", "dataset"),
        ]
        data_dir = next((d for d in cand_dirs if os.path.exists(d)), cand_dirs[0])

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(base_dir, "output")

    create_splits(data_dir, output_dir)
