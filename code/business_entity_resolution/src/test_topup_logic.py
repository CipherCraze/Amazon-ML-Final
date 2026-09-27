import os
import re
import sys
import time
import array
from collections import defaultdict
import pandas as pd
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

def extract_indian_keys(name, addr):
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

def test_on_validation_sample():
    print("Testing Indian top-up retrieval on validation sample...")
    val_gt_path = 'output/val_gt_split.tsv'
    gt_df = pd.read_csv(val_gt_path, sep='\t', dtype=str)
    gt_map = {}
    for _, row in gt_df.iterrows():
        sid = row['source1_entity_id']
        m = row.get('matched_entity_ids', '')
        if pd.notna(m) and str(m).strip() and str(m).strip() != 'nan':
            gt_map[sid] = set(str(m).strip().split(','))
            
    # Sample 1000 Indian queries with GT
    s1_df = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t', nrows=50000, dtype=str)
    ind_s1 = s1_df[(s1_df['country'] == 'India') & (s1_df['entity_id'].isin(gt_map))].head(1000)
    print(f"Sampled {len(ind_s1)} Indian S1 queries.")
    
    # Check sample keys
    key_counts = 0
    for _, row in ind_s1.iterrows():
        keys = extract_indian_keys(row['business_name'], row['business_address'])
        key_counts += len(keys)
    print(f"Average keys per Indian S1 query: {key_counts / len(ind_s1):.2f}")

if __name__ == '__main__':
    test_on_validation_sample()
