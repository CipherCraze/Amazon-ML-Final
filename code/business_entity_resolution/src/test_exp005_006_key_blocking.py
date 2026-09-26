"""
Unit tests for Experiment 005 (Free-Form Address Blocking) and Experiment 006 (Compound Brand Blocking).
"""

import os
import sys

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from expand_candidates_key_blocking import (
    extract_freeform_address_keys,
    extract_compound_brand_tokens,
    BUSINESS_STOPWORDS
)


def test_freeform_address_key_extraction():
    print("=" * 60)
    print("1. Testing Free-Form Address Key Extraction (Exp 005)")
    print("=" * 60)
    
    # House number at the end
    addr1 = "Alliance, OH, 71 Oxford St"
    keys1 = extract_freeform_address_keys(addr1, country="US")
    assert any("71_oxford" in k for k in keys1), f"Failed to extract 71_oxford from '{addr1}'. Got: {keys1}"
    
    # House number at the beginning
    addr2 = "71 Oxford St, Alliance, OH"
    keys2 = extract_freeform_address_keys(addr2, country="US")
    assert any("71_oxford" in k for k in keys2), f"Failed to extract 71_oxford from '{addr2}'. Got: {keys2}"
    
    # Both permutations share the exact same key!
    common_keys = set(keys1).intersection(set(keys2))
    assert len(common_keys) > 0, f"Permutations did not produce common key! keys1: {keys1}, keys2: {keys2}"
    print(f"  Shared key extracted from both permutations: {common_keys}")
    
    # Suite prefix with house number in middle
    addr3 = "Suite 400, 123 Main Road"
    keys3 = extract_freeform_address_keys(addr3, country="US")
    assert any("123_main" in k for k in keys3), f"Failed to extract 123_main from '{addr3}'. Got: {keys3}"
    
    # French road format
    addr4 = "14 Rue de la Paix, 75001 Paris"
    keys4 = extract_freeform_address_keys(addr4, country="FR")
    assert any("14_rue" in k for k in keys4), f"Failed to extract 14_rue from '{addr4}'. Got: {keys4}"
    
    # Empty / Missing address
    assert extract_freeform_address_keys("", country="US") == []
    assert extract_freeform_address_keys(None, country="US") == []
    
    print("  Free-form address key extraction verified successfully!\n")


def test_compound_brand_token_extraction():
    print("=" * 60)
    print("2. Testing Compound Brand Token Extraction (Exp 006)")
    print("=" * 60)
    
    name1 = "Thompson and Delgado Cafe"
    keys1 = extract_compound_brand_tokens(name1, country="US")
    assert any("thompson_delgado" in k for k in keys1), f"Failed to extract thompson_delgado from '{name1}'. Got: {keys1}"
    
    name2 = "Thompson Thompson & Delgado Services"
    keys2 = extract_compound_brand_tokens(name2, country="US")
    assert any("thompson_delgado" in k for k in keys2), f"Failed to extract thompson_delgado from '{name2}'. Got: {keys2}"
    
    # Both compound names share the exact same blocking key!
    common_keys = set(keys1).intersection(set(keys2))
    assert len(common_keys) > 0, f"Compound names did not produce common key! keys1: {keys1}, keys2: {keys2}"
    print(f"  Shared compound key extracted from both variants: {common_keys}")
    
    # Verify that generic stopwords are never standalone keys
    generic_name = "Global Services Limited Company"
    keys_generic = extract_compound_brand_tokens(generic_name, country="US")
    assert len(keys_generic) == 0, f"Generic stopwords should not produce keys! Got: {keys_generic}"
    
    print("  Compound brand token extraction verified successfully!\n")


if __name__ == "__main__":
    test_freeform_address_key_extraction()
    test_compound_brand_token_extraction()
    print("=" * 60)
    print("ALL EXPERIMENT 005 & 006 TESTS PASSED CLEANLY!")
    print("=" * 60)
