"""
Unit tests for Experiment 007: Improved Indic Script Normalization.
Tests Latin, accented, Indic, mixed, missing values, and transliteration availability.
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import preprocess as pp


def test_indic_script_normalization():
    print("=" * 60)
    print("1. Testing Indic Script Transliteration & Normalization (Exp 007)")
    print("=" * 60)
    
    # 1. Indic Hindi examples requested by user
    # 'स्काई' -> sky
    sky_norm = pp.normalize_text("स्काई")
    print(f"  स्काई -> '{sky_norm}'")
    assert sky_norm == "sky", f"Expected 'sky', got '{sky_norm}'"
    
    # 'प्राइवेट' -> private
    pvt_norm = pp.normalize_text("प्राइवेट")
    print(f"  प्राइवेट -> '{pvt_norm}'")
    assert pvt_norm == "private", f"Expected 'private', got '{pvt_norm}'"
    
    # 'लिमिटेड' -> limited
    ltd_norm = pp.normalize_text("लिमिटेड")
    print(f"  लिमिटेड -> '{ltd_norm}'")
    assert ltd_norm == "limited", f"Expected 'limited', got '{ltd_norm}'"
    
    # Combined Indic entity
    full_indic = "स्काई प्राइवेट लिमिटेड"
    full_norm = pp.normalize_text(full_indic)
    print(f"  {full_indic} -> '{full_norm}'")
    assert full_norm == "sky private limited", f"Expected 'sky private limited', got '{full_norm}'"
    
    # 2. Mixed-script text
    mixed = "स्काई Tech Solutions Pvt Ltd"
    mixed_norm = pp.normalize_text(mixed)
    print(f"  {mixed} -> '{mixed_norm}'")
    assert "sky" in mixed_norm and "tech" in mixed_norm and "private limited" in mixed_norm
    
    print("  Indic script examples verified successfully!\n")


def test_latin_and_accented_preservation():
    print("=" * 60)
    print("2. Testing Latin & Accented Text (Preserving English Vowels)")
    print("=" * 60)
    
    # English text with double vowels must NOT have its vowels collapsed!
    latin_sample = "Green Street Coffee Shop LLC"
    norm_latin = pp.normalize_text(latin_sample)
    norm_phonetic = pp.normalize_text_phonetic(latin_sample)
    print(f"  {latin_sample} -> '{norm_latin}'")
    assert "green" in norm_latin and "street" in norm_latin and "coffee" in norm_latin, \
        f"English double vowels corrupted! Got: '{norm_latin}'"
    assert norm_phonetic == norm_latin, "Phonetic variant should match standard for Latin text!"
    
    # Accented European text
    accented_sample = "Société Générale Établissements SARL"
    norm_accented = pp.normalize_text(accented_sample)
    print(f"  {accented_sample} -> '{norm_accented}'")
    assert "societe" in norm_accented and "generale" in norm_accented and "etablissements" in norm_accented
    assert "é" not in norm_accented, "Accents were not stripped!"
    
    print("  Latin and accented text verified successfully!\n")


def test_missing_and_edge_values():
    print("=" * 60)
    print("3. Testing Missing Values & Unidecode Graceful Fallback")
    print("=" * 60)
    
    import pandas as pd
    import numpy as np
    
    assert pp.normalize_text(None) == ""
    assert pp.normalize_text("") == ""
    assert pp.normalize_text(np.nan) == ""
    assert pp.normalize_text_phonetic(None) == ""
    assert pp.normalize_text_phonetic("") == ""
    assert pp.normalize_text_phonetic(np.nan) == ""
    
    # Simulate unidecode unavailable
    real_unidecode = pp.unidecode
    try:
        pp.unidecode = None
        # Should not crash when unidecode is None
        res_no_unidecode = pp.normalize_text("Acme Corp")
        assert res_no_unidecode == "acme corporation"
        res_indic_no_unidecode = pp.normalize_text("स्काई")
        assert isinstance(res_indic_no_unidecode, str)
        print("  Graceful fallback when unidecode is None verified!")
    finally:
        pp.unidecode = real_unidecode
        
    print("  Missing and edge values verified successfully!\n")


if __name__ == "__main__":
    test_indic_script_normalization()
    test_latin_and_accented_preservation()
    test_missing_and_edge_values()
    print("=" * 60)
    print("ALL EXPERIMENT 007 TESTS PASSED CLEANLY!")
    print("=" * 60)
