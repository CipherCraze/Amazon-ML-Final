#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Experiment 006: Compound Brand Word Blocking
Exposes conservative compound brand token blocking as a standalone module.
"""

import os
import sys
import argparse

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from expand_candidates_key_blocking import (
    extract_compound_brand_tokens,
    run_key_blocking_expansion,
    BUSINESS_STOPWORDS
)


def main():
    parser = argparse.ArgumentParser(description="Experiment 006: Compound Brand Word Blocking")
    parser.add_argument("--s1-path", default="", help="Path to S1 TSV")
    parser.add_argument("--catalog-path", default="", help="Path to Catalog TSV")
    parser.add_argument("--output-tsv", default="", help="Output TSV for new candidate pairs")
    parser.add_argument("--max-block-size", default=50, type=int, help="Maximum block size ceiling before capping")
    parser.add_argument("--dry-run", action="store_true", help="Run benchmark on synthetic fixture")
    args = parser.parse_args()
    
    from expand_candidates_key_blocking import run_benchmark_experiment_005_006
    args.output_dir = os.path.dirname(args.output_tsv) if args.output_tsv else "output"
    sys.exit(run_benchmark_experiment_005_006(args))


if __name__ == "__main__":
    main()
