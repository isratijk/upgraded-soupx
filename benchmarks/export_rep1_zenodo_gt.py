#!/usr/bin/env python3
"""
Export CAST GT from rep1_Zenodo RDS into CSV and optionally Parquet.

Usage:
  python3 benchmarks/export_rep1_zenodo_gt.py
  python3 benchmarks/export_rep1_zenodo_gt.py --base datasets/rep1_Zenodo
"""

import argparse
from pathlib import Path
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    from .rep1_zenodo_utils import load_rep1_zenodo_ground_truth
except ImportError:
    from rep1_zenodo_utils import load_rep1_zenodo_ground_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="datasets/rep1_Zenodo")
    parser.add_argument("--rds", default=None, help="Path to perCell_noMito_CAST_binom.RDS")
    parser.add_argument("--csv", default=None, help="Output CSV path")
    parser.add_argument("--parquet", default=None, help="Output Parquet path")
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base}")

    rds_path = args.rds or str(base / "perCell_noMito_CAST_binom.RDS")
    out_csv = Path(args.csv) if args.csv else base / "rep1_cast_gt.csv"
    out_parquet = Path(args.parquet) if args.parquet else base / "rep1_cast_gt.parquet"

    gt = load_rep1_zenodo_ground_truth(str(base), gt_path=rds_path).reset_index()
    gt.to_csv(out_csv, index=False)
    print(f"Wrote CSV: {out_csv}")

    try:
        gt.to_parquet(out_parquet, index=False)
        print(f"Wrote Parquet: {out_parquet}")
    except Exception as exc:
        print(f"Skipped Parquet export: {exc}")

    print(f"Rows: {len(gt):,}")
    print(f"Columns: {', '.join(gt.columns)}")


if __name__ == "__main__":
    main()
