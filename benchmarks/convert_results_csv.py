#!/usr/bin/env python3
"""
Convert human-readable results.csv (formatted print-table output) to the
machine-readable format expected by plot_results.py --csv.

Usage:
    python benchmarks/convert_results_csv.py
    python benchmarks/convert_results_csv.py --in results.csv --out results_raw.csv
"""

import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _na(v):
    return pd.isna(v) or str(v).strip() in ("N/A", "", "nan")


def parse_pct(v):
    if _na(v):
        return float("nan")
    try:
        return float(str(v).strip().replace("%", "")) / 100.0
    except ValueError:
        return float("nan")


def parse_float(v):
    if _na(v):
        return None
    try:
        return float(str(v).strip())
    except ValueError:
        return None


def parse_fold_pass(v):
    """'1.01×✗' -> (1.01, False), '1.01×✓' -> (1.01, True)"""
    if _na(v):
        return None, None
    s = str(v).strip()
    passed = True if "✓" in s else (False if "✗" in s else None)
    s = s.replace("✓", "").replace("✗", "").replace("×", "")
    try:
        return float(s), passed
    except ValueError:
        return None, None


def parse_fc_improved(v):
    """'1.007×↑' -> (1.007, True), '3.500×↓' -> (3.5, False)"""
    if _na(v):
        return None, None
    s = str(v).strip()
    improved = True if "↑" in s else (False if "↓" in s else None)
    s = s.replace("↑", "").replace("↓", "").replace("×", "")
    try:
        return float(s), improved
    except ValueError:
        return None, None


def parse_delta_improved(v):
    """+0.0005↑ -> (0.0005, True), -0.0007↓ -> (-0.0007, False)"""
    if _na(v):
        return None, None
    s = str(v).strip()
    improved = True if "↑" in s else (False if "↓" in s else None)
    s = s.replace("↑", "").replace("↓", "")
    try:
        return float(s), improved
    except ValueError:
        return None, None


def parse_pp(v):
    """'0.848pp' -> 0.848"""
    if _na(v):
        return None
    try:
        return float(str(v).strip().replace("pp", ""))
    except ValueError:
        return None


def parse_fold(v):
    """'1.10×' -> 1.10"""
    if _na(v):
        return None
    try:
        return float(str(v).strip().replace("×", ""))
    except ValueError:
        return None


def parse_int(v):
    if _na(v):
        return None
    try:
        return int(str(v).strip())
    except ValueError:
        return None


def convert(in_path, out_path):
    df = pd.read_csv(in_path)

    rows = []
    for _, row in df.iterrows():
        m1_fold, m1_pass     = parse_fold_pass(row.get("M1-Fold"))
        m2_fc, m2_impr       = parse_fc_improved(row.get("M2-FC"))
        m6_sil, m6_impr      = parse_delta_improved(row.get("M6-Sil"))
        m8_rank, m8_impr     = parse_delta_improved(row.get("M8-MkRk"))

        rows.append({
            "dataset":          str(row["Dataset"]).strip(),
            "pipeline":         str(row["Pipeline"]).strip(),
            "n_cells":          int(row["n_cells"]),
            "rho_mean":         parse_pct(row["rho_mean"]),
            "rho_std":          parse_pct(row["rho_std"]),
            "m1_fold":          m1_fold,
            "m1_pass":          m1_pass,
            "m2_fc_ratio":      m2_fc,
            "m2_improved":      m2_impr,
            "m3_ari":           parse_float(row.get("M3-ARI")),
            "m3_lost":          None,
            "m4_entropy_delta": parse_float(row.get("M4-Δent")),
            "m4_improved":      None,
            "m5_pct_reduction": parse_pp(row.get("M5-HBB")),
            "m5_reduced":       None,
            "gt_mae":           parse_pp(row.get("GT-MAE")),
            "gt_pearson":       parse_float(row.get("GT-r")),
            "excl_fold":        parse_fold(row.get("EX-Fold")),
            "m6_sil_delta":     m6_sil,
            "m6_improved":      m6_impr,
            "m7_n_spurious":    parse_int(row.get("M7-SpDE")),
            "m7_pct_spurious":  None,
            "m8_rank_delta":    m8_rank,
            "m8_improved":      m8_impr,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Converted {len(out_df)} rows -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",  dest="in_path",
                        default=os.path.join(REPO_ROOT, "results.csv"))
    parser.add_argument("--out", dest="out_path",
                        default=os.path.join(REPO_ROOT, "results_raw.csv"))
    args = parser.parse_args()
    convert(args.in_path, args.out_path)
