"""
Utilities for the local Zenodo-style rep1 benchmark directory.

Expected layout:
  datasets/rep1_Zenodo/
    raw_feature_bc_matrix.h5
    filtered_feature_bc_matrix.h5
    perCell_noMito_CAST_binom.RDS
    rep1_cast_gt.csv              # preferred exported GT
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd


# ── GT file discovery constants ───────────────────────────────────────────────

GT_SUFFIXES = (".csv", ".tsv", ".txt", ".parquet", ".rds")

_BARCODE_COLS = (
    "barcode", "barcodes", "cell_barcode", "cell_barcodes", "cell",
    "cells", "cell_id", "cell_ids", "cellid", "barcodes.1",
)
_GT_COLS = (
    "rho_gt", "gt", "rho", "rho_cell", "contpercell_binom",
    "contpercell", "background_noise_level", "contamination",
    "percell_nomito_cast_binom",
)
_CELLTYPE_COLS = ("cell_type", "celltype", "cell.label", "cell.labels", "label")
_STRAIN_COLS = ("strain",)
_SAMPLE_COLS = ("sample", "replicate", "dataset", "rep")
_CLUSTER_COLS = ("cluster", "clusters", "seurat_clusters")
_BATCH_COLS = ("batch",)


def _norm_barcode(barcode: object) -> str:
    text = str(barcode).strip().strip('"').strip("'")
    return text[:-2] if text.endswith("-1") else text


def _find_first_column(df: pd.DataFrame, candidates: tuple) -> Optional[str]:
    norm_to_col = {
        re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_"): col
        for col in df.columns
    }
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]+", "_", cand.strip().lower()).strip("_")
        if key in norm_to_col:
            return norm_to_col[key]
    return None


def _read_gt_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".rds":
        try:
            import pyreadr
        except ImportError as exc:
            raise ImportError(
                "Reading RDS GT files requires `pyreadr`, or export them "
                "to CSV/TSV/Parquet first."
            ) from exc
        result = pyreadr.read_r(str(path))
        if not result:
            raise ValueError(f"RDS file contained no tables: {path}")
        df = next(iter(result.values()))
        if isinstance(df, pd.Series):
            df = df.to_frame("rho_gt")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Unsupported RDS object type in {path}: {type(df)!r}")
        if df.index.name is not None or any(str(i) != str(j) for i, j in zip(df.index, range(len(df.index)))):
            df = df.reset_index().rename(columns={"index": "barcode"})
        return df
    raise ValueError(f"Unsupported GT file format: {path}")


def _candidate_gt_files(base: Path, sample_key: str):
    ranked = []
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in GT_SUFFIXES:
            continue
        lower = path.name.lower()
        score = (
            0 if sample_key in lower else 1,
            0 if "percell" in lower or "cell_metadata" in lower or "gt" in lower else 1,
            0 if "cast" in lower else 1,
            str(path),
        )
        ranked.append((score, path))
    ranked.sort(key=lambda x: x[0])
    return [path for _, path in ranked]


def _load_gt_from_dir(base_dir: str, sample_key: str = "rep1",
                      gt_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load per-cell GT contamination fractions from a directory.

    Supported formats: CSV, TSV/TXT, Parquet, and optionally RDS when
    `pyreadr` is installed. Returns a DataFrame indexed by normalized
    barcodes with a `rho_gt` column.
    """
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    paths = [Path(gt_path)] if gt_path is not None else _candidate_gt_files(base, sample_key)
    errors = []

    for path in paths:
        try:
            df = _read_gt_table(path)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            continue

        barcode_col = _find_first_column(df, _BARCODE_COLS)
        if barcode_col is None:
            if df.index.dtype == object and len(df.index):
                df = df.reset_index().rename(columns={"index": "barcode"})
                barcode_col = "barcode"
            else:
                errors.append(f"{path.name}: no barcode column found")
                continue

        sample_col = _find_first_column(df, _SAMPLE_COLS)
        if sample_col is not None:
            sample_mask = df[sample_col].astype(str).str.lower().str.contains(sample_key.lower(), na=False)
            if sample_mask.any():
                df = df.loc[sample_mask].copy()

        gt_col = _find_first_column(df, _GT_COLS)
        if gt_col is None:
            numeric_cols = [
                col for col in df.columns
                if col != barcode_col and pd.api.types.is_numeric_dtype(df[col])
            ]
            if len(numeric_cols) == 1:
                gt_col = numeric_cols[0]
            else:
                errors.append(f"{path.name}: no GT rho column found")
                continue

        out = pd.DataFrame(index=pd.Index(df[barcode_col].map(_norm_barcode), name="barcode"))
        out["rho_gt"] = pd.to_numeric(df[gt_col], errors="coerce").to_numpy()

        for out_col, candidates in (
            ("cell_type", _CELLTYPE_COLS),
            ("strain", _STRAIN_COLS),
            ("sample", _SAMPLE_COLS),
            ("cluster", _CLUSTER_COLS),
            ("batch", _BATCH_COLS),
        ):
            col = _find_first_column(df, candidates)
            if col is not None:
                out[out_col] = df[col].astype(str).to_numpy()

        out = out[~out.index.duplicated(keep="first")]
        out = out[out["rho_gt"].notna()]
        if len(out):
            return out

        errors.append(f"{path.name}: parsed successfully but no usable GT rows remained")

    searched = gt_path if gt_path is not None else str(base)
    joined = "; ".join(errors[:5]) if errors else "no candidate GT files found"
    raise FileNotFoundError(
        f"Could not load GT for {sample_key} under {searched}. {joined}"
    )


def align_ground_truth_to_cells(gt_df: pd.DataFrame, cells) -> pd.DataFrame:
    """Align GT rows to filtered-cell barcodes. Missing cells are dropped."""
    aligned = gt_df.copy()
    aligned.index = pd.Index([_norm_barcode(c) for c in aligned.index], name="barcode")
    cell_index = pd.Index([_norm_barcode(c) for c in cells], name="barcode")
    aligned = aligned.reindex(cell_index)
    aligned = aligned[aligned["rho_gt"].notna()]
    return aligned


# ── Sample loaders ────────────────────────────────────────────────────────────

def load_rep1_zenodo_sample(base_dir: str, verbose: bool = True, **kwargs):
    import numpy as np

    from SoupX.estimate_soup import estimate_soup
    from SoupX.io import read_10x_h5
    from SoupX.soup_channel import SoupChannel

    base = Path(base_dir)
    raw_h5 = base / "raw_feature_bc_matrix.h5"
    filt_h5 = base / "filtered_feature_bc_matrix.h5"
    if not raw_h5.exists():
        raise FileNotFoundError(f"Missing raw H5: {raw_h5}")
    if not filt_h5.exists():
        raise FileNotFoundError(f"Missing filtered H5: {filt_h5}")

    if verbose:
        print("Loading raw count data from H5")
    tod, genes, drop_barcodes, feature_types = read_10x_h5(str(raw_h5))
    keep = np.array([ft == "Gene Expression" for ft in feature_types], dtype=bool)
    if not keep.all():
        tod = tod[keep, :]
        genes = genes[keep]

    if verbose:
        print("Loading cell-only count data from H5")
    toc, _, cell_barcodes, filt_feature_types = read_10x_h5(str(filt_h5))
    keep_filt = np.array([ft == "Gene Expression" for ft in filt_feature_types], dtype=bool)
    if not keep_filt.all():
        toc = toc[keep_filt, :]

    sc = SoupChannel(
        tod=tod,
        toc=toc,
        genes=genes,
        cells=cell_barcodes,
        drop_barcodes=list(drop_barcodes),
        channel_name=kwargs.pop("channel_name", "rep1_zenodo"),
        calc_soup_profile=False,
        **kwargs,
    )
    estimate_soup(sc, inplace=True, keep_droplets=True)
    return sc


def load_rep1_zenodo_ground_truth(base_dir: str, gt_path: Optional[str] = None) -> pd.DataFrame:
    base = Path(base_dir)
    if gt_path is None:
        for name in (
            "rep1_cast_gt.parquet",
            "rep1_cast_gt.csv",
            "rep1_cast_gt.tsv",
            "perCell_noMito_CAST_binom.csv",
            "perCell_noMito_CAST_binom.parquet",
            "perCell_noMito_CAST_binom.tsv",
            "perCell_noMito_CAST_binom.RDS",
        ):
            candidate = base / name
            if candidate.exists():
                gt_path = str(candidate)
                break

    return _load_gt_from_dir(str(base), sample_key="rep1", gt_path=gt_path)


def load_rep1_zenodo_gt_aligned(base_dir: str, cells, gt_path: Optional[str] = None) -> pd.DataFrame:
    gt_df = load_rep1_zenodo_ground_truth(base_dir, gt_path=gt_path)
    return align_ground_truth_to_cells(gt_df, cells)
