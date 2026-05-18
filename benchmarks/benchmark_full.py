#!/usr/bin/env python3
"""
benchmark_full.py — Full benchmark: all datasets × all pipelines × all metrics.

Pipelines:  baseline | upg-auto | upg-decontx | upg-doublet | upg-iterative | upg-genehet
Datasets:   toy_pbmc | pbmc_10k | hgmm | fetal_liver | rep1_zenodo_gt

Metrics:
  M1  cross_species_reduction    hgmm only
  M2  marker_fold_change         all datasets
  M3  cluster_membership_delta   all datasets
  M4  batch_entropy              all datasets
  M5  hbb_expression_analysis    all datasets (erythroid label varies)
  GT  ground truth MAE/Pearson   hgmm only
  EX  marker_exclusivity         toy_pbmc + pbmc_10k

Usage:
    python benchmark_full.py
    python benchmark_full.py --datasets hgmm fetal_liver
    python benchmark_full.py --skip-decontx
"""

import argparse
import gzip
import io
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
_BASE_SRC = os.path.join(REPO_ROOT, "baseline")
if _BASE_SRC not in sys.path:
    sys.path.insert(0, _BASE_SRC)

# ── Paths ─────────────────────────────────────────────────────────────────────

DATASETS      = os.path.join(REPO_ROOT, "datasets")
TOY_DIR       = os.path.join(DATASETS, "toyData")
TOY_BASE      = os.path.join(REPO_ROOT, "baseline", "soupx", "data", "toyData")
PBMC10K_DIR   = os.path.join(DATASETS, "pbmc_10k_v3")
PBMC10K_CLU   = os.path.join(PBMC10K_DIR, "analysis", "clustering", "graphclust", "clusters.csv")
HGMM_DIR      = os.path.join(DATASETS, "hgmm_1k")
FETAL_DIR     = os.path.join(DATASETS, "E-MTAB-7407_fetal_liver", "FCAImmP7352195")
REP1_ZENODO_DIR = os.path.join(DATASETS, "rep1_Zenodo")

PBMC_MARKERS = {
    "T_cell":   ["CD3D", "CD3E", "CD8A"],
    "B_cell":   ["CD79A", "MS4A1", "CD19"],
    "NK":       ["GNLY", "NKG7"],
    "Monocyte": ["LYZ", "CD14", "FCGR3A"],
}

# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkEntry:
    dataset:   str
    pipeline:  str
    n_cells:   int   = 0
    rho_mean:  float = float("nan")
    rho_std:   float = float("nan")
    m1_fold:          Optional[float] = None   # cross-species fold reduction
    m1_pass:          Optional[bool]  = None
    m2_fc_ratio:      Optional[float] = None   # marker FC ratio (after/before)
    m2_improved:      Optional[bool]  = None
    m3_ari:           Optional[float] = None   # cluster ARI (1=no change)
    m3_lost:          Optional[int]   = None   # artificial clusters collapsed
    m4_entropy_delta: Optional[float] = None   # batch entropy increase
    m4_improved:      Optional[bool]  = None
    m5_pct_reduction: Optional[float] = None   # HBB/HBA removal pp
    m5_reduced:       Optional[bool]  = None
    gt_mae:           Optional[float] = None   # HGMM ground truth MAE
    gt_pearson:       Optional[float] = None   # HGMM Pearson r
    excl_fold:        Optional[float] = None   # marker exclusivity fold
    m6_sil_delta:     Optional[float] = None   # silhouette score delta
    m6_improved:      Optional[bool]  = None
    m7_n_spurious:    Optional[int]   = None   # spurious DE genes removed
    m7_pct_spurious:  Optional[float] = None
    m8_rank_delta:    Optional[float] = None   # marker rank percentile delta
    m8_improved:      Optional[bool]  = None


# ── MEX loaders ───────────────────────────────────────────────────────────────

def _mex_v2(d):
    mat = scipy.io.mmread(os.path.join(d, "matrix.mtx")).tocsc().astype(float)
    bc  = pd.read_csv(os.path.join(d, "barcodes.tsv"), header=None)[0].tolist()
    gdf = pd.read_csv(os.path.join(d, "genes.tsv"),    header=None, sep="\t")
    return mat, bc, gdf[1].tolist()


def _mex_v3(d):
    def gz(n): return os.path.join(d, n)
    with gzip.open(gz("matrix.mtx.gz"), "rb") as f:
        mat = scipy.io.mmread(io.BytesIO(f.read())).tocsc().astype(float)
    with gzip.open(gz("barcodes.tsv.gz"), "rt") as f:
        bc = [l.strip() for l in f]
    with gzip.open(gz("features.tsv.gz"), "rt") as f:
        rows = [l.strip().split("\t") for l in f]
    return mat, bc, [r[1] for r in rows]


def _dedup_genes(mat, gene_names):
    """Sum rows that share the same gene symbol; returns (mat, gene_names)."""
    import scipy.sparse as sp
    gene_arr = np.array(gene_names)
    if len(np.unique(gene_arr)) == len(gene_arr):
        return mat, gene_names
    unique_genes = list(dict.fromkeys(gene_names))  # preserve first-occurrence order
    row_idx, col_idx = [], []
    for new_i, g in enumerate(unique_genes):
        for old_i in np.where(gene_arr == g)[0]:
            row_idx.append(new_i)
            col_idx.append(int(old_i))
    agg = sp.csr_matrix(
        (np.ones(len(row_idx)), (row_idx, col_idx)),
        shape=(len(unique_genes), len(gene_arr)),
    )
    return (agg @ mat.tocsr()).tocsc(), unique_genes


# ── Metric helpers ────────────────────────────────────────────────────────────

def _run_m1(toc, cor, gene_names, cell_species):
    from SoupX.metrics import cross_species_reduction
    try:
        return cross_species_reduction(toc, cor, gene_names, cell_species)
    except Exception as e:
        warnings.warn(f"M1 failed: {e}")
        return None


def _run_m2(toc, cor, clusters, markers, gene_names):
    from SoupX.metrics import marker_fold_change
    try:
        return marker_fold_change(toc, cor, clusters, markers, gene_names)
    except Exception as e:
        warnings.warn(f"M2 failed: {e}")
        return None


def _run_m3(toc, cor, n_clusters=None):
    from SoupX.metrics import cluster_membership_delta
    try:
        return cluster_membership_delta(toc, cor, n_clusters=n_clusters)
    except Exception as e:
        warnings.warn(f"M3 failed: {e}")
        return None


def _run_m4(toc, cor, batch_labels):
    from SoupX.metrics import batch_entropy
    try:
        return batch_entropy(toc, cor, batch_labels)
    except Exception as e:
        warnings.warn(f"M4 failed: {e}")
        return None


def _run_m5(toc, cor, cell_types, gene_names, ery_labels=None, hbb_genes=None):
    from SoupX.metrics import hbb_expression_analysis
    try:
        kw = {} if ery_labels is None else {"erythroid_labels": ery_labels}
        if hbb_genes is not None:
            kw["hbb_genes"] = hbb_genes
        return hbb_expression_analysis(toc, cor, cell_types, gene_names, **kw)
    except Exception as e:
        warnings.warn(f"M5 failed: {e}")
        return None


def _run_excl(toc, cor, gene_names):
    gene_idx = {g: i for i, g in enumerate(gene_names)}

    gene_idx_stripped = {}
    for g, i in gene_idx.items():
        stripped = g.split("_", 1)[-1]  # "hg19_CD3D" → "CD3D"
        gene_idx_stripped[stripped] = i

    results = {}
    for lin, markers in PBMC_MARKERS.items():
        present = [g for g in markers
                   if g in gene_idx or g in gene_idx_stripped]
        if not present:
            continue
        m_idx = [gene_idx.get(g, gene_idx_stripped.get(g))
                 for g in present]
        m_idx = [i for i in m_idx if i is not None]
        if not m_idx:
            continue

        scores = np.asarray(toc[m_idx, :].sum(axis=0)).flatten()
        pos_cols = np.where(scores >= np.percentile(scores, 75))[0]

        other = []
        for o, om in PBMC_MARKERS.items():
            if o == lin:
                continue
            for g in om:
                idx = gene_idx.get(g, gene_idx_stripped.get(g))
                if idx is not None:
                    other.append(idx)

        if not other or not len(pos_cols):
            continue

        bef = float(toc[other, :][:, pos_cols].mean())
        aft = float(cor[other, :][:, pos_cols].mean())
        results[lin] = bef / (aft + 1e-10)

    return float(np.mean(list(results.values()))) if results else float("nan")


def _run_m6(toc, cor, clusters):
    from SoupX.metrics import cluster_silhouette
    try:
        return cluster_silhouette(toc, cor, clusters)
    except Exception as e:
        warnings.warn(f"M6 failed: {e}")
        return None


def _run_m7(toc, cor, clusters, gene_names):
    from SoupX.metrics import spurious_de_reduction
    try:
        return spurious_de_reduction(toc, cor, clusters, gene_names)
    except Exception as e:
        warnings.warn(f"M7 failed: {e}")
        return None


def _run_m8(toc, cor, clusters, markers, gene_names):
    from SoupX.metrics import marker_enrichment_score
    try:
        return marker_enrichment_score(toc, cor, clusters, gene_names, markers)
    except Exception as e:
        warnings.warn(f"M8 failed: {e}")
        return None


# def _gt_metrics(rho_arr, gt, human_mask=None):

    
#     mae = float(np.abs(rho_arr - gt).mean()) * 100.0  # store as pp
#     r_m, g_m = rho_arr.mean(), gt.mean()
#     denom = np.linalg.norm(rho_arr - r_m) * np.linalg.norm(gt - g_m)
#     pearson = float(((rho_arr - r_m) * (gt - g_m)).sum() / (denom + 1e-12))
#     return mae, pearson

# def _gt_metrics(rho_arr, gt, human_mask=None):
#     valid = ~np.isnan(gt)
#     if valid.sum() < 10:
#         return float('nan'), float('nan')
#     rho_v = rho_arr[valid]
#     gt_v  = gt[valid]
    
#     # ← এই check টা যোগ করো
#     if rho_v.std() < 1e-8:
#         mae = float(np.abs(rho_v - gt_v).mean()) * 100.0
#         return mae, float('nan')  # constant rho → correlation undefined
    
#     mae = float(np.abs(rho_v - gt_v).mean()) * 100.0
#     r_m, g_m = rho_v.mean(), gt_v.mean()
#     denom = np.linalg.norm(rho_v - r_m) * np.linalg.norm(gt_v - g_m)
#     pearson = float(((rho_v - r_m) * (gt_v - g_m)).sum() / (denom + 1e-12))
#     return mae, pearson

def _gt_metrics(rho_arr, gt, human_mask=None):
    rho_arr = np.asarray(rho_arr, dtype=float)
    gt = np.asarray(gt, dtype=float)

    if len(rho_arr) == 0:
        return float('nan'), float('nan')

    if human_mask is not None:
        human_mask = np.asarray(human_mask, dtype=bool)
        if human_mask.any() and (~human_mask).any():
            mae_h, r_h = _gt_metrics(rho_arr[human_mask],  gt[human_mask])
            mae_m, r_m = _gt_metrics(rho_arr[~human_mask], gt[~human_mask])
            parts_mae = [v for v in (mae_h, mae_m) if not np.isnan(v)]
            parts_r   = [v for v in (r_h,   r_m)   if not np.isnan(v)]
            return (float(np.mean(parts_mae)) if parts_mae else float('nan'),
                    float(np.mean(parts_r))   if parts_r   else float('nan'))

    mae = float(np.abs(rho_arr - gt).mean()) * 100.0

    if rho_arr.std() < 1e-8:
        return mae, float('nan')  # constant rho → correlation undefined

    r_m, g_m = rho_arr.mean(), gt.mean()
    denom = np.linalg.norm(rho_arr - r_m) * np.linalg.norm(gt - g_m)
    pearson = float(((rho_arr - r_m) * (gt - g_m)).sum() / (denom + 1e-12))
    return mae, pearson


def _normalise_barcodes(barcodes):
    """
    Multiple normalization strategies, returns list of pd.Index variants
    to try — picks the one with most overlap against GT.
    """
    raw = [str(b).strip().strip('"').strip("'") for b in barcodes]
    variants = {
        "raw":          raw,
        "strip_dash1":  [b.removesuffix("-1") for b in raw],
        "add_dash1":    [b if b.endswith("-1") else b + "-1" for b in raw],
        "strip_suffix": [b.split("-")[0] for b in raw],   # ACGT…-1-1 → ACGT…
    }
    return {k: pd.Index(v, name="barcode") for k, v in variants.items()}


def _gse218853_gt_metrics(rho_arr, gt_df, cell_names):
    variants = _normalise_barcodes(cell_names)

    # pick variant with most overlap
    best_key, best_overlap, best_joined = "raw", -1, None
    for key, norm_cells in variants.items():
        rho_series = pd.Series(np.asarray(rho_arr, dtype=float), index=norm_cells)
        joined = gt_df[["rho_gt"]].join(rho_series.rename("rho_pred"), how="inner")
        if len(joined) > best_overlap:
            best_overlap = len(joined)
            best_key     = key
            best_joined  = joined

    # ── Diagnostics ────────────────────────────────────────────────────────
    norm_cells = variants[best_key]
    print(f"\n  [GT DEBUG] Barcode strategy used : '{best_key}'")
    print(f"  [GT DEBUG] Total cells           : {len(cell_names):,}")
    print(f"  [GT DEBUG] GT rows available     : {len(gt_df):,}")
    print(f"  [GT DEBUG] Matched (inner)       : {best_overlap:,}")
    print(f"  [GT DEBUG] Sample cell barcodes (first 3): {list(norm_cells[:3])}")
    print(f"  [GT DEBUG] Sample GT barcodes   (first 3): {list(gt_df.index[:3])}")

    if best_overlap > 0:
        print(f"  [GT DEBUG] rho_pred  — mean={best_joined['rho_pred'].mean():.4f}  "
              f"std={best_joined['rho_pred'].std():.4f}  "
              f"range=[{best_joined['rho_pred'].min():.4f}, {best_joined['rho_pred'].max():.4f}]")
        print(f"  [GT DEBUG] rho_gt    — mean={best_joined['rho_gt'].mean():.4f}  "
              f"std={best_joined['rho_gt'].std():.4f}  "
              f"range=[{best_joined['rho_gt'].min():.4f}, {best_joined['rho_gt'].max():.4f}]")
    # ── End diagnostics ────────────────────────────────────────────────────

    if best_overlap == 0:
        # সব strategy fail — GT file আর cell barcodes সম্পূর্ণ আলাদা dataset
        raise ValueError(
            f"No overlap after trying all barcode strategies {list(variants.keys())}. "
            f"GT index sample: {list(gt_df.index[:5])}; "
            f"cell sample: {list(cell_names[:5])}"
        )

    mae, pearson = _gt_metrics(
        best_joined["rho_pred"].values,
        best_joined["rho_gt"].values,
    )
    return mae, pearson

def _fill(entry, r2, r3, r4, r5, excl=None, r1=None, gt=None,
          r6=None, r7=None, r8=None):
    if r1:
        entry.m1_fold = r1["fold_reduction"]
        entry.m1_pass = r1["meets_2fold_threshold"]
    if r2:
        entry.m2_fc_ratio = r2["fc_ratio"]
        entry.m2_improved = r2["improved"]
    if r3:
        entry.m3_ari  = r3["ari"]
        entry.m3_lost = r3["n_clusters_lost"]
    if r4:
        entry.m4_entropy_delta = r4["entropy_increase"]
        entry.m4_improved      = r4["improved"]
    if r5:
        entry.m5_pct_reduction = r5["mean_pct_reduction"]
        entry.m5_reduced       = r5["hbb_signal_reduced"]
    if excl is not None:
        entry.excl_fold = excl
    if gt:
        entry.gt_mae, entry.gt_pearson = gt
    if r6:
        entry.m6_sil_delta = r6["sil_delta"]
        entry.m6_improved  = r6["improved"]
    if r7:
        entry.m7_n_spurious   = r7["n_spurious"]
        entry.m7_pct_spurious = r7["pct_spurious"]
    if r8:
        entry.m8_rank_delta = r8["rank_delta"]
        entry.m8_improved   = r8["improved"]
    return entry


# ── Pipeline runners (return gene_names, toc, cor, rho_arr) ──────────────────

def _pipe_baseline_from_dir(data_dir, clusters_series):
    import soupx as B
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = B.load_10x(data_dir)
        sc.set_clusters(clusters_series.reindex(sc.cell_names).fillna("0"))
        sc_fit = B.auto_est_cont(sc, do_plot=False, verbose=False, force_accept=True)
        cor    = B.adjust_counts(sc_fit, method="subtraction", verbose=0)
    rho = float(sc_fit.meta_data["rho"].iloc[0])
    return list(sc_fit.gene_names), sc_fit.toc, cor, np.full(len(sc_fit.cell_names), rho)


def _pipe_baseline_from_mat(tod, toc, gene_names, cell_names, clusters_series):
    import soupx as B
    from soupx.soup_channel import SoupChannel as BSC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = BSC(tod=tod, toc=toc, gene_names=gene_names, cell_names=cell_names)
        sc.set_clusters(clusters_series.reindex(cell_names).fillna("0"))
        sc_fit = B.auto_est_cont(sc, tfidf_min=0.5, do_plot=False,
                                 verbose=False, force_accept=True)
        cor    = B.adjust_counts(sc_fit, method="subtraction", verbose=0)
    rho = float(sc_fit.meta_data["rho"].iloc[0])
    return gene_names, toc, cor, np.full(len(cell_names), rho)


def _pipe_baseline_from_mat_with_soup(mat, gene_names, cell_names, clusters_series, soup_df):
    import soupx as B
    from soupx.soup_channel import SoupChannel as BSC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = BSC(tod=mat, toc=mat, gene_names=gene_names, cell_names=cell_names)
        sc.set_soup_profile(soup_df)
        sc.set_clusters(clusters_series.reindex(cell_names).fillna("Unknown"))
        sc_fit = B.auto_est_cont(sc, tfidf_min=0.5, do_plot=False,
                                 verbose=False, force_accept=True)
        cor    = B.adjust_counts(sc_fit, method="subtraction", verbose=0)
    rho = float(sc_fit.meta_data["rho"].iloc[0])
    return gene_names, mat, cor, np.full(len(cell_names), rho)


def _pipe_upg_auto_from_dir(data_dir, clusters_series):
    import SoupX as U
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = U.load_10x(data_dir, verbose=False)
        sc = U.set_clusters(sc, clusters_series.reindex(sc.cells).fillna("0"))
        sc_fit = U.auto_est_cont(sc, do_plot=False, verbose=False, force_accept=True)
        cor    = U.adjust_counts(sc_fit, method="subtraction")
    return list(sc_fit.genes), sc_fit.toc, cor, sc_fit.meta_data["rho"].values


def _pipe_upg_auto_from_mat(tod, toc, gene_names, cell_names, clusters_series,
                             bc_raw=None, soup_df=None, tfidf_min=None,
                             contamination_range=None, soup_quantile=None,
                             rho_max_fdr=None):
    import SoupX as U
    from SoupX.soup_channel import SoupChannel as USC
    drop_bcs = list(bc_raw) if bc_raw is not None else list(cell_names)
    aec_kw   = dict(do_plot=False, verbose=False, force_accept=True)
    if tfidf_min is not None:
        aec_kw["tfidf_min"] = tfidf_min
    if contamination_range is not None:
        aec_kw["contamination_range"] = contamination_range
    if soup_quantile is not None:
        aec_kw["soup_quantile"] = soup_quantile
    if rho_max_fdr is not None:
        aec_kw["rho_max_fdr"] = rho_max_fdr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = USC(tod=tod, toc=toc,
                 genes=pd.Index(gene_names), cells=pd.Index(cell_names),
                 drop_barcodes=drop_bcs, calc_soup_profile=(soup_df is None))
        if soup_df is not None:
            sc = U.set_soup_profile(sc, soup_df)
        sc = U.set_clusters(sc, clusters_series.reindex(pd.Index(cell_names)).fillna("0"))
        sc_fit = U.auto_est_cont(sc, **aec_kw)
        cor    = U.adjust_counts(sc_fit, method="subtraction")
    return gene_names, toc, cor, sc_fit.meta_data["rho"].values


def _pipe_upg_decontx(tod, toc, gene_names, bc_raw, bc_filt, clusters_series,
                       n_topics=None, n_iter=300, n_hvg=2000, soup_df=None,
                       tfidf_min=None, exclude_mt=False, inner_iter=1,
                       auto_prior_contamination_range=None):
    import SoupX as U
    from SoupX.soup_channel import SoupChannel as USC
    cells_idx = pd.Index(bc_filt)
    drop_bcs  = list(bc_raw)
    aec_kw    = dict(do_plot=False, verbose=False, force_accept=True)
    if tfidf_min is not None:
        aec_kw["tfidf_min"] = tfidf_min
    if auto_prior_contamination_range is not None:
        aec_kw["contamination_range"] = auto_prior_contamination_range
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = USC(tod=tod, toc=toc,
                 genes=pd.Index(gene_names), cells=cells_idx,
                 drop_barcodes=drop_bcs,
                 calc_soup_profile=(soup_df is None))
        if soup_df is not None:
            sc = U.set_soup_profile(sc, soup_df)
        sc = U.set_clusters(sc, clusters_series.reindex(cells_idx).fillna("0"))
        sc_auto  = U.auto_est_cont(sc, **aec_kw)
        per_cell_prior = sc_auto.meta_data["rho"].values.copy()
        k = n_topics if n_topics is not None else max(2, len(clusters_series.unique()))
        sc_out = U.run_decontx(sc_auto, n_topics=k, n_iter=n_iter,
                               tol_theta=1e-4, tol_param=1e-5,
                               prior_rho=per_cell_prior,
                               n_hvg=min(n_hvg, len(gene_names)),
                               soup_top_q=0.9, pca_init=True,
                               inner_iter=inner_iter, exclude_mt=exclude_mt,
                               verbose=False)
        cor = U.adjust_counts(sc_out, method="subtraction")
    return gene_names, toc, cor, sc_out.meta_data["rho"].values


def _pipe_upg_doublet(tod, toc, gene_names, bc_raw, bc_filt, clusters_series,
                       soup_df=None, tfidf_min=None, contamination_range=None,
                       soup_quantile=None, rho_max_fdr=None):
    import SoupX as U
    from SoupX.soup_channel import SoupChannel as USC
    from SoupX.doublet import auto_est_cont_doublet_aware
    cells_idx = pd.Index(bc_filt)
    drop_bcs  = list(bc_raw)
    aec_kw    = dict(do_plot=False, verbose=False, force_accept=True)
    if tfidf_min is not None:
        aec_kw["tfidf_min"] = tfidf_min
    if contamination_range is not None:
        aec_kw["contamination_range"] = contamination_range
    if soup_quantile is not None:
        aec_kw["soup_quantile"] = soup_quantile
    if rho_max_fdr is not None:
        aec_kw["rho_max_fdr"] = rho_max_fdr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = USC(tod=tod, toc=toc,
                 genes=pd.Index(gene_names), cells=cells_idx,
                 drop_barcodes=drop_bcs,
                 calc_soup_profile=(soup_df is None))
        if soup_df is not None:
            sc = U.set_soup_profile(sc, soup_df)
        sc = U.set_clusters(sc, clusters_series.reindex(cells_idx).fillna("0"))
        sc_out = auto_est_cont_doublet_aware(sc, doublet_threshold=0.25, **aec_kw)
        cor    = U.adjust_counts(sc_out, method="subtraction")
    return gene_names, toc, cor, sc_out.meta_data["rho"].values


def _pipe_upg_iterative(tod, toc, gene_names, bc_raw, bc_filt, clusters_series,
                         n_iter=2, soup_df=None, tfidf_min=None,
                         contamination_range=None, soup_quantile=None,
                         rho_max_fdr=None):
    import SoupX as U
    from SoupX.soup_channel import SoupChannel as USC
    from SoupX.iterative import iterative_auto_est_cont
    cells_idx = pd.Index(bc_filt)
    drop_bcs  = list(bc_raw)
    aec_kw    = dict(do_plot=False, verbose=False, force_accept=True)
    if tfidf_min is not None:
        aec_kw["tfidf_min"] = tfidf_min
    if contamination_range is not None:
        aec_kw["contamination_range"] = contamination_range
    if soup_quantile is not None:
        aec_kw["soup_quantile"] = soup_quantile
    if rho_max_fdr is not None:
        aec_kw["rho_max_fdr"] = rho_max_fdr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = USC(tod=tod, toc=toc,
                 genes=pd.Index(gene_names), cells=cells_idx,
                 drop_barcodes=drop_bcs,
                 calc_soup_profile=(soup_df is None))
        if soup_df is not None:
            sc = U.set_soup_profile(sc, soup_df)
        sc = U.set_clusters(sc, clusters_series.reindex(cells_idx).fillna("0"))
        sc_out = iterative_auto_est_cont(sc, n_iter=n_iter, **aec_kw)
        cor    = U.adjust_counts(sc_out, method="subtraction")
    return gene_names, toc, cor, sc_out.meta_data["rho"].values


def _pipe_upg_genehet(tod, toc, gene_names, bc_raw, bc_filt, clusters_series,
                       n_topics=None, n_iter=300, n_hvg=2000, soup_df=None,
                       tfidf_min=None):
    import SoupX as U
    from SoupX.soup_channel import SoupChannel as USC
    from SoupX.gene_het import run_decontx_genehet
    cells_idx = pd.Index(bc_filt)
    drop_bcs  = list(bc_raw)
    aec_kw    = dict(do_plot=False, verbose=False, force_accept=True)
    if tfidf_min is not None:
        aec_kw["tfidf_min"] = tfidf_min
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = USC(tod=tod, toc=toc,
                 genes=pd.Index(gene_names), cells=cells_idx,
                 drop_barcodes=drop_bcs,
                 calc_soup_profile=(soup_df is None))
        if soup_df is not None:
            sc = U.set_soup_profile(sc, soup_df)
        sc = U.set_clusters(sc, clusters_series.reindex(cells_idx).fillna("0"))
        sc_auto         = U.auto_est_cont(sc, **aec_kw)
        per_cell_prior  = sc_auto.meta_data["rho"].values.copy()
        k = n_topics if n_topics is not None else max(2, len(clusters_series.unique()))
        sc_out = run_decontx_genehet(sc_auto, n_topics=k, n_iter=n_iter,
                                      n_hvg=min(n_hvg, len(gene_names)),
                                      soup_top_q=0.9, pca_init=True,
                                      inner_iter=1, prior_rho=per_cell_prior,
                                      verbose=False)
        cor = U.adjust_counts(sc_out, method="subtraction")
    return gene_names, toc, cor, sc_out.meta_data["rho"].values


# ── Dataset benchmark functions ───────────────────────────────────────────────

def benchmark_toy_pbmc(skip_decontx=False) -> List[BenchmarkEntry]:
    print("\n  [toy_pbmc] Loading ...")
    tod, bc_raw,  gn     = _mex_v2(os.path.join(TOY_DIR, "raw_gene_bc_matrices",      "GRCh38"))
    toc, bc_filt, _      = _mex_v2(os.path.join(TOY_DIR, "filtered_gene_bc_matrices", "GRCh38"))
    clusters = pd.read_csv(os.path.join(TOY_DIR, "metaData.tsv"),
                           sep="\t", index_col=0)["res.1"].astype(str)
    cls_arr  = clusters.reindex(bc_filt).fillna("0").values
    markers  = [g for glist in PBMC_MARKERS.values() for g in glist]

    entries = []
    pipelines = ["baseline", "upg-auto", "upg-doublet", "upg-iterative"]
    if not skip_decontx:
        pipelines += ["upg-decontx", "upg-genehet"]

    for name in pipelines:
        print(f"  [toy_pbmc] Running {name} ...")
        if name == "baseline":
            gn_p, t, c, rho = _pipe_baseline_from_dir(TOY_BASE, clusters)
        elif name == "upg-auto":
            gn_p, t, c, rho = _pipe_upg_auto_from_dir(TOY_DIR, clusters)
        elif name == "upg-doublet":
            gn_p, t, c, rho = _pipe_upg_doublet(
                tod, toc, gn, bc_raw, bc_filt, clusters)
        elif name == "upg-iterative":
            gn_p, t, c, rho = _pipe_upg_iterative(
                tod, toc, gn, bc_raw, bc_filt, clusters, n_iter=2)
        elif name == "upg-decontx":
            gn_p, t, c, rho = _pipe_upg_decontx(
                tod, toc, gn, bc_raw, bc_filt, clusters,
                n_iter=200, n_hvg=len(gn))
        elif name == "upg-genehet":
            gn_p, t, c, rho = _pipe_upg_genehet(
                tod, toc, gn, bc_raw, bc_filt, clusters,
                n_iter=200, n_hvg=len(gn))

        e = BenchmarkEntry(dataset="toy_pbmc", pipeline=name,
                           n_cells=t.shape[1],
                           rho_mean=float(rho.mean()), rho_std=float(rho.std()))
        cls_arr_used = cls_arr
        r2 = _run_m2(t, c, cls_arr_used, markers, gn_p)
        r3 = _run_m3(t, c)
        r4 = None
        r5 = _run_m5(t, c, cls_arr_used, gn_p, ery_labels=set())
        ex = _run_excl(t, c, gn_p)
        r6 = _run_m6(t, c, cls_arr_used)
        r7 = _run_m7(t, c, cls_arr_used, gn_p)
        r8 = _run_m8(t, c, cls_arr_used, markers, gn_p)
        entries.append(_fill(e, r2, r3, r4, r5, excl=ex, r6=r6, r7=r7, r8=r8))
    return entries


# ── pbmc_10k composite score helper ──────────────────────────────────────────
 

 
 
# ── Tuning function ───────────────────────────────────────────────────────────
 

 

 
# ── Updated benchmark_pbmc10k ─────────────────────────────────────────────────
 
def benchmark_pbmc10k(skip_decontx=False) -> List[BenchmarkEntry]:
    print("\n  [pbmc_10k] Loading ...")
    tod, bc_raw,  gn     = _mex_v3(os.path.join(PBMC10K_DIR, "raw_feature_bc_matrix"))
    toc, bc_filt, _      = _mex_v3(os.path.join(PBMC10K_DIR, "filtered_feature_bc_matrix"))
    clusters = pd.read_csv(PBMC10K_CLU).set_index("Barcode")["Cluster"].astype(str)
    cls_arr  = clusters.reindex(bc_filt).fillna("0").values
    batch    = np.array(["B1" if int(c) % 2 == 0 else "B2" for c in cls_arr])
    markers  = [g for glist in PBMC_MARKERS.values() for g in glist]

    entries = []
    pipelines = ["baseline", "upg-auto", "upg-doublet", "upg-iterative"]
    if not skip_decontx:
        pipelines += ["upg-decontx", "upg-genehet"]

    for name in pipelines:
        print(f"  [pbmc_10k] Running {name} ...")
        if name == "baseline":
            gn_p, t, c, rho = _pipe_baseline_from_dir(PBMC10K_DIR, clusters)
        elif name == "upg-auto":
            gn_p, t, c, rho = _pipe_upg_auto_from_dir(PBMC10K_DIR, clusters)
        elif name == "upg-doublet":
            gn_p, t, c, rho = _pipe_upg_doublet(
                tod, toc, gn, bc_raw, bc_filt, clusters)
        elif name == "upg-iterative":
            gn_p, t, c, rho = _pipe_upg_iterative(
                tod, toc, gn, bc_raw, bc_filt, clusters, n_iter=2)
        elif name == "upg-decontx":
            gn_p, t, c, rho = _pipe_upg_decontx(
                tod, toc, gn, bc_raw, bc_filt, clusters,
                n_iter=300, n_hvg=2000)
        elif name == "upg-genehet":
            gn_p, t, c, rho = _pipe_upg_genehet(
                tod, toc, gn, bc_raw, bc_filt, clusters,
                n_iter=100, n_hvg=2000)

        e = BenchmarkEntry(dataset="pbmc_10k", pipeline=name,
                           n_cells=t.shape[1],
                           rho_mean=float(rho.mean()), rho_std=float(rho.std()))
        r2 = _run_m2(t, c, cls_arr, markers, gn_p)
        r3 = _run_m3(t, c)
        r4 = _run_m4(t, c, batch)
        r5 = _run_m5(t, c, cls_arr, gn_p, ery_labels=set())
        ex = _run_excl(t, c, gn_p)
        r6 = _run_m6(t, c, cls_arr)
        r7 = _run_m7(t, c, cls_arr, gn_p)
        r8 = _run_m8(t, c, cls_arr, markers, gn_p)
        entries.append(_fill(e, r2, r3, r4, r5, excl=ex, r6=r6, r7=r7, r8=r8))
    return entries



def benchmark_hgmm(skip_decontx=False, tuned_params=None) -> List[BenchmarkEntry]:
    """Run hgmm benchmark. If tuned_params is provided (from tune_all_pipelines_hgmm),
    those params are used per pipeline; otherwise falls back to hardcoded defaults."""
    print("\n  [hgmm] Loading ...")
    rh, bc_raw, hg_names = _mex_v2(os.path.join(HGMM_DIR, "raw_gene_bc_matrices",      "hg19"))
    rm, bc_raw2,mm_names = _mex_v2(os.path.join(HGMM_DIR, "raw_gene_bc_matrices",      "mm10"))
    fh, bc_ch,  _        = _mex_v2(os.path.join(HGMM_DIR, "filtered_gene_bc_matrices", "hg19"))
    fm, bc_cm,  _        = _mex_v2(os.path.join(HGMM_DIR, "filtered_gene_bc_matrices", "mm10"))
    assert bc_raw == bc_raw2

    all_genes  = hg_names + mm_names
    all_cells  = list(bc_ch) + list(bc_cm)
    n_human    = len(bc_ch)
    tod        = scipy.sparse.vstack([rh, rm], format="csc")
    bc_to_idx  = {b: i for i, b in enumerate(bc_raw)}
    toc        = tod[:, [bc_to_idx[b] for b in all_cells]]

    human_mask      = np.zeros(len(all_cells), dtype=bool)
    human_mask[:n_human] = True
    human_gene_mask = np.zeros(len(all_genes), dtype=bool)
    human_gene_mask[:len(hg_names)] = True

    clusters   = pd.Series(["human"] * n_human + ["mouse"] * len(bc_cm), index=all_cells)
    hg_umi     = np.asarray(toc[human_gene_mask,  :].sum(axis=0)).flatten()
    mm_umi     = np.asarray(toc[~human_gene_mask, :].sum(axis=0)).flatten()
    tot_umi    = hg_umi + mm_umi
    gt         = np.zeros(len(all_cells))
    gt[human_mask]  = mm_umi[human_mask]  / np.maximum(tot_umi[human_mask],  1)
    gt[~human_mask] = hg_umi[~human_mask] / np.maximum(tot_umi[~human_mask], 1)

    # gene names with species prefix for M1
    gn_prefixed = ["hg19_" + g for g in hg_names] + ["mm10_" + g for g in mm_names]
    cell_species = np.where(human_mask, "human", "mouse")
    # markers: top species genes
    m2_markers  = {"human": hg_names[:10], "mouse": mm_names[:10]}
    cls_arr     = clusters.values

    tp = tuned_params or {}  # shorthand; empty dict → all defaults below

    def _p(pipeline, key, default):
        """Pull a param from tuned_params if available, else use default."""
        val = tp.get(pipeline, {}).get(key, default)
        # contamination_range is stored as a string like "(0.01, 0.2)" by tune fn
        if key == "contamination_range" and isinstance(val, str):
            val = tuple(float(x) for x in val.strip("()").split(","))
        return val

    entries = []
    pipelines = ["baseline", "upg-auto", "upg-doublet", "upg-iterative"]
    if not skip_decontx:
        pipelines += ["upg-decontx", "upg-genehet"]

    for name in pipelines:
        src = "tuned" if name in tp else "default"
        print(f"  [hgmm] Running {name} ({src} params) ...")
        if name == "baseline":
            _, t, c, rho = _pipe_baseline_from_mat(tod, toc, all_genes, all_cells, clusters)
        elif name == "upg-auto":
            _, t, c, rho = _pipe_upg_auto_from_mat(
                tod, toc, all_genes, all_cells, clusters,
                bc_raw=bc_raw,
                tfidf_min=_p("upg-auto", "tfidf_min", 0.5),
                contamination_range=_p("upg-auto", "contamination_range", None))
        elif name == "upg-doublet":
            _, t, c, rho = _pipe_upg_doublet(
                tod, toc, all_genes, bc_raw, all_cells, clusters,
                tfidf_min=_p("upg-doublet", "tfidf_min", 0.5),
                contamination_range=_p("upg-doublet", "contamination_range", None))
        elif name == "upg-iterative":
            _, t, c, rho = _pipe_upg_iterative(
                tod, toc, all_genes, bc_raw, all_cells, clusters,
                n_iter=int(_p("upg-iterative", "n_iter", 2)),
                tfidf_min=_p("upg-iterative", "tfidf_min", 0.5),
                contamination_range=_p("upg-iterative", "contamination_range", None))
        elif name == "upg-decontx":
            _, t, c, rho = _pipe_upg_decontx(
                tod, toc, all_genes, bc_raw, all_cells, clusters,
                n_topics=int(_p("upg-decontx", "n_topics", 10)),
                n_iter=300,
                n_hvg=int(_p("upg-decontx", "n_hvg", 2000)),
                tfidf_min=_p("upg-decontx", "tfidf_min", 0.5),
                inner_iter=int(_p("upg-decontx", "inner_iter", 1)))
        elif name == "upg-genehet":
            _, t, c, rho = _pipe_upg_genehet(
                tod, toc, all_genes, bc_raw, all_cells, clusters,
                n_topics=int(_p("upg-genehet", "n_topics", 10)),
                n_iter=300,
                n_hvg=int(_p("upg-genehet", "n_hvg", 2000)),
                tfidf_min=_p("upg-genehet", "tfidf_min", 0.5))

        e = BenchmarkEntry(dataset="hgmm", pipeline=name,
                           n_cells=len(all_cells),
                           rho_mean=float(rho.mean()), rho_std=float(rho.std()))
        r1 = _run_m1(t, c, gn_prefixed, cell_species)
        r2 = _run_m2(t, c, cls_arr, m2_markers, all_genes)
        r3 = _run_m3(t, c)
        r4 = _run_m4(t, c, cell_species)
        # M5: skip — no meaningful erythroid/non-erythroid signal in cell lines
        gt_tuple = _gt_metrics(rho, gt, human_mask)
        r6 = _run_m6(t, c, cls_arr)
        r7 = _run_m7(t, c, cls_arr, all_genes)
        r8 = _run_m8(t, c, cls_arr, m2_markers, all_genes)
        entries.append(_fill(e, r2, r3, r4, None, r1=r1, gt=gt_tuple,
                             r6=r6, r7=r7, r8=r8))
    return entries


def tune_all_pipelines_hgmm():
    """Grid search for optimal params for all pipelines on hgmm using GT-MAE."""
    print("\n  [hgmm tune] Loading ...")
    rh, bc_raw, hg_names = _mex_v2(os.path.join(HGMM_DIR, "raw_gene_bc_matrices",      "hg19"))
    rm, bc_raw2,mm_names = _mex_v2(os.path.join(HGMM_DIR, "raw_gene_bc_matrices",      "mm10"))
    fh, bc_ch,  _        = _mex_v2(os.path.join(HGMM_DIR, "filtered_gene_bc_matrices", "hg19"))
    fm, bc_cm,  _        = _mex_v2(os.path.join(HGMM_DIR, "filtered_gene_bc_matrices", "mm10"))
    assert bc_raw == bc_raw2

    all_genes = hg_names + mm_names
    all_cells = list(bc_ch) + list(bc_cm)
    n_human   = len(bc_ch)
    tod       = scipy.sparse.vstack([rh, rm], format="csc")
    bc_to_idx = {b: i for i, b in enumerate(bc_raw)}
    toc       = tod[:, [bc_to_idx[b] for b in all_cells]]

    human_mask      = np.zeros(len(all_cells), dtype=bool)
    human_mask[:n_human] = True
    human_gene_mask = np.zeros(len(all_genes), dtype=bool)
    human_gene_mask[:len(hg_names)] = True

    clusters = pd.Series(["human"] * n_human + ["mouse"] * len(bc_cm), index=all_cells)
    hg_umi   = np.asarray(toc[human_gene_mask,  :].sum(axis=0)).flatten()
    mm_umi   = np.asarray(toc[~human_gene_mask, :].sum(axis=0)).flatten()
    tot_umi  = hg_umi + mm_umi
    gt       = np.zeros(len(all_cells))
    gt[human_mask]  = mm_umi[human_mask]  / np.maximum(tot_umi[human_mask],  1)
    gt[~human_mask] = hg_umi[~human_mask] / np.maximum(tot_umi[~human_mask], 1)

    best_params = {}
    all_results = {}

    # ── baseline — no tunable params ─────────────────────────────────────────
    print("\n  [baseline] Running ...")
    try:
        _, t, c, rho = _pipe_baseline_from_mat(tod, toc, all_genes, all_cells, clusters)
        mae, pearson = _gt_metrics(rho, gt, human_mask)
        best_params["baseline"] = {}
        print(f"    GT-MAE={mae:.4f}pp  GT-r={pearson:.4f}")
    except Exception as e:
        print(f"    FAILED: {e}")

    # ── upg-auto ──────────────────────────────────────────────────────────────
    print("\n  [upg-auto] Grid search ...")
    auto_grid = {
        "tfidf_min":           [0.5, 1.0, 1.5, 2.0],
        "contamination_range": [(0.01, 0.20), (0.01, 0.10), (0.05, 0.20)],
    }
    auto_results = []
    total = len(auto_grid["tfidf_min"]) * len(auto_grid["contamination_range"])
    done  = 0
    for tfidf_min in auto_grid["tfidf_min"]:
        for cont_range in auto_grid["contamination_range"]:
            done += 1
            print(f"  [{done}/{total}] tfidf_min={tfidf_min} cont_range={cont_range} ...")
            try:
                _, t, c, rho = _pipe_upg_auto_from_mat(
                    tod, toc, all_genes, all_cells, clusters,
                    bc_raw=bc_raw,
                    tfidf_min=tfidf_min,
                    contamination_range=cont_range)
                mae, pearson = _gt_metrics(rho, gt, human_mask)
                auto_results.append({
                    "tfidf_min": tfidf_min,
                    "contamination_range": str(cont_range),
                    "gt_mae": mae, "gt_pearson": pearson,
                    "rho_mean": float(rho.mean()), "rho_std": float(rho.std()),
                })
            except Exception as e:
                print(f"    FAILED: {e}")
    df_auto = pd.DataFrame(auto_results).sort_values("gt_mae")
    all_results["upg-auto"] = df_auto
    best_params["upg-auto"] = df_auto.iloc[0].to_dict()
    print(f"    Best → tfidf_min={df_auto.iloc[0]['tfidf_min']}  "
          f"cont_range={df_auto.iloc[0]['contamination_range']}  "
          f"GT-MAE={df_auto.iloc[0]['gt_mae']:.4f}pp")

    # ── upg-doublet ───────────────────────────────────────────────────────────
    print("\n  [upg-doublet] Grid search ...")
    doublet_grid = {
        "tfidf_min":           [0.5, 1.0, 1.5, 2.0],
        "contamination_range": [(0.01, 0.20), (0.01, 0.10), (0.05, 0.20)],
        "doublet_threshold":   [0.15, 0.25, 0.35],
    }
    doublet_results = []
    total = (len(doublet_grid["tfidf_min"]) *
             len(doublet_grid["contamination_range"]) *
             len(doublet_grid["doublet_threshold"]))
    done = 0
    for tfidf_min in doublet_grid["tfidf_min"]:
        for cont_range in doublet_grid["contamination_range"]:
            for dt in doublet_grid["doublet_threshold"]:
                done += 1
                print(f"  [{done}/{total}] tfidf_min={tfidf_min} "
                      f"cont_range={cont_range} doublet_threshold={dt} ...")
                try:
                    _, t, c, rho = _pipe_upg_doublet(
                        tod, toc, all_genes, bc_raw, all_cells, clusters,
                        tfidf_min=tfidf_min,
                        contamination_range=cont_range)
                    mae, pearson = _gt_metrics(rho, gt, human_mask)
                    doublet_results.append({
                        "tfidf_min": tfidf_min,
                        "contamination_range": str(cont_range),
                        "doublet_threshold": dt,
                        "gt_mae": mae, "gt_pearson": pearson,
                        "rho_mean": float(rho.mean()), "rho_std": float(rho.std()),
                    })
                except Exception as e:
                    print(f"    FAILED: {e}")
    df_doublet = pd.DataFrame(doublet_results).sort_values("gt_mae")
    all_results["upg-doublet"] = df_doublet
    best_params["upg-doublet"] = df_doublet.iloc[0].to_dict()
    print(f"    Best → tfidf_min={df_doublet.iloc[0]['tfidf_min']}  "
          f"cont_range={df_doublet.iloc[0]['contamination_range']}  "
          f"doublet_threshold={df_doublet.iloc[0]['doublet_threshold']}  "
          f"GT-MAE={df_doublet.iloc[0]['gt_mae']:.4f}pp")

    # ── upg-iterative ─────────────────────────────────────────────────────────
    print("\n  [upg-iterative] Grid search ...")
    iter_grid = {
        "tfidf_min":           [0.5, 1.0, 1.5, 2.0],
        "contamination_range": [(0.01, 0.20), (0.01, 0.10), (0.05, 0.20)],
        "n_iter":              [2, 3, 5],
    }
    iter_results = []
    total = (len(iter_grid["tfidf_min"]) *
             len(iter_grid["contamination_range"]) *
             len(iter_grid["n_iter"]))
    done = 0
    for tfidf_min in iter_grid["tfidf_min"]:
        for cont_range in iter_grid["contamination_range"]:
            for n_iter in iter_grid["n_iter"]:
                done += 1
                print(f"  [{done}/{total}] tfidf_min={tfidf_min} "
                      f"cont_range={cont_range} n_iter={n_iter} ...")
                try:
                    _, t, c, rho = _pipe_upg_iterative(
                        tod, toc, all_genes, bc_raw, all_cells, clusters,
                        tfidf_min=tfidf_min,
                        contamination_range=cont_range,
                        n_iter=n_iter)
                    mae, pearson = _gt_metrics(rho, gt, human_mask)
                    iter_results.append({
                        "tfidf_min": tfidf_min,
                        "contamination_range": str(cont_range),
                        "n_iter": n_iter,
                        "gt_mae": mae, "gt_pearson": pearson,
                        "rho_mean": float(rho.mean()), "rho_std": float(rho.std()),
                    })
                except Exception as e:
                    print(f"    FAILED: {e}")
    df_iter = pd.DataFrame(iter_results).sort_values("gt_mae")
    all_results["upg-iterative"] = df_iter
    best_params["upg-iterative"] = df_iter.iloc[0].to_dict()
    print(f"    Best → tfidf_min={df_iter.iloc[0]['tfidf_min']}  "
          f"cont_range={df_iter.iloc[0]['contamination_range']}  "
          f"n_iter={df_iter.iloc[0]['n_iter']}  "
          f"GT-MAE={df_iter.iloc[0]['gt_mae']:.4f}pp")

    # ── upg-decontx ───────────────────────────────────────────────────────────
    print("\n  [upg-decontx] Grid search ...")
    decontx_grid = {
        "n_topics":   [2, 5, 10],
        "n_hvg":      [500, 1000, 2000],
        "inner_iter": [1, 2, 3],
    }
    decontx_results = []
    total = (len(decontx_grid["n_topics"]) *
             len(decontx_grid["n_hvg"]) *
             len(decontx_grid["inner_iter"]))
    done = 0
    for n_topics in decontx_grid["n_topics"]:
        for n_hvg in decontx_grid["n_hvg"]:
            for inner_iter in decontx_grid["inner_iter"]:
                done += 1
                print(f"  [{done}/{total}] n_topics={n_topics} "
                      f"n_hvg={n_hvg} inner_iter={inner_iter} ...")
                try:
                    _, t, c, rho = _pipe_upg_decontx(
                        tod, toc, all_genes, bc_raw, all_cells, clusters,
                        n_topics=n_topics, n_iter=300,
                        n_hvg=min(n_hvg, len(all_genes)),
                        tfidf_min=0.5,
                        inner_iter=inner_iter)
                    mae, pearson = _gt_metrics(rho, gt, human_mask)
                    decontx_results.append({
                        "n_topics": n_topics, "n_hvg": n_hvg,
                        "inner_iter": inner_iter,
                        "gt_mae": mae, "gt_pearson": pearson,
                        "rho_mean": float(rho.mean()), "rho_std": float(rho.std()),
                    })
                except Exception as e:
                    print(f"    FAILED: {e}")
    df_decontx = pd.DataFrame(decontx_results).sort_values("gt_mae")
    all_results["upg-decontx"] = df_decontx
    best_params["upg-decontx"] = df_decontx.iloc[0].to_dict()
    print(f"    Best → n_topics={int(df_decontx.iloc[0]['n_topics'])}  "
          f"n_hvg={int(df_decontx.iloc[0]['n_hvg'])}  "
          f"inner_iter={int(df_decontx.iloc[0]['inner_iter'])}  "
          f"GT-MAE={df_decontx.iloc[0]['gt_mae']:.4f}pp")

    # ── upg-genehet ───────────────────────────────────────────────────────────
    print("\n  [upg-genehet] Grid search ...")
    genehet_grid = {
        "n_topics": [2, 5, 10],
        "n_hvg":    [500, 1000, 2000],
    }
    genehet_results = []
    total = len(genehet_grid["n_topics"]) * len(genehet_grid["n_hvg"])
    done  = 0
    for n_topics in genehet_grid["n_topics"]:
        for n_hvg in genehet_grid["n_hvg"]:
            done += 1
            print(f"  [{done}/{total}] n_topics={n_topics} n_hvg={n_hvg} ...")
            try:
                _, t, c, rho = _pipe_upg_genehet(
                    tod, toc, all_genes, bc_raw, all_cells, clusters,
                    n_topics=n_topics, n_iter=300,
                    n_hvg=min(n_hvg, len(all_genes)),
                    tfidf_min=0.5)
                mae, pearson = _gt_metrics(rho, gt, human_mask)
                genehet_results.append({
                    "n_topics": n_topics, "n_hvg": n_hvg,
                    "gt_mae": mae, "gt_pearson": pearson,
                    "rho_mean": float(rho.mean()), "rho_std": float(rho.std()),
                })
            except Exception as e:
                print(f"    FAILED: {e}")
    df_genehet = pd.DataFrame(genehet_results).sort_values("gt_mae")
    all_results["upg-genehet"] = df_genehet
    best_params["upg-genehet"] = df_genehet.iloc[0].to_dict()
    print(f"    Best → n_topics={int(df_genehet.iloc[0]['n_topics'])}  "
          f"n_hvg={int(df_genehet.iloc[0]['n_hvg'])}  "
          f"GT-MAE={df_genehet.iloc[0]['gt_mae']:.4f}pp")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  BEST PARAMS PER PIPELINE (hgmm, optimized by GT-MAE)")
    print("=" * 80)
    for pipeline, params in best_params.items():
        print(f"\n  {pipeline}:")
        for k, v in params.items():
            if k not in ("gt_mae", "gt_pearson", "rho_mean", "rho_std"):
                print(f"    {k} = {v}")
        if "gt_mae" in params:
            print(f"    → GT-MAE = {params['gt_mae']:.4f}pp  "
                  f"GT-r = {params.get('gt_pearson', float('nan')):.4f}")
    print("=" * 80)

    return best_params, all_results



# ── Fetal liver: composite scoring helper ─────────────────────────────────────

def benchmark_fetal_liver(skip_decontx=False) -> List[BenchmarkEntry]:
    print("\n  [fetal_liver] Loading ...")
    mat, bc_raw, gene_names = _mex_v2(os.path.join(FETAL_DIR, "GRCh38"))
    mat, gene_names = _dedup_genes(mat, gene_names)
    barcodes = [b.replace("-1", "") for b in bc_raw]
    meta     = pd.read_csv(os.path.join(FETAL_DIR, "FCAImmP7352195.csv"))
    meta["Barcodes"]    = meta["Barcodes"].str.strip('"')
    meta["Cell.Labels"] = meta["Cell.Labels"].str.strip('"').str.strip()
    clusters = (meta.set_index("Barcodes")
                    .reindex(barcodes)["Cell.Labels"].fillna("Unknown"))
    clusters.index = barcodes

    ery_mask    = clusters.str.contains("Erythroid|erythroid", na=False).values
    non_ery_mat = mat[:, ~ery_mask]
    agg         = np.asarray(non_ery_mat.sum(axis=1)).flatten().astype(float)
    soup_df     = pd.DataFrame({"counts": agg, "est": agg / (agg.sum() + 1e-10)},
                               index=gene_names)

    cls_arr    = clusters.values
    cell_types = clusters.values
    batch      = np.where(ery_mask, "erythroid", "non-erythroid")
    m2_markers = ["HBB", "HBA2", "HBA1", "ALB", "APOA2", "CD3D", "CD79A", "LYZ"]
    ery_labels = {l for l in np.unique(cls_arr) if "erythroid" in l.lower()}

    entries = []
    pipelines = ["baseline", "upg-auto", "upg-doublet", "upg-iterative"]
    if not skip_decontx:
        pipelines += ["upg-decontx", "upg-genehet"]

    for name in pipelines:
        print(f"  [fetal_liver] Running {name} ...")
        if name == "baseline":
            _, t, c, rho = _pipe_baseline_from_mat_with_soup(
                mat, gene_names, barcodes, clusters, soup_df)
        elif name == "upg-auto":
            _, t, c, rho = _pipe_upg_auto_from_mat(
                mat, mat, gene_names, barcodes, clusters, soup_df=soup_df)
        elif name == "upg-doublet":
            _, t, c, rho = _pipe_upg_doublet(
                mat, mat, gene_names, barcodes, barcodes, clusters,
                soup_df=soup_df)
        elif name == "upg-iterative":
            _, t, c, rho = _pipe_upg_iterative(
                mat, mat, gene_names, barcodes, barcodes, clusters,
                n_iter=2, soup_df=soup_df)
        elif name == "upg-decontx":
            _, t, c, rho = _pipe_upg_decontx(
                mat, mat, gene_names, barcodes, barcodes,
                clusters, n_iter=500, n_hvg=3000, soup_df=soup_df)
        elif name == "upg-genehet":
            _, t, c, rho = _pipe_upg_genehet(
                mat, mat, gene_names, barcodes, barcodes,
                clusters, n_iter=100, n_hvg=3000, soup_df=soup_df)

        e = BenchmarkEntry(dataset="fetal_liver", pipeline=name,
                           n_cells=len(barcodes),
                           rho_mean=float(rho.mean()), rho_std=float(rho.std()))
        r2 = _run_m2(t, c, cls_arr, m2_markers, gene_names)
        r3 = _run_m3(t, c)
        r4 = _run_m4(t, c, batch)
        r5 = _run_m5(t, c, cell_types, gene_names, ery_labels=ery_labels)
        r6 = _run_m6(t, c, cls_arr)
        r7 = _run_m7(t, c, cls_arr, gene_names)
        r8 = _run_m8(t, c, cls_arr, m2_markers, gene_names)
        entries.append(_fill(e, r2, r3, r4, r5, r6=r6, r7=r7, r8=r8))
    return entries





def _make_clusters_pca_kmeans(toc, n_clusters=10, n_pcs=30, random_state=42):
    """Generate cluster labels via sparse log-norm → SVD → KMeans."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    mat = toc.T.tocsr().astype(float) if scipy.sparse.issparse(toc) else scipy.sparse.csr_matrix(np.asarray(toc.T, dtype=float))
    lib = np.asarray(mat.sum(axis=1)).flatten()
    lib[lib == 0] = 1.0
    mat = scipy.sparse.diags(1e4 / lib) @ mat
    mat.data = np.log1p(mat.data)

    n_pcs = min(n_pcs, mat.shape[1] - 1, mat.shape[0] - 1)
    if n_pcs < 2:
        return np.array(["0"] * mat.shape[0], dtype=str)
    svd = TruncatedSVD(n_components=n_pcs, random_state=random_state)
    emb = svd.fit_transform(mat)
    emb = normalize(emb)

    k = min(n_clusters, emb.shape[0])
    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(emb)
    return labels.astype(str)



# def _gse218853_gt_metrics(rho_arr, gt_df, cell_names):
#     norm_cells = pd.Index(
#         [str(c).strip().strip('"').strip("'").removesuffix("-1") for c in cell_names],
#         name="barcode",
#     )
#     rho_series = pd.Series(np.asarray(rho_arr, dtype=float), index=norm_cells)
#     joined = gt_df[["rho_gt"]].join(rho_series.rename("rho_pred"), how="inner")

#     # ── Diagnostic block — remove after debugging ──────────────────────────
#     print(f"\n  [GT DEBUG] Total cells       : {len(cell_names):,}")
#     print(f"  [GT DEBUG] GT rows available : {len(gt_df):,}")
#     print(f"  [GT DEBUG] Matched (inner)   : {len(joined):,}")
#     if len(joined) > 0:
#         print(f"  [GT DEBUG] rho_pred  — mean={joined['rho_pred'].mean():.4f}  "
#               f"std={joined['rho_pred'].std():.4f}  "
#               f"range=[{joined['rho_pred'].min():.4f}, {joined['rho_pred'].max():.4f}]")
#         print(f"  [GT DEBUG] rho_gt    — mean={joined['rho_gt'].mean():.4f}  "
#               f"std={joined['rho_gt'].std():.4f}  "
#               f"range=[{joined['rho_gt'].min():.4f}, {joined['rho_gt'].max():.4f}]")
#         # barcode sample check
#         print(f"  [GT DEBUG] Sample cell barcodes (first 3): "
#               f"{list(norm_cells[:3])}")
#         print(f"  [GT DEBUG] Sample GT barcodes   (first 3): "
#               f"{list(gt_df.index[:3])}")
#     else:
#         print(f"  [GT DEBUG] WARNING: zero overlap!")
#         print(f"  [GT DEBUG] Sample cell barcodes (first 3): "
#               f"{list(norm_cells[:3])}")
#         print(f"  [GT DEBUG] Sample GT barcodes   (first 3): "
#               f"{list(gt_df.index[:3])}")
#     # ── End diagnostic ─────────────────────────────────────────────────────

#     if joined.empty:
#         raise ValueError("No overlap between predicted rho values and GT barcodes.")

#     mae, pearson = _gt_metrics(
#         joined["rho_pred"].values,
#         joined["rho_gt"].values,
#     )
#     return mae, pearson

def _load_rep1_zenodo_gt_context(gt_path=None):
    try:
        from .rep1_zenodo_utils import load_rep1_zenodo_gt_aligned, load_rep1_zenodo_sample
    except ImportError:
        from rep1_zenodo_utils import load_rep1_zenodo_gt_aligned, load_rep1_zenodo_sample

    sc = load_rep1_zenodo_sample(REP1_ZENODO_DIR, verbose=False)
    gt_df = load_rep1_zenodo_gt_aligned(REP1_ZENODO_DIR, sc.cells, gt_path=gt_path)
    if gt_df.empty:
        raise ValueError("GT loaded, but no rep1_Zenodo GT barcodes matched the filtered H5 barcodes.")

    n_clusters = min(14, max(6, int(np.sqrt(len(sc.cells)) / 12)))
    clusters = pd.Series(
        _make_clusters_pca_kmeans(sc.toc, n_clusters=n_clusters, n_pcs=30),
        index=sc.cells,
        dtype=str,
    )
    return {
        "tod": sc.tod,
        "toc": sc.toc,
        "gene_names": list(sc.genes),
        "cell_names": list(sc.cells),
        "bc_raw": list(sc.n_drop_umis.index),
        "clusters": clusters,
        "gt_df": gt_df,
    }


def tune_all_pipelines_rep1_zenodo_gt(gt_path=None):
    """Grid search all pipelines on rep1_Zenodo using GT-MAE on CAST GT barcodes."""
    print("\n  [rep1_zenodo_gt tune] Loading rep1 + GT ...")
    ctx = _load_rep1_zenodo_gt_context(gt_path=gt_path)
    tod = ctx["tod"]
    toc = ctx["toc"]
    gene_names = ctx["gene_names"]
    cell_names = ctx["cell_names"]
    bc_raw = ctx["bc_raw"]
    clusters = ctx["clusters"]
    gt_df = ctx["gt_df"]

    # print(f"  [rep1_zenodo_gt tune] {len(cell_names):,} filtered cells  {len(gene_names):,} genes")
    # print(f"  [rep1_zenodo_gt tune] GT cells matched: {len(gt_df):,}")

    best_params = {}
    all_results = {}

    def _finalize_grid(pipeline, results, summary_fields):
        df = pd.DataFrame(results)
        if df.empty or "gt_mae" not in df.columns:
            print(f"    No valid parameter setting found for {pipeline}.")
            all_results[pipeline] = df
            return None
        df = df.sort_values("gt_mae")
        all_results[pipeline] = df
        best = df.iloc[0].to_dict()
        best_params[pipeline] = best
        summary = "  ".join(f"{field}={best[field]}" for field in summary_fields)
        print(f"    Best → {summary}  GT-MAE={best['gt_mae']:.4f}pp")
        return best

    print("\n  [baseline] Running ...")
    try:
        _, t, c, rho = _pipe_baseline_from_mat(tod, toc, gene_names, cell_names, clusters)
        mae, pearson = _gse218853_gt_metrics(rho, gt_df, cell_names)
        best_params["baseline"] = {}
        print(f"    GT-MAE={mae:.4f}pp  GT-r={pearson:.4f}")
    except Exception as e:
        print(f"    FAILED: {e}")

    print("\n  [upg-auto] Grid search ...")
    auto_grid = {
        "tfidf_min": [0.1, 0.5],
        "contamination_range": [(0.01, 0.10), (0.01, 0.20)],
        "soup_quantile": [0.50, 0.90],
    }
    auto_results = []
    total = len(auto_grid["tfidf_min"]) * len(auto_grid["contamination_range"]) * len(auto_grid["soup_quantile"])
    done = 0
    for tfidf_min in auto_grid["tfidf_min"]:
        for cont_range in auto_grid["contamination_range"]:
            for soup_quantile in auto_grid["soup_quantile"]:
                done += 1
                print(f"  [{done}/{total}] tfidf_min={tfidf_min}  cont_range={cont_range}  soup_q={soup_quantile} ...")
                try:
                    _, t, c, rho = _pipe_upg_auto_from_mat(
                        tod, toc, gene_names, cell_names, clusters,
                        bc_raw=bc_raw,
                        tfidf_min=tfidf_min,
                        contamination_range=cont_range,
                        soup_quantile=soup_quantile,
                    )
                    mae, pearson = _gse218853_gt_metrics(rho, gt_df, cell_names)
                    auto_results.append({
                        "tfidf_min": tfidf_min,
                        "contamination_range": str(cont_range),
                        "soup_quantile": soup_quantile,
                        "gt_mae": mae,
                        "gt_pearson": pearson,
                        "rho_mean": float(np.mean(rho)),
                        "rho_std": float(np.std(rho)),
                    })
                except Exception as e:
                    print(f"    FAILED: {e}")
    _finalize_grid("upg-auto", auto_results, ["tfidf_min", "contamination_range", "soup_quantile"])

    print("\n  [upg-doublet] Grid search ...")
    doublet_grid = {
        "tfidf_min": [0.1, 0.5],
        "contamination_range": [(0.01, 0.10), (0.01, 0.20)],
        "soup_quantile": [0.50, 0.90],
    }
    doublet_results = []
    total = len(doublet_grid["tfidf_min"]) * len(doublet_grid["contamination_range"]) * len(doublet_grid["soup_quantile"])
    done = 0
    for tfidf_min in doublet_grid["tfidf_min"]:
        for cont_range in doublet_grid["contamination_range"]:
            for soup_quantile in doublet_grid["soup_quantile"]:
                done += 1
                print(f"  [{done}/{total}] tfidf_min={tfidf_min}  cont_range={cont_range}  soup_q={soup_quantile} ...")
                try:
                    _, t, c, rho = _pipe_upg_doublet(
                        tod, toc, gene_names, bc_raw, cell_names, clusters,
                        tfidf_min=tfidf_min,
                        contamination_range=cont_range,
                        soup_quantile=soup_quantile,
                    )
                    mae, pearson = _gse218853_gt_metrics(rho, gt_df, cell_names)
                    doublet_results.append({
                        "tfidf_min": tfidf_min,
                        "contamination_range": str(cont_range),
                        "soup_quantile": soup_quantile,
                        "gt_mae": mae,
                        "gt_pearson": pearson,
                        "rho_mean": float(np.mean(rho)),
                        "rho_std": float(np.std(rho)),
                    })
                except Exception as e:
                    print(f"    FAILED: {e}")
    _finalize_grid("upg-doublet", doublet_results, ["tfidf_min", "contamination_range", "soup_quantile"])

    print("\n  [upg-iterative] Grid search ...")
    iter_grid = {
        "tfidf_min": [0.1, 0.5],
        "contamination_range": [(0.01, 0.10), (0.01, 0.20)],
        "soup_quantile": [0.50, 0.90],
        "n_iter": [2, 3],
    }
    iter_results = []
    total = len(iter_grid["tfidf_min"]) * len(iter_grid["contamination_range"]) * len(iter_grid["soup_quantile"]) * len(iter_grid["n_iter"])
    done = 0
    for tfidf_min in iter_grid["tfidf_min"]:
        for cont_range in iter_grid["contamination_range"]:
            for soup_quantile in iter_grid["soup_quantile"]:
                for n_iter in iter_grid["n_iter"]:
                    done += 1
                    print(f"  [{done}/{total}] tfidf_min={tfidf_min}  cont_range={cont_range}  soup_q={soup_quantile}  n_iter={n_iter} ...")
                    try:
                        _, t, c, rho = _pipe_upg_iterative(
                            tod, toc, gene_names, bc_raw, cell_names, clusters,
                            tfidf_min=tfidf_min,
                            contamination_range=cont_range,
                            soup_quantile=soup_quantile,
                            n_iter=n_iter,
                        )
                        mae, pearson = _gse218853_gt_metrics(rho, gt_df, cell_names)
                        iter_results.append({
                            "tfidf_min": tfidf_min,
                            "contamination_range": str(cont_range),
                            "soup_quantile": soup_quantile,
                            "n_iter": n_iter,
                            "gt_mae": mae,
                            "gt_pearson": pearson,
                            "rho_mean": float(np.mean(rho)),
                            "rho_std": float(np.std(rho)),
                        })
                    except Exception as e:
                        print(f"    FAILED: {e}")
    _finalize_grid("upg-iterative", iter_results, ["tfidf_min", "contamination_range", "soup_quantile", "n_iter"])

    print("\n  [upg-decontx] Grid search ...")
    decontx_grid = {
        "n_topics": [10, 15],
        "n_hvg": [2000, 3000],
        "inner_iter": [1, 2],
    }
    decontx_results = []
    total = len(decontx_grid["n_topics"]) * len(decontx_grid["n_hvg"]) * len(decontx_grid["inner_iter"])
    done = 0
    for n_topics in decontx_grid["n_topics"]:
        for n_hvg in decontx_grid["n_hvg"]:
            for inner_iter in decontx_grid["inner_iter"]:
                done += 1
                print(f"  [{done}/{total}] n_topics={n_topics}  n_hvg={n_hvg}  inner_iter={inner_iter} ...")
                try:
                    _, t, c, rho = _pipe_upg_decontx(
                        tod, toc, gene_names, bc_raw, cell_names, clusters,
                        n_topics=n_topics,
                        n_iter=200,
                        n_hvg=n_hvg,
                        tfidf_min=0.5,
                        inner_iter=inner_iter,
                    )
                    mae, pearson = _gse218853_gt_metrics(rho, gt_df, cell_names)
                    decontx_results.append({
                        "n_topics": n_topics,
                        "n_hvg": n_hvg,
                        "inner_iter": inner_iter,
                        "gt_mae": mae,
                        "gt_pearson": pearson,
                        "rho_mean": float(np.mean(rho)),
                        "rho_std": float(np.std(rho)),
                    })
                except Exception as e:
                    print(f"    FAILED: {e}")
    _finalize_grid("upg-decontx", decontx_results, ["n_topics", "n_hvg", "inner_iter"])

    print("\n  [upg-genehet] Grid search ...")
    genehet_grid = {
        "n_topics": [10, 15],
        "n_hvg": [2000, 3000],
    }
    genehet_results = []
    total = len(genehet_grid["n_topics"]) * len(genehet_grid["n_hvg"])
    done = 0
    for n_topics in genehet_grid["n_topics"]:
        for n_hvg in genehet_grid["n_hvg"]:
            done += 1
            print(f"  [{done}/{total}] n_topics={n_topics}  n_hvg={n_hvg} ...")
            try:
                _, t, c, rho = _pipe_upg_genehet(
                    tod, toc, gene_names, bc_raw, cell_names, clusters,
                    n_topics=n_topics,
                    n_iter=200,
                    n_hvg=n_hvg,
                    tfidf_min=0.5,
                )
                mae, pearson = _gse218853_gt_metrics(rho, gt_df, cell_names)
                genehet_results.append({
                    "n_topics": n_topics,
                    "n_hvg": n_hvg,
                    "gt_mae": mae,
                    "gt_pearson": pearson,
                    "rho_mean": float(np.mean(rho)),
                    "rho_std": float(np.std(rho)),
                })
            except Exception as e:
                print(f"    FAILED: {e}")
    _finalize_grid("upg-genehet", genehet_results, ["n_topics", "n_hvg"])

    print("\n" + "=" * 80)
    print("  BEST PARAMS PER PIPELINE (rep1_zenodo_gt, optimized by GT-MAE)")
    print("=" * 80)
    for pipeline, params in best_params.items():
        print(f"\n  {pipeline}:")
        for k, v in params.items():
            if k not in ("gt_mae", "gt_pearson", "rho_mean", "rho_std"):
                print(f"    {k} = {v}")
        if "gt_mae" in params:
            print(f"    → GT-MAE = {params['gt_mae']:.4f}pp  GT-r = {params.get('gt_pearson', float('nan')):.4f}")
    print("=" * 80)
    return best_params, all_results


def benchmark_rep1_zenodo_gt(skip_decontx=False, tuned_params=None, gt_path=None) -> List[BenchmarkEntry]:
    """Run local rep1_Zenodo benchmark with GT-evaluation on matched CAST GT barcodes."""
    print("\n  [rep1_zenodo_gt] Loading rep1 + GT ...")
    ctx = _load_rep1_zenodo_gt_context(gt_path=gt_path)
    tod = ctx["tod"]
    toc = ctx["toc"]
    gene_names = ctx["gene_names"]
    cell_names = ctx["cell_names"]
    bc_raw = ctx["bc_raw"]
    clusters = ctx["clusters"]
    gt_df = ctx["gt_df"]
    cls_arr = clusters.reindex(cell_names).fillna("0").values

    tp = tuned_params or {}

    def _p(pipeline, key, default):
        val = tp.get(pipeline, {}).get(key, default)
        if key == "contamination_range" and isinstance(val, str):
            val = tuple(float(x) for x in val.strip("()").split(","))
        return val

    entries = []
    pipelines = ["baseline", "upg-auto", "upg-doublet", "upg-iterative"]
    if not skip_decontx:
        pipelines += ["upg-decontx", "upg-genehet"]

    for name in pipelines:
        src = "tuned" if name in tp else "default"
        print(f"  [rep1_zenodo_gt] Running {name} ({src} params) ...")
        try:
            if name == "baseline":
                _, t, c, rho = _pipe_baseline_from_mat(tod, toc, gene_names, cell_names, clusters)
            elif name == "upg-auto":
                _, t, c, rho = _pipe_upg_auto_from_mat(
                    tod, toc, gene_names, cell_names, clusters,
                    bc_raw=bc_raw,
                    tfidf_min=_p("upg-auto", "tfidf_min", 0.5),
                    contamination_range=_p("upg-auto", "contamination_range", (0.01, 0.10)),
                    soup_quantile=_p("upg-auto", "soup_quantile", 0.90),
                )
            elif name == "upg-doublet":
                _, t, c, rho = _pipe_upg_doublet(
                    tod, toc, gene_names, bc_raw, cell_names, clusters,
                    tfidf_min=_p("upg-doublet", "tfidf_min", 0.5),
                    contamination_range=_p("upg-doublet", "contamination_range", (0.01, 0.10)),
                    soup_quantile=_p("upg-doublet", "soup_quantile", 0.90),
                )
            elif name == "upg-iterative":
                _, t, c, rho = _pipe_upg_iterative(
                    tod, toc, gene_names, bc_raw, cell_names, clusters,
                    n_iter=int(_p("upg-iterative", "n_iter", 2)),
                    tfidf_min=_p("upg-iterative", "tfidf_min", 0.5),
                    contamination_range=_p("upg-iterative", "contamination_range", (0.01, 0.10)),
                    soup_quantile=_p("upg-iterative", "soup_quantile", 0.90),
                )
            elif name == "upg-decontx":
                _, t, c, rho = _pipe_upg_decontx(
                    tod, toc, gene_names, bc_raw, cell_names, clusters,
                    n_topics=int(_p("upg-decontx", "n_topics", 15)),
                    n_iter=200,
                    n_hvg=int(_p("upg-decontx", "n_hvg", 3000)),
                    tfidf_min=_p("upg-decontx", "tfidf_min", 0.5),
                    inner_iter=int(_p("upg-decontx", "inner_iter", 1)),
                )
            elif name == "upg-genehet":
                _, t, c, rho = _pipe_upg_genehet(
                    tod, toc, gene_names, bc_raw, cell_names, clusters,
                    n_topics=int(_p("upg-genehet", "n_topics", 15)),
                    n_iter=200,
                    n_hvg=int(_p("upg-genehet", "n_hvg", 3000)),
                    tfidf_min=_p("upg-genehet", "tfidf_min", 0.5),
                )
        except Exception as exc:
            print(f"    FAILED: {exc}")
            continue

        e = BenchmarkEntry(
            dataset="rep1_zenodo_gt",
            pipeline=name,
            n_cells=len(cell_names),
            rho_mean=float(np.mean(rho)),
            rho_std=float(np.std(rho)),
        )
        gt_tuple = _gse218853_gt_metrics(rho, gt_df, cell_names)
        r3 = _run_m3(t, c)
        r6 = _run_m6(t, c, cls_arr)
        r7 = _run_m7(t, c, cls_arr, gene_names)
        entries.append(_fill(e, None, r3, None, None, gt=gt_tuple, r6=r6, r7=r7))


    return entries

# ── DataFrame export ──────────────────────────────────────────────────────────

def entries_to_dataframe(entries: List[BenchmarkEntry]) -> "pd.DataFrame":
    from dataclasses import asdict
    return pd.DataFrame([asdict(e) for e in entries])


# ── Print ─────────────────────────────────────────────────────────────────────

_NA  = "  N/A  "
_W   = 150

def _fmt(v, fmt=".3f", suffix=""):
    return f"{v:{fmt}}{suffix}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else _NA.strip()

def _rule(c="="): print(c * _W)

def _print_table(all_entries: List[BenchmarkEntry]):
    _rule()
    print("  FULL BENCHMARK — SoupX: Baseline vs Upgraded (all datasets × all pipelines × all metrics)")
    _rule()
    h = (f"  {'Dataset':<14} {'Pipeline':<14} {'n_cells':>7} {'rho_mean':>9} {'rho_std':>8}"
         f" {'M1-Fold':>8} {'M2-FC':>7} {'M3-ARI':>7} {'M4-Δent':>8}"
         f" {'M5-HBB':>8} {'GT-MAE':>8} {'GT-r':>7} {'EX-Fold':>8}"
         f" {'M6-Sil':>8} {'M7-SpDE':>8} {'M8-MkRk':>8}")
    print(h)
    _rule("-")
    prev_ds = None
    for e in all_entries:
        if prev_ds and e.dataset != prev_ds:
            _rule("-")
        prev_ds = e.dataset
        m1 = _fmt(e.m1_fold, ".2f", "×") + ("✓" if e.m1_pass else "✗") if e.m1_fold is not None else _NA.strip()
        m2 = _fmt(e.m2_fc_ratio, ".3f", "×") + ("↑" if e.m2_improved else "↓") if e.m2_fc_ratio is not None else _NA.strip()
        m3 = _fmt(e.m3_ari, ".3f")                 if e.m3_ari   is not None else _NA.strip()
        m4 = _fmt(e.m4_entropy_delta, "+.4f")       if e.m4_entropy_delta is not None else _NA.strip()
        m5 = _fmt(e.m5_pct_reduction, ".1f", "pp")  if e.m5_pct_reduction is not None else _NA.strip()
        gm = _fmt(e.gt_mae,    ".3f", "pp")          if e.gt_mae     is not None else _NA.strip()
        gp = _fmt(e.gt_pearson, ".4f")               if e.gt_pearson is not None else _NA.strip()
        ex = _fmt(e.excl_fold, ".2f", "×")           if e.excl_fold  is not None else _NA.strip()
        m6 = (_fmt(e.m6_sil_delta, "+.4f") + ("↑" if e.m6_improved else "↓")
              if e.m6_sil_delta is not None else _NA.strip())
        m7 = _fmt(e.m7_n_spurious, "d")             if e.m7_n_spurious  is not None else _NA.strip()
        m8 = (_fmt(e.m8_rank_delta, "+.4f") + ("↑" if e.m8_improved else "↓")
              if e.m8_rank_delta is not None else _NA.strip())
        print(f"  {e.dataset:<14} {e.pipeline:<14} {e.n_cells:>7,}"
              f" {e.rho_mean*100:>8.3f}% {e.rho_std*100:>7.3f}%"
              f" {m1:>9} {m2:>9} {m3:>7} {m4:>8}"
              f" {m5:>8} {gm:>8} {gp:>7} {ex:>8}"
              f" {m6:>9} {m7:>8} {m8:>9}")
    _rule()


def _print_metric_winners(all_entries: List[BenchmarkEntry]):
    _rule()
    print("  METRIC WINNERS — best pipeline per metric per dataset")
    _rule()
    datasets = list(dict.fromkeys(e.dataset for e in all_entries))
    for ds in datasets:
        es = [e for e in all_entries if e.dataset == ds]
        print(f"  {ds}:")

        def _winner(attr, higher_better=True):
            vals = [(e.pipeline, getattr(e, attr)) for e in es
                    if getattr(e, attr) is not None
                    and not (isinstance(getattr(e, attr), float) and np.isnan(getattr(e, attr)))]
            if not vals:
                return "N/A"
            return max(vals, key=lambda x: x[1] if higher_better else -x[1])[0]

        print(f"    M1 cross-species fold : {_winner('m1_fold')}")
        print(f"    M2 marker FC ratio    : {_winner('m2_fc_ratio')}")
        print(f"    M3 cluster ARI (↑=stable): {_winner('m3_ari')}")
        print(f"    M4 batch entropy Δ   : {_winner('m4_entropy_delta')}")
        print(f"    M5 HBB pp reduction  : {_winner('m5_pct_reduction')}")
        print(f"    GT ground truth MAE  : {_winner('gt_mae', higher_better=False)}")
        print(f"    EX marker exclusivity: {_winner('excl_fold')}")
        print(f"    M6 silhouette delta  : {_winner('m6_sil_delta')}")
        print(f"    M7 spurious DE genes : {_winner('m7_n_spurious', higher_better=False)}")
        print(f"    M8 marker rank delta : {_winner('m8_rank_delta')}")
    _rule()


# ── CLI ───────────────────────────────────────────────────────────────────────

_AVAIL = {
    "toy_pbmc":      lambda: os.path.isdir(TOY_DIR) and os.path.isdir(TOY_BASE),
    "pbmc_10k":      lambda: os.path.isdir(PBMC10K_DIR) and os.path.isfile(PBMC10K_CLU),
    "hgmm":          lambda: os.path.isdir(os.path.join(HGMM_DIR,   "raw_gene_bc_matrices")),
    "fetal_liver":   lambda: os.path.isdir(os.path.join(FETAL_DIR, "GRCh38")),
    "rep1_zenodo_gt": lambda: (
        os.path.isfile(os.path.join(REP1_ZENODO_DIR, "raw_feature_bc_matrix.h5"))
        and os.path.isfile(os.path.join(REP1_ZENODO_DIR, "filtered_feature_bc_matrix.h5"))
    ),

}
_RUNNERS = {
    "toy_pbmc":      benchmark_toy_pbmc,
    "pbmc_10k":      benchmark_pbmc10k,
    "hgmm":          benchmark_hgmm,
    "fetal_liver":   benchmark_fetal_liver,
    "rep1_zenodo_gt": benchmark_rep1_zenodo_gt,
}


def get_corrected_matrix_toy_pbmc(pipeline="upg-auto"):
    """Return (toc, cor, gene_names, cls_arr) for given pipeline on toy_pbmc."""
    tod, bc_raw, gn = _mex_v2(os.path.join(TOY_DIR, "raw_gene_bc_matrices",      "GRCh38"))
    toc, bc_filt, _ = _mex_v2(os.path.join(TOY_DIR, "filtered_gene_bc_matrices", "GRCh38"))
    clusters = pd.read_csv(os.path.join(TOY_DIR, "metaData.tsv"),
                           sep="\t", index_col=0)["res.1"].astype(str)
    cls_arr = clusters.reindex(bc_filt).fillna("0").values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pipeline == "baseline":
            gn_p, t, c, _ = _pipe_baseline_from_dir(TOY_BASE, clusters)
        elif pipeline == "upg-auto":
            gn_p, t, c, _ = _pipe_upg_auto_from_dir(TOY_DIR, clusters)
        elif pipeline == "upg-doublet":
            gn_p, t, c, _ = _pipe_upg_doublet(tod, toc, gn, bc_raw, bc_filt, clusters)
        elif pipeline == "upg-iterative":
            gn_p, t, c, _ = _pipe_upg_iterative(tod, toc, gn, bc_raw, bc_filt, clusters, n_iter=2)
        elif pipeline == "upg-decontx":
            gn_p, t, c, _ = _pipe_upg_decontx(tod, toc, gn, bc_raw, bc_filt, clusters,
                                               n_iter=200, n_hvg=len(gn))
        elif pipeline == "upg-genehet":
            gn_p, t, c, _ = _pipe_upg_genehet(tod, toc, gn, bc_raw, bc_filt, clusters,
                                               n_iter=200, n_hvg=len(gn))
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")
    return t, c, gn_p, cls_arr


def get_fitted_sc_toy_pbmc():
    """Return fitted SoupChannel (post-autoEstCont) for toy_pbmc via upg-auto."""
    import SoupX as U
    clusters = pd.read_csv(os.path.join(TOY_DIR, "metaData.tsv"),
                           sep="\t", index_col=0)["res.1"].astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = U.load_10x(TOY_DIR, verbose=False)
        sc = U.set_clusters(sc, clusters.reindex(sc.cells).fillna("0"))
        sc_fit = U.auto_est_cont(sc, do_plot=False, verbose=False, force_accept=True)
    return sc_fit


def get_corrected_matrix_pbmc10k(pipeline="upg-auto"):
    """Return (toc, cor, gene_names, cls_arr) for given pipeline on pbmc_10k."""
    tod, bc_raw, gn = _mex_v3(os.path.join(PBMC10K_DIR, "raw_feature_bc_matrix"))
    toc, bc_filt, _ = _mex_v3(os.path.join(PBMC10K_DIR, "filtered_feature_bc_matrix"))
    clusters = pd.read_csv(PBMC10K_CLU).set_index("Barcode")["Cluster"].astype(str)
    cls_arr  = clusters.reindex(bc_filt).fillna("0").values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pipeline == "baseline":
            gn_p, t, c, _ = _pipe_baseline_from_dir(PBMC10K_DIR, clusters)
        elif pipeline == "upg-auto":
            gn_p, t, c, _ = _pipe_upg_auto_from_dir(PBMC10K_DIR, clusters)
        elif pipeline == "upg-doublet":
            gn_p, t, c, _ = _pipe_upg_doublet(tod, toc, gn, bc_raw, bc_filt, clusters)
        elif pipeline == "upg-iterative":
            gn_p, t, c, _ = _pipe_upg_iterative(tod, toc, gn, bc_raw, bc_filt, clusters, n_iter=2)
        elif pipeline == "upg-decontx":
            gn_p, t, c, _ = _pipe_upg_decontx(tod, toc, gn, bc_raw, bc_filt, clusters,
                                               n_iter=300, n_hvg=2000)
        elif pipeline == "upg-genehet":
            gn_p, t, c, _ = _pipe_upg_genehet(tod, toc, gn, bc_raw, bc_filt, clusters,
                                               n_iter=100, n_hvg=2000)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")
    return t, c, gn_p, cls_arr


def get_fitted_sc_pbmc10k():
    """Return fitted SoupChannel (post-autoEstCont) for pbmc_10k via upg-auto."""
    import SoupX as U
    clusters = pd.read_csv(PBMC10K_CLU).set_index("Barcode")["Cluster"].astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = U.load_10x(PBMC10K_DIR, verbose=False)
        sc = U.set_clusters(sc, clusters.reindex(sc.cells).fillna("0"))
        sc_fit = U.auto_est_cont(sc, do_plot=False, verbose=False, force_accept=True)
    return sc_fit


def get_fitted_sc_fetal_liver():
    """Return fitted SoupChannel (post-autoEstCont) for fetal_liver.

    Soup profile built from non-erythroid cells (ambient RNA proxy).
    """
    import SoupX as U
    from SoupX.soup_channel import SoupChannel as USC
    mat, bc_raw, gene_names = _mex_v2(os.path.join(FETAL_DIR, "GRCh38"))
    mat, gene_names = _dedup_genes(mat, gene_names)
    barcodes = [b.replace("-1", "") for b in bc_raw]
    meta = pd.read_csv(os.path.join(FETAL_DIR, "FCAImmP7352195.csv"))
    meta["Barcodes"]    = meta["Barcodes"].str.strip('"')
    meta["Cell.Labels"] = meta["Cell.Labels"].str.strip('"').str.strip()
    clusters = (meta.set_index("Barcodes")
                    .reindex(barcodes)["Cell.Labels"].fillna("Unknown"))
    clusters.index = barcodes
    ery_mask = clusters.str.contains("Erythroid|erythroid", na=False).values
    agg      = np.asarray(mat[:, ~ery_mask].sum(axis=1)).flatten().astype(float)
    soup_df  = pd.DataFrame({"counts": agg, "est": agg / (agg.sum() + 1e-10)},
                            index=gene_names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = USC(tod=mat, toc=mat,
                 genes=pd.Index(gene_names), cells=pd.Index(barcodes),
                 drop_barcodes=barcodes, calc_soup_profile=False)
        sc = U.set_soup_profile(sc, soup_df)
        sc = U.set_clusters(sc, clusters)
        sc_fit = U.auto_est_cont(sc, do_plot=False, verbose=False,
                                 force_accept=True, tfidf_min=0.5)
    return sc_fit


def get_corrected_matrix_hgmm(pipeline="upg-auto"):
    """Return (toc, cor, gene_names, cls_arr) for given pipeline on hgmm."""
    rh, bc_raw,  hg_names = _mex_v2(os.path.join(HGMM_DIR, "raw_gene_bc_matrices",      "hg19"))
    rm, bc_raw2, mm_names = _mex_v2(os.path.join(HGMM_DIR, "raw_gene_bc_matrices",      "mm10"))
    fh, bc_ch,  _         = _mex_v2(os.path.join(HGMM_DIR, "filtered_gene_bc_matrices", "hg19"))
    fm, bc_cm,  _         = _mex_v2(os.path.join(HGMM_DIR, "filtered_gene_bc_matrices", "mm10"))
    assert bc_raw == bc_raw2
    all_genes = hg_names + mm_names
    all_cells = list(bc_ch) + list(bc_cm)
    n_human   = len(bc_ch)
    tod       = scipy.sparse.vstack([rh, rm], format="csc")
    bc_to_idx = {b: i for i, b in enumerate(bc_raw)}
    toc       = tod[:, [bc_to_idx[b] for b in all_cells]]
    clusters  = pd.Series(["human"] * n_human + ["mouse"] * len(bc_cm), index=all_cells)
    cls_arr   = clusters.values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pipeline == "baseline":
            _, t, c, _ = _pipe_baseline_from_mat(tod, toc, all_genes, all_cells, clusters)
        elif pipeline == "upg-auto":
            _, t, c, _ = _pipe_upg_auto_from_mat(tod, toc, all_genes, all_cells, clusters,
                                                  bc_raw=bc_raw, tfidf_min=0.5)
        elif pipeline == "upg-doublet":
            _, t, c, _ = _pipe_upg_doublet(tod, toc, all_genes, bc_raw, all_cells, clusters,
                                            tfidf_min=0.5)
        elif pipeline == "upg-iterative":
            _, t, c, _ = _pipe_upg_iterative(tod, toc, all_genes, bc_raw, all_cells, clusters,
                                              n_iter=2, tfidf_min=0.5)
        elif pipeline == "upg-decontx":
            _, t, c, _ = _pipe_upg_decontx(tod, toc, all_genes, bc_raw, all_cells, clusters,
                                            n_iter=300, n_hvg=2000, tfidf_min=0.5)
        elif pipeline == "upg-genehet":
            _, t, c, _ = _pipe_upg_genehet(tod, toc, all_genes, bc_raw, all_cells, clusters,
                                            n_iter=300, n_hvg=2000, tfidf_min=0.5)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")
    return toc, c, all_genes, cls_arr


def get_corrected_matrix_fetal_liver(pipeline="upg-auto"):
    """Return (toc, cor, gene_names, cls_arr) for given pipeline on fetal_liver."""
    mat, bc_raw, gene_names = _mex_v2(os.path.join(FETAL_DIR, "GRCh38"))
    mat, gene_names = _dedup_genes(mat, gene_names)
    barcodes = [b.replace("-1", "") for b in bc_raw]
    meta     = pd.read_csv(os.path.join(FETAL_DIR, "FCAImmP7352195.csv"))
    meta["Barcodes"]    = meta["Barcodes"].str.strip('"')
    meta["Cell.Labels"] = meta["Cell.Labels"].str.strip('"').str.strip()
    clusters = (meta.set_index("Barcodes")
                    .reindex(barcodes)["Cell.Labels"].fillna("Unknown"))
    clusters.index = barcodes
    ery_mask = clusters.str.contains("Erythroid|erythroid", na=False).values
    agg      = np.asarray(mat[:, ~ery_mask].sum(axis=1)).flatten().astype(float)
    soup_df  = pd.DataFrame({"counts": agg, "est": agg / (agg.sum() + 1e-10)},
                            index=gene_names)
    cls_arr  = clusters.values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pipeline == "baseline":
            _, t, c, _ = _pipe_baseline_from_mat_with_soup(mat, gene_names, barcodes,
                                                            clusters, soup_df)
        elif pipeline == "upg-auto":
            _, t, c, _ = _pipe_upg_auto_from_mat(mat, mat, gene_names, barcodes, clusters,
                                                  soup_df=soup_df)
        elif pipeline == "upg-doublet":
            _, t, c, _ = _pipe_upg_doublet(mat, mat, gene_names, barcodes, barcodes,
                                            clusters, soup_df=soup_df)
        elif pipeline == "upg-iterative":
            _, t, c, _ = _pipe_upg_iterative(mat, mat, gene_names, barcodes, barcodes,
                                              clusters, n_iter=2, soup_df=soup_df)
        elif pipeline == "upg-decontx":
            _, t, c, _ = _pipe_upg_decontx(mat, mat, gene_names, barcodes, barcodes,
                                            clusters, n_iter=500, n_hvg=3000, soup_df=soup_df)
        elif pipeline == "upg-genehet":
            _, t, c, _ = _pipe_upg_genehet(mat, mat, gene_names, barcodes, barcodes,
                                            clusters, n_iter=100, n_hvg=3000, soup_df=soup_df)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")
    return mat, c, gene_names, cls_arr


def get_corrected_matrix_rep1_zenodo_gt(pipeline="upg-auto"):
    """Return (toc, cor, gene_names, cls_arr) for given pipeline on rep1_zenodo_gt."""
    ctx        = _load_rep1_zenodo_gt_context()
    tod        = ctx["tod"]
    toc        = ctx["toc"]
    gene_names = ctx["gene_names"]
    cell_names = ctx["cell_names"]
    bc_raw     = ctx["bc_raw"]
    clusters   = ctx["clusters"]
    cls_arr    = clusters.reindex(cell_names).fillna("0").values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pipeline == "baseline":
            _, t, c, _ = _pipe_baseline_from_mat(tod, toc, gene_names, cell_names, clusters)
        elif pipeline == "upg-auto":
            _, t, c, _ = _pipe_upg_auto_from_mat(tod, toc, gene_names, cell_names, clusters,
                                                  bc_raw=bc_raw,
                                                  contamination_range=(0.01, 0.10),
                                                  soup_quantile=0.90)
        elif pipeline == "upg-doublet":
            _, t, c, _ = _pipe_upg_doublet(tod, toc, gene_names, bc_raw, cell_names, clusters,
                                            contamination_range=(0.01, 0.10),
                                            soup_quantile=0.90)
        elif pipeline == "upg-iterative":
            _, t, c, _ = _pipe_upg_iterative(tod, toc, gene_names, bc_raw, cell_names, clusters,
                                              n_iter=2,
                                              contamination_range=(0.01, 0.10),
                                              soup_quantile=0.90)
        elif pipeline == "upg-decontx":
            _, t, c, _ = _pipe_upg_decontx(tod, toc, gene_names, bc_raw, cell_names, clusters,
                                            n_topics=15, n_iter=200, n_hvg=3000)
        elif pipeline == "upg-genehet":
            _, t, c, _ = _pipe_upg_genehet(tod, toc, gene_names, bc_raw, cell_names, clusters,
                                            n_topics=15, n_iter=200, n_hvg=3000)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")
    return toc, c, gene_names, cls_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(_AVAIL), default=None)
    parser.add_argument("--skip-decontx", action="store_true",
                        help="Skip DecontX pipeline (faster)")
    parser.add_argument("--save-csv", default=None, metavar="PATH",
                        help="Save benchmark results to this CSV path")
    args = parser.parse_args()

    _rule()
    print("  SoupX Full Benchmark")
    print(f"  DecontX: {'DISABLED (--skip-decontx)' if args.skip_decontx else 'ENABLED'}")
    _rule()
    print("  Dataset availability:")
    for k, check in _AVAIL.items():
        print(f"    {'✓' if check() else '✗'}  {k}")
    print()

    to_run = args.datasets or [k for k, c in _AVAIL.items() if c()]
    all_entries: List[BenchmarkEntry] = []



    # ── hgmm : tune first, then benchmark with winning params ─────────────────
    hgmm_tuned_params = None
    if "hgmm" in to_run and _AVAIL["hgmm"]():
        print("\n  ── hgmm: running grid search before benchmark ──")
        try:
            hgmm_tuned_params, _ = tune_all_pipelines_hgmm()
        except Exception:
            import traceback
            print("\n  WARNING: hgmm tuning failed — falling back to default params")
            traceback.print_exc()


    rep1_zenodo_gt_tuned_params = None
    if "rep1_zenodo_gt" in to_run and _AVAIL["rep1_zenodo_gt"]():
        print("\n  ── rep1_zenodo_gt: running grid search before benchmark ──")
        try:
            rep1_zenodo_gt_tuned_params, _ = tune_all_pipelines_rep1_zenodo_gt()
        except Exception:
            import traceback
            print("\n  WARNING: rep1_zenodo_gt tuning failed — falling back to default params")
            traceback.print_exc()



    # ── Run benchmarks ─────────────────────────────────────────────────────────
    for key in to_run:
        if not _AVAIL.get(key, lambda: False)():
            print(f"  SKIP {key} — data not found")
            continue
        try:

            if key == "hgmm":
                entries = benchmark_hgmm(
                    skip_decontx=args.skip_decontx,
                    tuned_params=hgmm_tuned_params)
            elif key == "rep1_zenodo_gt":
                entries = benchmark_rep1_zenodo_gt(
                    skip_decontx=args.skip_decontx,
                    tuned_params=rep1_zenodo_gt_tuned_params)

            else:
                # toy_pbmc, fetal_liver — no tuning
                entries = _RUNNERS[key](skip_decontx=args.skip_decontx)
            all_entries.extend(entries)
        except Exception:
            import traceback
            print(f"\n  ERROR in {key}:")
            traceback.print_exc()

    if all_entries:
        _print_table(all_entries)
        _print_metric_winners(all_entries)
        df = entries_to_dataframe(all_entries)
        save_path = args.save_csv if args.save_csv else os.path.join(REPO_ROOT, "results_raw.csv")
        df.to_csv(save_path, index=False)
        print(f"\n  Results saved to: {save_path}")


if __name__ == "__main__":
    main()
