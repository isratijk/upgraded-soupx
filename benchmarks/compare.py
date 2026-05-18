#!/usr/bin/env python3
"""
compare.py — Baseline soupx vs Upgraded SoupX: full pipeline comparison.

Baseline  : auto_est_cont → adjust_counts (unchanged baseline code)
Upgraded  : DecontX (HGMM, fetal liver) / auto_est_cont (toy, pbmc10k) + full upgraded pipeline

Datasets (no mouse brain)
--------
  toy_pbmc     — in-repo toy PBMC, regression / soup profile check
  pbmc_10k     — 10k PBMC v3, real-world auto_est_cont
  hgmm         — barnyard hgmm_1k: EXACT per-cell ground truth (minority species)
  fetal_liver  — E-MTAB-7407: erythroid vs non-erythroid rho contrast

Usage
-----
    python compare.py
    python compare.py --datasets hgmm fetal_liver
    python compare.py --datasets toy_pbmc pbmc_10k
"""

import argparse
import gzip
import io
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, REPO_ROOT)
_BASELINE_SRC = os.path.join(REPO_ROOT, "baseline")
if _BASELINE_SRC not in sys.path:
    sys.path.insert(0, _BASELINE_SRC)

# ── Dataset paths ─────────────────────────────────────────────────────────────

DATASETS     = os.path.join(REPO_ROOT, "datasets")
TOY_DIR_UPG  = os.path.join(DATASETS, "toyData")
TOY_DIR_BASE = os.path.join(REPO_ROOT, "baseline", "soupx", "data", "toyData")
PBMC10K_DIR  = os.path.join(DATASETS, "pbmc_10k_v3")
PBMC10K_CLU  = os.path.join(PBMC10K_DIR, "analysis", "clustering", "graphclust", "clusters.csv")

HGMM_DIR  = os.path.join(DATASETS, "hgmm_1k")
FETAL_DIR = os.path.join(DATASETS, "E-MTAB-7407_fetal_liver", "FCAImmP7352195")


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class StandardResult:
    """toy_pbmc / pbmc_10k: auto_est_cont both sides."""
    impl:          str
    dataset:       str
    rho:           float
    top5_genes:    List[str]  = field(default_factory=list)
    top5_est:      List[float]= field(default_factory=list)
    n_cells:       int        = 0
    n_genes:       int        = 0
    counts_before: float      = 0.0
    counts_after:  float      = 0.0


@dataclass
class HgmmResult:
    """HGMM barnyard: per-cell ground truth available."""
    impl:       str
    rho_mean:   float
    rho_human:  float          # mean rho for human cells
    rho_mouse:  float          # mean rho for mouse cells
    mae:        float          # MAE vs per-cell ground truth
    mae_human:  float
    mae_mouse:  float
    pearson_r:  float          # Pearson r(rho, ground_truth)
    gt_mean:    float
    rho_values: np.ndarray     = field(default_factory=lambda: np.array([]))
    n_cells:    int            = 0


@dataclass
class FetalResult:
    """Fetal liver: erythroid vs non-erythroid rho contrast."""
    impl:        str
    rho_mean:    float
    ery_rho:     float         # mean rho erythroid cells
    non_ery_rho: float         # mean rho non-erythroid cells
    hb_in_top10: List[str]     = field(default_factory=list)
    top5_genes:  List[str]     = field(default_factory=list)
    n_cells:     int           = 0


@dataclass
class DecontXPBMCResult:
    """toy_pbmc / pbmc_10k: DecontX per-cell rho."""
    impl:          str
    dataset:       str
    rho_mean:      float
    rho_p10:       float
    rho_p50:       float
    rho_p90:       float
    n_cells:       int   = 0
    counts_before: float = 0.0
    counts_after:  float = 0.0
    marker_excl:   dict  = field(default_factory=dict)


# ── MEX loaders ───────────────────────────────────────────────────────────────

def _load_mex_v2(directory: str):
    mat = scipy.io.mmread(os.path.join(directory, "matrix.mtx")).tocsc().astype(float)
    bc  = pd.read_csv(os.path.join(directory, "barcodes.tsv"), header=None)[0].tolist()
    gdf = pd.read_csv(os.path.join(directory, "genes.tsv"), header=None, sep="\t")
    return mat, bc, gdf[1].tolist()


def _load_mex_v3(directory: str):
    def _gz(n): return os.path.join(directory, n)
    with gzip.open(_gz("matrix.mtx.gz"), "rb") as f:
        mat = scipy.io.mmread(io.BytesIO(f.read())).tocsc().astype(float)
    with gzip.open(_gz("barcodes.tsv.gz"), "rt") as f:
        bc = [l.strip() for l in f]
    with gzip.open(_gz("features.tsv.gz"), "rt") as f:
        rows = [l.strip().split("\t") for l in f]
    return mat, bc, [r[1] for r in rows]


# ── Baseline runners ──────────────────────────────────────────────────────────

def _baseline_standard(data_dir: str, clusters: pd.Series, label: str) -> StandardResult:
    import soupx as B
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = B.load_10x(data_dir)
        sc.set_clusters(clusters.reindex(sc.cell_names).fillna("0"))
        sc_fit = B.auto_est_cont(sc, do_plot=False, verbose=False, force_accept=True)
        corrected = B.adjust_counts(sc_fit, method="subtraction", verbose=0)
    rho  = float(sc_fit.meta_data["rho"].iloc[0])
    top5 = sc_fit.soup_profile.nlargest(5, "est")
    return StandardResult(
        impl="baseline", dataset=label, rho=rho,
        top5_genes=top5.index.tolist(), top5_est=top5["est"].tolist(),
        n_cells=len(sc_fit.cell_names), n_genes=len(sc_fit.gene_names),
        counts_before=float(sc_fit.toc.sum()), counts_after=float(corrected.sum()),
    )


def _baseline_hgmm(tod, toc, all_genes, all_cells, clusters,
                   human_mask, gt) -> HgmmResult:
    import soupx as B
    from soupx.soup_channel import SoupChannel as BSC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = BSC(tod=tod, toc=toc, gene_names=all_genes, cell_names=all_cells)
        sc.set_clusters(clusters.reindex(all_cells).fillna("0"))
        sc_fit = B.auto_est_cont(sc, tfidf_min=0.5, do_plot=False,
                                 verbose=False, force_accept=True)
    global_rho = float(sc_fit.meta_data["rho"].iloc[0])
    rho_arr    = np.full(len(all_cells), global_rho)   # baseline = one value for all cells
    return _hgmm_metrics("baseline", rho_arr, human_mask, gt, len(all_cells))


def _baseline_fetal(mat, gene_names, barcodes, clusters, soup_df) -> FetalResult:
    import soupx as B
    from soupx.soup_channel import SoupChannel as BSC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = BSC(tod=mat, toc=mat, gene_names=gene_names, cell_names=barcodes)
        sc.set_soup_profile(soup_df)
        sc.set_clusters(clusters.reindex(barcodes).fillna("Unknown"))
        sc_fit = B.auto_est_cont(sc, tfidf_min=0.5, do_plot=False,
                                 verbose=False, force_accept=True)
    global_rho = float(sc_fit.meta_data["rho"].iloc[0])
    rho_arr    = np.full(len(barcodes), global_rho)
    return _fetal_metrics("baseline", rho_arr, clusters.values,
                          sc_fit.soup_profile, len(barcodes))


# ── Upgraded runners ──────────────────────────────────────────────────────────

def _upgraded_standard(data_dir: str, clusters: pd.Series, label: str) -> StandardResult:
    import SoupX as U
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = U.load_10x(data_dir, verbose=False)
        sc = U.set_clusters(sc, clusters.reindex(sc.cells).fillna("0"))
        sc_fit = U.auto_est_cont(sc, do_plot=False, verbose=False, force_accept=True)
        corrected = U.adjust_counts(sc_fit, method="subtraction")
    rho  = float(sc_fit.meta_data["rho"].iloc[0])
    top5 = sc_fit.soup_profile.nlargest(5, "est")
    return StandardResult(
        impl="upgraded", dataset=label, rho=rho,
        top5_genes=top5.index.tolist(), top5_est=top5["est"].tolist(),
        n_cells=len(sc_fit.cells), n_genes=len(sc_fit.genes),
        counts_before=float(sc_fit.toc.sum()), counts_after=float(corrected.sum()),
    )


def _upgraded_hgmm_decontx(tod, toc, all_genes, all_cells, bc_raw,
                            human_mask, gt) -> HgmmResult:
    import SoupX as U
    from SoupX.soup_channel import SoupChannel as USC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = USC(tod=tod, toc=toc,
                 genes=pd.Index(all_genes), cells=pd.Index(all_cells),
                 drop_barcodes=list(bc_raw), calc_soup_profile=True)
        sc = U.set_clusters(sc, np.where(human_mask, "human", "mouse"))
        sc_out = U.run_decontx(sc, n_topics=10, n_iter=300,
                               tol_theta=1e-4, tol_param=1e-5,
                               n_hvg=2000, soup_top_q=0.9,
                               pca_init=True, verbose=False)
    rho_arr = sc_out.meta_data["rho"].values
    return _hgmm_metrics("upgraded", rho_arr, human_mask, gt, len(all_cells))


def _upgraded_fetal_decontx(mat, gene_names, barcodes, clusters, soup_df) -> FetalResult:
    import SoupX as U
    from SoupX.soup_channel import SoupChannel as USC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = USC(tod=mat, toc=mat,
                 genes=pd.Index(gene_names), cells=pd.Index(barcodes),
                 drop_barcodes=list(barcodes), calc_soup_profile=False)
        sc = U.set_soup_profile(sc, soup_df)
        sc = U.set_clusters(sc, clusters.reindex(pd.Index(barcodes)).fillna("Unknown"))
        sc_out = U.run_decontx(sc, n_topics=20, n_iter=500,
                               tol_theta=1e-4, tol_param=1e-5,
                               n_hvg=3000, soup_top_q=0.9,
                               pca_init=True, verbose=False)
    rho_arr = sc_out.meta_data["rho"].values
    return _fetal_metrics("upgraded", rho_arr, clusters.values,
                          sc_out.soup_profile, len(barcodes))


# ── Shared metric helpers ─────────────────────────────────────────────────────

def _hgmm_metrics(impl, rho_arr, human_mask, gt, n_cells) -> HgmmResult:
    mae_h = float(np.abs(rho_arr[human_mask]  - gt[human_mask]).mean())
    mae_m = float(np.abs(rho_arr[~human_mask] - gt[~human_mask]).mean())
    mae   = float(np.abs(rho_arr - gt).mean())
    r_m, g_m = rho_arr.mean(), gt.mean()
    denom    = np.linalg.norm(rho_arr - r_m) * np.linalg.norm(gt - g_m)
    pearson  = float(((rho_arr - r_m) * (gt - g_m)).sum() / (denom + 1e-12))
    return HgmmResult(
        impl=impl,
        rho_mean=float(rho_arr.mean()),
        rho_human=float(rho_arr[human_mask].mean()),
        rho_mouse=float(rho_arr[~human_mask].mean()),
        mae=mae, mae_human=mae_h, mae_mouse=mae_m,
        pearson_r=pearson,
        gt_mean=float(gt.mean()),
        rho_values=rho_arr,
        n_cells=n_cells,
    )


def _fetal_metrics(impl, rho_arr, label_arr, soup_profile, n_cells) -> FetalResult:
    ery_mask    = pd.Series(label_arr).str.contains("Erythroid|erythroid", na=False).values
    non_ery_rho = float(rho_arr[~ery_mask].mean()) if (~ery_mask).sum() > 0 else float("nan")
    ery_rho     = float(rho_arr[ery_mask].mean())  if ery_mask.sum()  > 0 else float("nan")
    hb_genes    = ["HBB", "HBA2", "HBA1", "HBD", "HBG1", "HBG2"]
    top10       = soup_profile.nlargest(10, "est")
    hb_in_top10 = [g for g in hb_genes if g in top10.index]
    top5        = soup_profile.nlargest(5, "est")
    return FetalResult(
        impl=impl, rho_mean=float(rho_arr.mean()),
        ery_rho=ery_rho, non_ery_rho=non_ery_rho,
        hb_in_top10=hb_in_top10,
        top5_genes=top5.index.tolist(),
        n_cells=n_cells,
    )


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_hgmm():
    rh, bc_raw,  hg_names = _load_mex_v2(os.path.join(HGMM_DIR, "raw_gene_bc_matrices",      "hg19"))
    rm, bc_raw2, mm_names = _load_mex_v2(os.path.join(HGMM_DIR, "raw_gene_bc_matrices",      "mm10"))
    fh, bc_ch,   _        = _load_mex_v2(os.path.join(HGMM_DIR, "filtered_gene_bc_matrices", "hg19"))
    fm, bc_cm,   _        = _load_mex_v2(os.path.join(HGMM_DIR, "filtered_gene_bc_matrices", "mm10"))
    assert bc_raw == bc_raw2

    all_genes   = hg_names + mm_names
    all_cells   = list(bc_ch) + list(bc_cm)
    n_hg        = len(hg_names)
    n_human     = len(bc_ch)

    tod         = scipy.sparse.vstack([rh, rm], format="csc")
    bc_to_idx   = {b: i for i, b in enumerate(bc_raw)}
    toc         = tod[:, [bc_to_idx[b] for b in all_cells]]

    human_mask      = np.zeros(len(all_cells), dtype=bool)
    human_mask[:n_human] = True
    human_gene_mask = np.zeros(len(all_genes), dtype=bool)
    human_gene_mask[:n_hg] = True

    clusters = pd.Series(
        ["human"] * n_human + ["mouse"] * len(bc_cm), index=all_cells
    )

    hg_umi  = np.array(toc[human_gene_mask,  :].sum(axis=0)).flatten()
    mm_umi  = np.array(toc[~human_gene_mask, :].sum(axis=0)).flatten()
    tot_umi = hg_umi + mm_umi
    gt      = np.zeros(len(all_cells))
    gt[human_mask]  = mm_umi[human_mask]  / np.maximum(tot_umi[human_mask],  1)
    gt[~human_mask] = hg_umi[~human_mask] / np.maximum(tot_umi[~human_mask], 1)

    return tod, toc, all_genes, all_cells, bc_raw, clusters, human_mask, gt


def _load_fetal():
    mat, bc_raw, gene_names = _load_mex_v2(os.path.join(FETAL_DIR, "GRCh38"))
    barcodes = [b.replace("-1", "") for b in bc_raw]
    meta     = pd.read_csv(os.path.join(FETAL_DIR, "FCAImmP7352195.csv"))
    meta["Barcodes"]    = meta["Barcodes"].str.strip('"')
    meta["Cell.Labels"] = meta["Cell.Labels"].str.strip('"').str.strip()
    clusters = (meta.set_index("Barcodes")
                    .reindex(barcodes)["Cell.Labels"]
                    .fillna("Unknown"))
    clusters.index = barcodes
    # Exclude erythroid cells from soup proxy: their genuine HBB/HBA expression
    # would inflate those genes in the soup profile, causing DecontX to over-assign
    # HBB counts in erythroid cells as contamination (true signal ≠ soup signal).
    ery_mask    = clusters.str.contains("Erythroid|erythroid", na=False).values
    non_ery_mat = mat[:, ~ery_mask]
    agg     = np.array(non_ery_mat.sum(axis=1)).flatten().astype(float)
    tot     = agg.sum()
    soup_df = pd.DataFrame({"counts": agg, "est": agg / (tot + 1e-10)},
                           index=gene_names)
    return mat, gene_names, barcodes, clusters, soup_df


# ── PBMC matrix loaders ───────────────────────────────────────────────────────

def _load_toy_matrices():
    raw_dir = os.path.join(TOY_DIR_UPG, "raw_gene_bc_matrices",      "GRCh38")
    fil_dir = os.path.join(TOY_DIR_UPG, "filtered_gene_bc_matrices", "GRCh38")
    tod, bc_raw,  gene_names = _load_mex_v2(raw_dir)
    toc, bc_filt, _          = _load_mex_v2(fil_dir)
    return tod, toc, gene_names, bc_raw, bc_filt


def _load_pbmc10k_matrices():
    raw_dir = os.path.join(PBMC10K_DIR, "raw_feature_bc_matrix")
    fil_dir = os.path.join(PBMC10K_DIR, "filtered_feature_bc_matrix")
    tod, bc_raw,  gene_names = _load_mex_v3(raw_dir)
    toc, bc_filt, _          = _load_mex_v3(fil_dir)
    return tod, toc, gene_names, bc_raw, bc_filt


# ── Marker exclusivity ────────────────────────────────────────────────────────

_PBMC_LINEAGE_MARKERS = {
    "T_cell":   ["CD3D",   "CD3E",  "CD8A"],
    "B_cell":   ["CD79A",  "MS4A1", "CD19"],
    "NK":       ["GNLY",   "NKG7"],
    "Monocyte": ["LYZ",    "CD14",  "FCGR3A"],
}


def _marker_exclusivity(toc_before, toc_after, gene_names):
    """
    Cross-lineage contamination check (sparse-safe).
    For each lineage: positive cells = top-25% by lineage-marker sum.
    Measure mean expression of OTHER lineages' markers in those cells.
    Returns per-lineage fold reduction and mean fold.
    """
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    results  = {}
    for lineage, markers in _PBMC_LINEAGE_MARKERS.items():
        present = [g for g in markers if g in gene_idx]
        if not present:
            continue
        m_idx     = [gene_idx[g] for g in present]
        scores    = np.asarray(toc_before[m_idx, :].sum(axis=0)).flatten()
        threshold = np.percentile(scores, 75)
        pos_cols  = np.where(scores >= threshold)[0]

        other_idx = [gene_idx[g]
                     for other, om in _PBMC_LINEAGE_MARKERS.items()
                     if other != lineage
                     for g in om if g in gene_idx]
        if not other_idx or len(pos_cols) == 0:
            continue

        sub_b  = toc_before[other_idx, :][:, pos_cols]
        sub_a  = toc_after[other_idx,  :][:, pos_cols]
        before = float(sub_b.mean())
        after  = float(sub_a.mean())
        results[lineage] = {
            "before": before, "after": after,
            "fold":   before / (after + 1e-10),
            "n_pos":  len(pos_cols),
        }
    mean_fold = float(np.mean([v["fold"] for v in results.values()])) if results else 0.0
    return {"per_lineage": results, "mean_fold": mean_fold}


# ── Correction helpers (for marker exclusivity) ───────────────────────────────

def _baseline_correction_matrix(data_dir, clusters):
    import soupx as B
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = B.load_10x(data_dir)
        sc.set_clusters(clusters.reindex(sc.cell_names).fillna("0"))
        sc_fit = B.auto_est_cont(sc, do_plot=False, verbose=False, force_accept=True)
        cor    = B.adjust_counts(sc_fit, method="subtraction", verbose=0)
    return list(sc_fit.gene_names), sc_fit.toc, cor


def _upgraded_autoestcont_correction_matrix(data_dir, clusters):
    import SoupX as U
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = U.load_10x(data_dir, verbose=False)
        sc = U.set_clusters(sc, clusters.reindex(sc.cells).fillna("0"))
        sc_fit = U.auto_est_cont(sc, do_plot=False, verbose=False, force_accept=True)
        cor    = U.adjust_counts(sc_fit, method="subtraction")
    return list(sc_fit.genes), sc_fit.toc, cor


# ── DecontX PBMC runner ───────────────────────────────────────────────────────

def _upgraded_pbmc_decontx(tod, toc, gene_names, bc_raw, bc_filt, clusters,
                            label, n_topics=10, n_iter=300, n_hvg=2000):
    import SoupX as U
    from SoupX.soup_channel import SoupChannel as USC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = USC(tod=tod, toc=toc,
                 genes=pd.Index(gene_names), cells=pd.Index(bc_filt),
                 drop_barcodes=list(bc_raw), calc_soup_profile=True)
        sc = U.set_clusters(sc, clusters.reindex(pd.Index(bc_filt)).fillna("0"))
        sc_out = U.run_decontx(sc, n_topics=n_topics, n_iter=n_iter,
                               tol_theta=1e-4, tol_param=1e-5,
                               n_hvg=min(n_hvg, len(gene_names)),
                               soup_top_q=0.9, pca_init=True, verbose=False)
        cor = U.adjust_counts(sc_out, method="subtraction")
    rho_arr = sc_out.meta_data["rho"].values
    excl    = _marker_exclusivity(toc, cor, gene_names)
    result  = DecontXPBMCResult(
        impl="upgraded-decontx", dataset=label,
        rho_mean=float(rho_arr.mean()),
        rho_p10=float(np.percentile(rho_arr, 10)),
        rho_p50=float(np.percentile(rho_arr, 50)),
        rho_p90=float(np.percentile(rho_arr, 90)),
        n_cells=len(bc_filt),
        counts_before=float(toc.sum()),
        counts_after=float(cor.sum()),
        marker_excl=excl,
    )
    return result, cor


# ── Print helpers ─────────────────────────────────────────────────────────────

_W = 72

def _rule(c="="): print(c * _W)
def _row(label, bv, uv, match=None):
    flag = "  ✓" if match is True else ("  ✗ DIFF" if match is False else "")
    print(f"  {label:<30}  {bv:>14}  {uv:>14}{flag}")


def _print_standard(base: StandardResult, upg: StandardResult):
    _rule()
    print(f"  Dataset : {base.dataset}")
    print(f"  Method  : Baseline → auto_est_cont   |   Upgraded → auto_est_cont (upgraded)")
    _rule()
    print(f"  {'Metric':<30}  {'Baseline':>14}  {'Upgraded':>14}  Match")
    _rule("-")
    _row("rho (global)", f"{base.rho:.4f}",  f"{upg.rho:.4f}",
         abs(base.rho - upg.rho) < 0.001)
    _row("n_cells",   str(base.n_cells),  str(upg.n_cells),  base.n_cells == upg.n_cells)
    _row("n_genes",   str(base.n_genes),  str(upg.n_genes),  base.n_genes == upg.n_genes)
    _row("counts_before", f"{base.counts_before:,.0f}", f"{upg.counts_before:,.0f}",
         abs(base.counts_before - upg.counts_before) < 1)
    rb = base.counts_before - base.counts_after
    ru = upg.counts_before  - upg.counts_after
    _row("counts_removed", f"{rb:,.0f}", f"{ru:,.0f}",
         abs(rb - ru) / (max(rb, ru) + 1) < 0.01)
    _row("  pct removed",
         f"{rb/base.counts_before*100:.2f}%",
         f"{ru/upg.counts_before*100:.2f}%")
    _rule("-")
    for i, (bg, ug, be, ue) in enumerate(
        zip(base.top5_genes, upg.top5_genes, base.top5_est, upg.top5_est), 1
    ):
        _row(f"soup gene #{i}", f"{bg} ({be:.4f})", f"{ug} ({ue:.4f})", bg == ug)
    _rule()
    print()


def _print_hgmm(base: HgmmResult, upg: HgmmResult):
    _rule()
    print("  Dataset : HGMM Barnyard (hgmm_1k)")
    print("  Method  : Baseline → auto_est_cont (global rho)   |   Upgraded → DecontX (per-cell rho)")
    print("  Ground truth: minority-species UMIs / total UMIs per cell (exact)")
    _rule()
    print(f"  {'Metric':<30}  {'Baseline':>14}  {'Upgraded':>14}  Better")
    _rule("-")
    _row("Mean rho (all cells)",
         f"{base.rho_mean*100:.3f}%", f"{upg.rho_mean*100:.3f}%")
    _row("  Human cells mean rho",
         f"{base.rho_human*100:.3f}%", f"{upg.rho_human*100:.3f}%")
    _row("  Mouse cells mean rho",
         f"{base.rho_mouse*100:.3f}%", f"{upg.rho_mouse*100:.3f}%")
    _row("Ground truth mean",
         f"{base.gt_mean*100:.3f}%", f"{upg.gt_mean*100:.3f}%")
    _rule("-")

    mae_better = upg.mae < base.mae
    _row("MAE vs ground truth",
         f"{base.mae*100:.3f} pp", f"{upg.mae*100:.3f} pp",
         mae_better)
    _row("  MAE human cells",
         f"{base.mae_human*100:.3f} pp", f"{upg.mae_human*100:.3f} pp",
         upg.mae_human < base.mae_human)
    _row("  MAE mouse cells",
         f"{base.mae_mouse*100:.3f} pp", f"{upg.mae_mouse*100:.3f} pp",
         upg.mae_mouse < base.mae_mouse)

    pr_better = abs(upg.pearson_r) > abs(base.pearson_r)
    _row("Pearson r(rho, GT)",
         f"{base.pearson_r:.4f}", f"{upg.pearson_r:.4f}",
         pr_better)

    print(f"\n  Per-cell rho (upgraded only — baseline has 1 global value):")
    pctls = {p: float(np.percentile(upg.rho_values, p)) for p in [10, 25, 50, 75, 90]}
    print(f"    p10={pctls[10]*100:.2f}%  p25={pctls[25]*100:.2f}%  "
          f"p50={pctls[50]*100:.2f}%  p75={pctls[75]*100:.2f}%  p90={pctls[90]*100:.2f}%")

    _rule("-")
    upg_better = sum([mae_better, pr_better])
    print(f"\n  Upgraded wins on {upg_better}/2 key metrics  "
          f"(lower MAE: {'YES' if mae_better else 'NO'},  "
          f"higher |Pearson r|: {'YES' if pr_better else 'NO'})")
    _rule()
    print()


def _print_fetal(base: FetalResult, upg: FetalResult):
    _rule()
    print("  Dataset : Fetal Liver (E-MTAB-7407)")
    print("  Method  : Baseline → auto_est_cont (global rho)   |   Upgraded → DecontX (per-cell rho)")
    print("  Criterion: non-erythroid rho > erythroid rho  (HBB/HBA = contamination in non-blood)")
    _rule()
    print(f"  {'Metric':<30}  {'Baseline':>14}  {'Upgraded':>14}  Better")
    _rule("-")
    _row("Mean rho (all cells)",
         f"{base.rho_mean*100:.3f}%", f"{upg.rho_mean*100:.3f}%")

    base_contrast = base.non_ery_rho > base.ery_rho if not np.isnan(base.non_ery_rho) else False
    upg_contrast  = upg.non_ery_rho  > upg.ery_rho  if not np.isnan(upg.non_ery_rho)  else False

    _row("Non-erythroid rho  (↑ = contam.)",
         f"{base.non_ery_rho*100:.3f}%", f"{upg.non_ery_rho*100:.3f}%",
         upg.non_ery_rho > base.non_ery_rho if not np.isnan(upg.non_ery_rho) else None)
    _row("Erythroid rho      (↓ = genuine)",
         f"{base.ery_rho*100:.3f}%", f"{upg.ery_rho*100:.3f}%",
         upg.ery_rho < base.ery_rho if not np.isnan(upg.ery_rho) else None)

    b_ratio = base.non_ery_rho / (base.ery_rho + 1e-10)
    u_ratio = upg.non_ery_rho  / (upg.ery_rho  + 1e-10)
    ratio_better = u_ratio > b_ratio
    _row("Contrast ratio (non-ery / ery)",
         f"{b_ratio:.2f}×", f"{u_ratio:.2f}×",
         ratio_better)

    _rule("-")
    _row("Non-ery > ery contrast (PASS/FAIL)",
         "PASS" if base_contrast else "FAIL",
         "PASS" if upg_contrast  else "FAIL",
         upg_contrast)

    for i, (bg, ug) in enumerate(zip(base.top5_genes, upg.top5_genes), 1):
        _row(f"soup gene #{i}", bg, ug, bg == ug)

    hb_genes = ["HBB", "HBA2", "HBA1", "HBD", "HBG1", "HBG2"]
    for g in hb_genes:
        b_in = g in base.hb_in_top10
        u_in = g in upg.hb_in_top10
        if b_in or u_in:
            _row(f"  {g} in top-10 soup",
                 "YES" if b_in else "NO",
                 "YES" if u_in else "NO")

    _rule("-")
    upg_better = sum([ratio_better, upg_contrast and not base_contrast])
    print(f"\n  Upgraded wins: better contrast ratio: {'YES' if ratio_better else 'NO'},  "
          f"contrast direction: upgraded={'PASS' if upg_contrast else 'FAIL'} "
          f"baseline={'PASS' if base_contrast else 'FAIL'}")
    _rule()
    print()


def _print_pbmc_decontx_result(r: DecontXPBMCResult):
    _rule()
    print(f"  Dataset : {r.dataset}")
    print(f"  Method  : Upgraded → DecontX (per-cell rho)")
    _rule("-")
    rb = r.counts_before - r.counts_after
    print(f"  n_cells            : {r.n_cells:,}")
    print(f"  rho mean           : {r.rho_mean*100:.3f}%")
    print(f"  rho p10/p50/p90    : {r.rho_p10*100:.2f}% / {r.rho_p50*100:.2f}% / {r.rho_p90*100:.2f}%")
    print(f"  counts removed     : {rb:,.0f}  ({rb/r.counts_before*100:.2f}%)")
    _rule()
    print()


def _print_marker_exclusivity(excl_base, excl_upg_auto, excl_upg_dx, label):
    _rule()
    print(f"  Marker Exclusivity — {label}")
    print(f"  Cross-lineage expression fold reduction in lineage-positive cells (top-25%)")
    print(f"  Higher fold = more contamination removed from wrong-lineage cells")
    _rule()
    print(f"  {'Lineage':<12}  {'Baseline':>12}  {'Upg-Auto':>12}  {'Upg-DecontX':>12}  Best")
    _rule("-")
    all_lins = sorted(
        set(excl_base.get("per_lineage", {}).keys())
        | set(excl_upg_auto.get("per_lineage", {}).keys())
        | set(excl_upg_dx.get("per_lineage", {}).keys())
    )
    for lin in all_lins:
        bf  = excl_base.get("per_lineage", {}).get(lin, {}).get("fold", 0.0)
        uaf = excl_upg_auto.get("per_lineage", {}).get(lin, {}).get("fold", 0.0)
        udf = excl_upg_dx.get("per_lineage", {}).get(lin, {}).get("fold", 0.0)
        best_val = max(bf, uaf, udf)
        winner   = ("baseline" if bf  == best_val else
                    "upg-auto" if uaf == best_val else "upg-dx")
        print(f"  {lin:<12}  {bf:>12.3f}×  {uaf:>12.3f}×  {udf:>12.3f}×  {winner}")
    _rule("-")
    bm  = excl_base.get("mean_fold", 0.0)
    uam = excl_upg_auto.get("mean_fold", 0.0)
    udm = excl_upg_dx.get("mean_fold", 0.0)
    bv  = max(bm, uam, udm)
    wm  = "baseline" if bm == bv else ("upg-auto" if uam == bv else "upg-dx")
    print(f"  {'MEAN':<12}  {bm:>12.3f}×  {uam:>12.3f}×  {udm:>12.3f}×  {wm}")
    _rule()
    print()


# ── Paper assessment metrics (head-to-head, synthetic data) ───────────────────

def compare_paper_metrics():
    """
    Run all 5 paper evaluation metrics for both baseline and upgraded pipelines.

    Synthetic data has variable contamination (high ~0.20, low ~0.05) and
    interleaved batches to exercise all 5 metrics properly.

    Baseline  : global rho via auto_est_cont → adjust_counts (subtraction)
    Upgraded  : global rho → estimate_cell_rho (per-cell EB) → adjust_counts
    """
    import numpy as np
    import scipy.sparse

    _rule()
    print("  Paper Assessment Metrics — Baseline vs Upgraded (synthetic data)")
    print("  Baseline : auto_est_cont → global rho → subtraction")
    print("  Upgraded : auto_est_cont → estimate_cell_rho (per-cell EB) → subtraction")
    _rule()

    rng = np.random.default_rng(42)

    # ── Synthetic data ─────────────────────────────────────────────────────────
    # 30 genes: 10 hg_, 10 mm_, 9 Gene*, 1 HBB
    # 120 cells:
    #   0:30   human HIGH-contam  (rho ~0.20, lots of mm_ cross-contam)
    #   30:60  human LOW-contam   (rho ~0.05)
    #   60:90  mouse HIGH-contam  (rho ~0.20, lots of hg_ cross-contam)
    #   90:120 mouse LOW-contam   (rho ~0.05)
    # Batch: interleaved (B1 = even index, B2 = odd index) — not correlated
    # Erythroid: last 10 cells (110:120), all low-contam mouse
    ng, nc = 30, 120
    gn = ([f'hg_{i}' for i in range(10)] +
          [f'mm_{i}' for i in range(10)] +
          [f'Gene{i}' for i in range(9)] +
          ['HBB'])                            # index 29

    sp  = ['human'] * 60 + ['mouse'] * 60
    cl  = np.array(['human'] * 60 + ['mouse'] * 60)
    bat = np.array(['B1' if i % 2 == 0 else 'B2' for i in range(nc)])
    ct  = ['T_cell'] * 110 + ['erythroid'] * 10

    b1  = np.where(bat == 'B1')[0]   # 60 cells, interleaved
    b2  = np.where(bat == 'B2')[0]

    toc_arr = np.zeros((ng, nc))

    # Species-specific expression (true biology)
    toc_arr[:10,   0:60]  = rng.poisson(10.0, (10, 60))   # hg_ in human
    toc_arr[10:20, 60:120] = rng.poisson(10.0, (10, 60))  # mm_ in mouse

    # Cross-species contamination (observable metric 1)
    # Human cells 0:30 HIGH → mm_ genes elevated (rho ~0.20)
    toc_arr[10:20, 0:30]   = rng.poisson(2.0, (10, 30))
    # Human cells 30:60 LOW  → mm_ genes low (rho ~0.05)
    toc_arr[10:20, 30:60]  = rng.poisson(0.5, (10, 30))
    # Mouse cells 60:90 HIGH → hg_ genes elevated (rho ~0.20)
    toc_arr[:10,   60:90]  = rng.poisson(2.0, (10, 30))
    # Mouse cells 90:120 LOW → hg_ genes low (rho ~0.05)
    toc_arr[:10,   90:120] = rng.poisson(0.5, (10, 30))

    # Batch-specific soup artifacts (distinct genes per batch → PCA batch signal)
    toc_arr[20:23, b1] = rng.poisson(4.0, (3, len(b1)))   # Gene0-2: B1 artifact
    toc_arr[23:26, b2] = rng.poisson(4.0, (3, len(b2)))   # Gene3-5: B2 artifact
    toc_arr[26:29, :]  = rng.poisson(0.3, (3, nc))        # Gene6-8: background

    # HBB (index 29): genuine in erythroid, contamination in T cells
    toc_arr[29, 110:] = rng.poisson(20.0, 10)    # genuine erythroid
    toc_arr[29, :110] = rng.poisson(3.0,  110)   # contamination in T cells

    # Empty droplets (tod) → soup profile
    tod_arr               = rng.poisson(0.15, (ng, nc * 4))
    tod_arr[:10,  :]      = rng.poisson(0.3,  (10, nc * 4))   # hg_ in soup
    tod_arr[10:20, :]     = rng.poisson(0.3,  (10, nc * 4))   # mm_ in soup
    tod_arr[20:23, :]     = rng.poisson(0.5,  (3,  nc * 4))   # Gene0-2 in soup
    tod_arr[23:26, :]     = rng.poisson(0.5,  (3,  nc * 4))   # Gene3-5 in soup
    tod_arr[29, :]        = rng.poisson(1.5,  nc * 4)          # HBB in soup

    toc      = scipy.sparse.csc_matrix(toc_arr)
    tod      = scipy.sparse.csc_matrix(tod_arr)
    cell_ids = [f'C{i:04d}' for i in range(nc)]
    drop_ids = [f'D{i:05d}' for i in range(nc * 4)]

    markers = {
        'human': [f'hg_{i}' for i in range(5)],
        'mouse': [f'mm_{i}' for i in range(5)],
    }

    # ── Baseline pipeline: global rho via auto_est_cont ────────────────────────
    import soupx as B
    from soupx.soup_channel import SoupChannel as BSC
    from soupx.metrics import (
        cross_species_reduction  as B_csr,
        marker_fold_change       as B_mfc,
        cluster_membership_delta as B_cmd,
        batch_entropy            as B_be,
        hbb_expression_analysis  as B_hbb,
    )
    import pandas as pd

    global_rho = 0.13   # fallback; replaced by auto_est_cont if it succeeds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc_b = BSC(tod=tod, toc=toc, gene_names=gn, cell_names=cell_ids)
        sc_b.set_clusters(pd.Series(cl, index=cell_ids))
        try:
            sc_b = B.auto_est_cont(sc_b, do_plot=False, verbose=False,
                                   force_accept=True)
            global_rho = float(sc_b.meta_data['rho'].iloc[0])
            print(f"  Baseline auto_est_cont → global rho = {global_rho:.4f}")
        except Exception as e:
            sc_b.set_contamination_fraction(rho=global_rho)
            print(f"  Baseline auto_est_cont failed ({e!s:.60}), using rho={global_rho:.3f}")
        cor_b = B.adjust_counts(sc_b, method='subtraction', verbose=0)

    m1_b = B_csr(toc, cor_b, gn, sp)
    m2_b = B_mfc(toc, cor_b, cl, markers, gn)
    m3_b = B_cmd(toc, cor_b, n_clusters=3)
    m4_b = B_be(toc,  cor_b, bat, n_neighbors=15)
    m5_b = B_hbb(toc, cor_b, ct, gn, hbb_genes=['HBB'])

    # ── Upgraded pipeline: per-cell rho via estimate_cell_rho ─────────────────
    import SoupX as U
    from SoupX.soup_channel import SoupChannel as USC
    from SoupX.metrics import (
        cross_species_reduction  as U_csr,
        marker_fold_change       as U_mfc,
        cluster_membership_delta as U_cmd,
        batch_entropy            as U_be,
        hbb_expression_analysis  as U_hbb,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc_u = USC(tod=tod, toc=toc,
                   genes=pd.Index(gn), cells=pd.Index(cell_ids),
                   drop_barcodes=drop_ids, calc_soup_profile=True)
        sc_u = U.set_clusters(sc_u, cl)
        # Start from same global rho as baseline, then refine per-cell
        sc_u = U.set_contamination_fraction(sc_u, global_rho)
        sc_u = U.estimate_cell_rho(sc_u, soup_quantile=0.7, prior_std=0.08)
        rho_vals = sc_u.meta_data['rho'].values
        print(f"  Upgraded per-cell rho: "
              f"p10={np.percentile(rho_vals,10):.3f}  "
              f"p50={np.percentile(rho_vals,50):.3f}  "
              f"p90={np.percentile(rho_vals,90):.3f}")
        cor_u = U.adjust_counts(sc_u, method='subtraction')

    m1_u = U_csr(toc, cor_u, gn, sp)
    m2_u = U_mfc(toc, cor_u, cl, markers, gn)
    m3_u = U_cmd(toc, cor_u, n_clusters=3)
    m4_u = U_be(toc,  cor_u, bat, n_neighbors=15)
    m5_u = U_hbb(toc, cor_u, ct, gn, hbb_genes=['HBB'])

    # ── Print comparison ───────────────────────────────────────────────────────
    _rule()
    print(f"  {'Metric':<38}  {'Baseline':>14}  {'Upgraded':>14}  Better")
    _rule("-")

    def _better(bv, uv, higher_is_better=True):
        return uv > bv if higher_is_better else uv < bv

    # 1. Cross-species fold reduction
    b1 = m1_b['fold_reduction']
    u1 = m1_u['fold_reduction']
    _row("1. Cross-species fold reduction",
         f"{b1:.2f}×", f"{u1:.2f}×", _better(b1, u1))
    _row("   contam before → after (baseline)",
         f"{m1_b['contamination_before']*100:.2f}%→{m1_b['contamination_after']*100:.2f}%", "", None)
    _row("   contam before → after (upgraded)",
         "", f"{m1_u['contamination_before']*100:.2f}%→{m1_u['contamination_after']*100:.2f}%", None)
    _row("   meets ≥2x threshold",
         "PASS" if m1_b['meets_2fold_threshold'] else "FAIL",
         "PASS" if m1_u['meets_2fold_threshold'] else "FAIL",
         m1_u['meets_2fold_threshold'])
    _rule("-")

    # 2. Marker gene fold change
    b2 = m2_b['mean_fc_after']
    u2 = m2_u['mean_fc_after']
    _row("2. Marker FC after correction",
         f"{b2:.2f}×", f"{u2:.2f}×", _better(b2, u2))
    _row("   FC ratio (after/before)",
         f"{m2_b['fc_ratio']:.3f}", f"{m2_u['fc_ratio']:.3f}", _better(m2_b['fc_ratio'], m2_u['fc_ratio']))
    _row("   majority markers improved",
         "YES" if m2_b['improved'] else "NO",
         "YES" if m2_u['improved'] else "NO",
         m2_u['improved'])
    _rule("-")

    # 3. Cluster membership
    b3 = m3_b['n_clusters_lost']
    u3 = m3_u['n_clusters_lost']
    _row("3. Clusters lost after correction",
         str(b3), str(u3), _better(b3, u3))
    _row("   % cells changed cluster",
         f"{m3_b['pct_cells_changed']:.1f}%", f"{m3_u['pct_cells_changed']:.1f}%", None)
    _row("   ARI (before vs after labels)",
         f"{m3_b['ari']:.4f}", f"{m3_u['ari']:.4f}", None)
    _rule("-")

    # 4. Batch entropy
    b4 = m4_b['entropy_increase']
    u4 = m4_u['entropy_increase']
    _row("4. Batch entropy increase",
         f"{b4:+.4f}", f"{u4:+.4f}", _better(b4, u4))
    _row("   normalized entropy (after)",
         f"{m4_b['normalized_after']:.4f}", f"{m4_u['normalized_after']:.4f}",
         _better(m4_b['normalized_after'], m4_u['normalized_after']))
    _row("   cross-batch mixing improved",
         "YES" if m4_b['improved'] else "NO",
         "YES" if m4_u['improved'] else "NO",
         m4_u['improved'])
    _rule("-")

    # 5. HBB expression
    b5 = m5_b['mean_pct_reduction']
    u5 = m5_u['mean_pct_reduction']
    _row("5. HBB non-eryth % reduction",
         f"{b5:.1f} pp", f"{u5:.1f} pp", _better(b5, u5))
    _row("   HBB non-eryth before → after (baseline)",
         f"{m5_b['mean_pct_noneryth_before']:.1f}%→{m5_b['mean_pct_noneryth_after']:.1f}%", "", None)
    _row("   HBB non-eryth before → after (upgraded)",
         "", f"{m5_u['mean_pct_noneryth_before']:.1f}%→{m5_u['mean_pct_noneryth_after']:.1f}%", None)
    _row("   HBB signal reduced",
         "YES" if m5_b['hbb_signal_reduced'] else "NO",
         "YES" if m5_u['hbb_signal_reduced'] else "NO",
         m5_u['hbb_signal_reduced'])
    _rule()

    wins = sum([
        _better(b1, u1), _better(b2, u2), _better(b3, u3),
        _better(b4, u4), _better(b5, u5),
    ])
    print(f"\n  Upgraded wins on {wins}/5 paper evaluation metrics")
    _rule()
    print()

    base_metrics = (m1_b, m2_b, m3_b, m4_b, m5_b)
    upg_metrics  = (m1_u, m2_u, m3_u, m4_u, m5_u)
    return base_metrics, upg_metrics


# ── Dataset runners ───────────────────────────────────────────────────────────

def compare_toy_pbmc():
    print("\n  Loading Toy PBMC ...")
    tod, toc, gene_names, bc_raw, bc_filt = _load_toy_matrices()
    clusters = pd.read_csv(os.path.join(TOY_DIR_UPG, "metaData.tsv"),
                           sep="\t", index_col=0)["res.1"].astype(str)

    print("  Baseline: auto_est_cont ...")
    base = _baseline_standard(TOY_DIR_BASE, clusters, "Toy PBMC")
    print("  Upgraded: auto_est_cont ...")
    upg  = _upgraded_standard(TOY_DIR_UPG, clusters, "Toy PBMC")
    _print_standard(base, upg)

    print("  Upgraded: DecontX ...")
    dx_result, cor_dx = _upgraded_pbmc_decontx(
        tod, toc, gene_names, bc_raw, bc_filt,
        clusters, "Toy PBMC", n_topics=5, n_iter=200, n_hvg=226)
    _print_pbmc_decontx_result(dx_result)

    print("  Marker exclusivity (re-running corrections) ...")
    gn_b, toc_b, cor_b = _baseline_correction_matrix(TOY_DIR_BASE, clusters)
    gn_u, toc_u, cor_u = _upgraded_autoestcont_correction_matrix(TOY_DIR_UPG, clusters)
    excl_base = _marker_exclusivity(toc_b, cor_b, gn_b)
    excl_upg  = _marker_exclusivity(toc_u, cor_u, gn_u)
    _print_marker_exclusivity(excl_base, excl_upg, dx_result.marker_excl, "Toy PBMC")

    return base, upg


def compare_pbmc10k():
    print("\n  Loading PBMC 10k v3 ...")
    tod, toc, gene_names, bc_raw, bc_filt = _load_pbmc10k_matrices()
    clusters = pd.read_csv(PBMC10K_CLU).set_index("Barcode")["Cluster"].astype(str)

    print("  Baseline: auto_est_cont ...")
    base = _baseline_standard(PBMC10K_DIR, clusters, "PBMC 10k v3")
    print("  Upgraded: auto_est_cont ...")
    upg  = _upgraded_standard(PBMC10K_DIR, clusters, "PBMC 10k v3")
    _print_standard(base, upg)

    print("  Upgraded: DecontX ...")
    dx_result, cor_dx = _upgraded_pbmc_decontx(
        tod, toc, gene_names, bc_raw, bc_filt,
        clusters, "PBMC 10k v3", n_topics=15, n_iter=300, n_hvg=2000)
    _print_pbmc_decontx_result(dx_result)

    print("  Marker exclusivity (re-running corrections) ...")
    gn_b, toc_b, cor_b = _baseline_correction_matrix(PBMC10K_DIR, clusters)
    gn_u, toc_u, cor_u = _upgraded_autoestcont_correction_matrix(PBMC10K_DIR, clusters)
    excl_base = _marker_exclusivity(toc_b, cor_b, gn_b)
    excl_upg  = _marker_exclusivity(toc_u, cor_u, gn_u)
    _print_marker_exclusivity(excl_base, excl_upg, dx_result.marker_excl, "PBMC 10k v3")

    return base, upg


def compare_hgmm():
    print("\n  Loading HGMM barnyard matrices ...")
    tod, toc, all_genes, all_cells, bc_raw, clusters, human_mask, gt = _load_hgmm()
    print(f"  Cells: {len(all_cells):,}  ({human_mask.sum()} human HEK293T, "
          f"{(~human_mask).sum()} mouse NIH3T3)")
    print(f"  GT mean contamination: {gt.mean()*100:.3f}%")
    print("  Baseline: auto_est_cont (global rho) ...")
    base = _baseline_hgmm(tod, toc, all_genes, all_cells, clusters, human_mask, gt)
    print("  Upgraded: DecontX per-cell EM ...")
    upg  = _upgraded_hgmm_decontx(tod, toc, all_genes, all_cells, bc_raw, human_mask, gt)
    _print_hgmm(base, upg)
    return base, upg


def compare_fetal_liver():
    print("\n  Loading Fetal Liver matrix ...")
    mat, gene_names, barcodes, clusters, soup_df = _load_fetal()
    print(f"  Cells: {mat.shape[1]:,}   Cell types: {clusters.nunique()}")
    print("  Baseline: auto_est_cont (global rho) ...")
    base = _baseline_fetal(mat, gene_names, barcodes, clusters, soup_df)
    print("  Upgraded: DecontX per-cell EM ...")
    upg  = _upgraded_fetal_decontx(mat, gene_names, barcodes, clusters, soup_df)
    _print_fetal(base, upg)
    return base, upg


# ── Final summary ─────────────────────────────────────────────────────────────

def _print_summary(results: list):
    _rule()
    print("  OVERALL SUMMARY")
    _rule()
    for item in results:
        key, base, upg = item
        if key in ("toy_pbmc", "pbmc_10k"):
            delta = upg.rho - base.rho
            print(f"  {base.dataset:<22}  rho baseline={base.rho:.4f}  "
                  f"upgraded={upg.rho:.4f}  delta={delta:+.4f}")
        elif key == "hgmm":
            mae_imp = (base.mae - upg.mae) / base.mae * 100
            pr_imp  = upg.pearson_r - base.pearson_r
            print(f"  {'HGMM Barnyard':<22}  "
                  f"MAE improvement={mae_imp:+.1f}%  "
                  f"Pearson r: {base.pearson_r:.4f}→{upg.pearson_r:.4f}  "
                  f"(+{pr_imp:.4f})")
        elif key == "fetal_liver":
            b_ratio = base.non_ery_rho / (base.ery_rho + 1e-10)
            u_ratio = upg.non_ery_rho  / (upg.ery_rho  + 1e-10)
            print(f"  {'Fetal Liver':<22}  "
                  f"contrast ratio: {b_ratio:.2f}×→{u_ratio:.2f}×  "
                  f"non-ery>ery: baseline={'PASS' if base.non_ery_rho>base.ery_rho else 'FAIL'}  "
                  f"upgraded={'PASS' if upg.non_ery_rho>upg.ery_rho else 'FAIL'}")
        elif key == "paper_metrics":
            m1_b, m2_b, m3_b, m4_b, m5_b = base
            m1_u, m2_u, m3_u, m4_u, m5_u = upg
            wins = sum([
                m1_u['fold_reduction']    > m1_b['fold_reduction'],
                m2_u['mean_fc_after']     > m2_b['mean_fc_after'],
                m3_u['n_clusters_lost']   > m3_b['n_clusters_lost'],
                m4_u['entropy_increase']  > m4_b['entropy_increase'],
                m5_u['mean_pct_reduction']> m5_b['mean_pct_reduction'],
            ])
            print(f"  {'Paper Metrics (5)':<22}  upgraded wins {wins}/5 metrics")
    _rule()


# ── CLI ───────────────────────────────────────────────────────────────────────

_RUNNERS = {
    "toy_pbmc":      compare_toy_pbmc,
    "pbmc_10k":      compare_pbmc10k,
    "hgmm":          compare_hgmm,
    "fetal_liver":   compare_fetal_liver,
    "paper_metrics": compare_paper_metrics,
}


def main():
    parser = argparse.ArgumentParser(description="Baseline vs Upgraded SoupX comparison")
    parser.add_argument("--datasets", nargs="+", choices=list(_RUNNERS),
                        default=None, metavar="DATASET")
    args = parser.parse_args()

    _rule()
    print("  SoupX: Baseline vs Upgraded — Full Pipeline Comparison")
    print("  Baseline : auto_est_cont pipeline (unchanged)")
    print("  Upgraded : DecontX (HGMM/fetal liver) + upgraded auto_est_cont (toy/pbmc10k)")
    _rule()

    avail = {
        "toy_pbmc":      os.path.isdir(TOY_DIR_UPG) and os.path.isdir(TOY_DIR_BASE),
        "pbmc_10k":      os.path.isdir(PBMC10K_DIR) and os.path.isfile(PBMC10K_CLU),
        "hgmm":          os.path.isdir(os.path.join(HGMM_DIR, "raw_gene_bc_matrices")),
        "fetal_liver":   os.path.isdir(os.path.join(FETAL_DIR, "GRCh38")),
        "paper_metrics": True,  # always available — uses synthetic data
    }
    print("\n  Dataset availability:")
    for k, ok in avail.items():
        print(f"    {'✓' if ok else '✗'}  {k}")
    print()

    to_run  = args.datasets or [k for k, ok in avail.items() if ok]
    summary = []
    for key in to_run:
        if not avail.get(key, False):
            print(f"  SKIP {key} — data not found")
            continue
        try:
            base, upg = _RUNNERS[key]()
            summary.append((key, base, upg))
        except Exception:
            import traceback
            print(f"\n  ERROR running {key}:")
            traceback.print_exc()

    if summary:
        _print_summary(summary)


if __name__ == "__main__":
    main()
