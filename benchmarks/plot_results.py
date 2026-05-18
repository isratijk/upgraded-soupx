#!/usr/bin/env python3
"""
plot_results.py — Visualise SoupX benchmark results.

Usage
-----
    python plot_results.py                           # run benchmark + plot
    python plot_results.py --csv results.csv         # load pre-saved CSV
    python plot_results.py --datasets hgmm fetal_liver  # specific datasets
    python plot_results.py --skip-decontx            # skip slow pipelines
    python plot_results.py --out-dir ./my_plots/     # custom output dir
    python plot_results.py --save-csv results.csv    # run + save CSV + plot

Outputs (in ./plots/ by default)
    01_rho_comparison.png   — rho mean ± std per pipeline per dataset
    02_metric_heatmap.png   — normalised metric heatmap (pipeline × metric)
    03_gt_metrics.png       — GT-MAE and GT-Pearson (HGMM datasets only)
    04_cluster_quality.png  — M3-ARI and M6-Silhouette delta
    05_radar_chart.png      — aggregate radar chart across all metrics
"""

import argparse
import os
import sys
import warnings
from math import pi

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# ── Style constants ───────────────────────────────────────────────────────────

PIPELINE_COLORS = {
    "baseline":      "#7f7f7f",
    "upg-auto":      "#1f77b4",
    "upg-doublet":   "#2ca02c",
    "upg-iterative": "#ff7f0e",
    "upg-decontx":   "#9467bd",
    "upg-genehet":   "#17becf",
}
PIPELINE_ORDER = ["baseline", "upg-auto", "upg-doublet",
                  "upg-iterative", "upg-decontx", "upg-genehet"]

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _color(p):
    return PIPELINE_COLORS.get(p, "#333333")


def _pipelines_in(df):
    present = set(df["pipeline"].unique())
    return [p for p in PIPELINE_ORDER if p in present] + \
           [p for p in df["pipeline"].unique() if p not in PIPELINE_ORDER]


def _val(df_sub, pipeline, col):
    row = df_sub[df_sub["pipeline"] == pipeline]
    if len(row) == 0:
        return np.nan
    v = row[col].values[0]
    return float(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else np.nan


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _normalise(values_dict, higher_better=True):
    """Map {pipeline: raw_value} to {pipeline: 0..1 normalised}."""
    items = {p: v for p, v in values_dict.items() if not np.isnan(v)}
    if len(items) < 2:
        return {p: 0.5 for p in items}
    vmin, vmax = min(items.values()), max(items.values())
    span = vmax - vmin
    if span < 1e-12:
        return {p: 0.5 for p in items}
    return {p: (v - vmin) / span if higher_better else 1 - (v - vmin) / span
            for p, v in items.items()}


# ── Plot 1: rho comparison ────────────────────────────────────────────────────

def plot_rho_comparison(df, out_dir):
    datasets  = list(df["dataset"].unique())
    pipelines = _pipelines_in(df)
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, n_ds, figsize=(max(4, 3.5 * n_ds), 5),
                             sharey=False, squeeze=False)
    axes = axes[0]

    for ax, ds in zip(axes, datasets):
        sub   = df[df["dataset"] == ds]
        pipes = [p for p in pipelines if p in sub["pipeline"].values]
        means = [_val(sub, p, "rho_mean") * 100 for p in pipes]
        stds  = [_val(sub, p, "rho_std")  * 100 for p in pipes]
        colors = [_color(p) for p in pipes]
        x = np.arange(len(pipes))

        ax.bar(x, means, yerr=stds, capsize=4, color=colors,
               edgecolor="white", linewidth=0.5,
               error_kw={"linewidth": 1.5, "ecolor": "black", "alpha": 0.7})

        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(pipes, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Contamination ρ (%)", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)

    legend_patches = [mpatches.Patch(color=_color(p), label=p) for p in pipelines]
    fig.legend(handles=legend_patches, loc="lower center", ncol=min(6, len(pipelines)),
               fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Estimated Contamination Fraction per Pipeline",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "01_rho_comparison.png"))


# ── Plot 2: metric heatmap ────────────────────────────────────────────────────

_METRICS_DEF = [
    ("m1_fold",          True,  "M1\nCross-Species"),
    ("m2_fc_ratio",      True,  "M2\nMarker FC"),
    ("m3_ari",           True,  "M3\nARI"),
    ("m4_entropy_delta", True,  "M4\nEntropy"),
    ("m5_pct_reduction", True,  "M5\nHBB"),
    ("gt_mae",           False, "GT\nMAE↓"),
    ("gt_pearson",       True,  "GT\nPearson"),
    ("m6_sil_delta",     True,  "M6\nSilhouette"),
    ("m7_n_spurious",    True,  "M7\nSpurious"),
    ("m8_rank_delta",    True,  "M8\nMkRank"),
]


def plot_metric_heatmap(df, out_dir):
    datasets  = list(df["dataset"].unique())
    pipelines = _pipelines_in(df)
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, n_ds,
                             figsize=(max(5, 3.2 * n_ds), 2 + 0.55 * len(pipelines)),
                             squeeze=False)
    axes = axes[0]

    cmap = plt.get_cmap("RdYlGn")

    for ax_idx, (ax, ds) in enumerate(zip(axes, datasets)):
        sub = df[df["dataset"] == ds]
        pipes = [p for p in pipelines if p in sub["pipeline"].values]

        valid_metrics = [(col, hb, nm) for col, hb, nm in _METRICS_DEF
                         if sub[col].notna().any()]
        n_metrics = len(valid_metrics)

        mat = np.full((len(pipes), n_metrics), np.nan)
        for j, (col, hb, _) in enumerate(valid_metrics):
            raw = {p: _val(sub, p, col) for p in pipes}
            norm = _normalise(raw, higher_better=hb)
            for i, p in enumerate(pipes):
                if p in norm:
                    mat[i, j] = norm[p]

        # draw heatmap manually
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        for i in range(len(pipes)):
            for j in range(n_metrics):
                val = mat[i, j]
                if not np.isnan(val):
                    text_color = "black" if 0.25 < val < 0.75 else "white"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=text_color, fontweight="bold")

        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels([nm for _, _, nm in valid_metrics], fontsize=7, rotation=0)
        ax.set_yticks(range(len(pipes)))
        ax.set_yticklabels(pipes if ax_idx == 0 else [""] * len(pipes), fontsize=8)
        ax.set_title(ds, fontsize=11, fontweight="bold", pad=8)
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(-0.5, n_metrics, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(pipes), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    # colorbar on last axis
    cb_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, cax=cb_ax, label="Normalised Score\n(0=worst, 1=best)")

    fig.suptitle("Metric Heatmap — Normalised per Metric per Dataset",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    _save(fig, os.path.join(out_dir, "02_metric_heatmap.png"))


# ── Plot 3: GT metrics (HGMM only) ───────────────────────────────────────────

def plot_gt_metrics(df, out_dir):
    hgmm_df = df[df["gt_mae"].notna()]
    if hgmm_df.empty:
        print("  [skip] 03_gt_metrics.png — no GT metrics in data")
        return

    datasets  = list(hgmm_df["dataset"].unique())
    pipelines = _pipelines_in(hgmm_df)
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, 2, figsize=(11, max(4, 1 + 0.6 * len(pipelines))))

    ds_colors = [cm.Set2(i / max(n_ds - 1, 1)) for i in range(n_ds)]
    w = 0.75 / max(n_ds, 1)
    offsets = np.linspace(-(n_ds - 1) * w / 2, (n_ds - 1) * w / 2, n_ds)
    x = np.arange(len(pipelines))

    for ax_idx, (col, label, ylabel, threshold, invert) in enumerate([
        ("gt_mae",     "GT-MAE (lower = better)",    "MAE (pp)",  5.0,   True),
        ("gt_pearson", "GT-Pearson r (higher = better)", "Pearson r", 0.50, False),
    ]):
        ax = axes[ax_idx]
        for di, (ds, dc) in enumerate(zip(datasets, ds_colors)):
            sub  = hgmm_df[hgmm_df["dataset"] == ds]
            vals = []
            for p in pipelines:
                v = _val(sub, p, col)
                vals.append(v * 100 if col == "gt_mae" and not np.isnan(v) else v)
            bars = ax.bar(x + offsets[di], vals, width=w * 0.92, label=ds,
                          color=dc, edgecolor="white", linewidth=0.5)

        ax.axhline(threshold, color="#d62728", linestyle="--",
                   linewidth=1.2, alpha=0.8, label=f"threshold ({threshold})")
        ax.set_xticks(x)
        ax.set_xticklabels(pipelines, rotation=40, ha="right", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)
        if col == "gt_pearson":
            ax.set_ylim(bottom=-0.05, top=1.05)

    fig.suptitle("Ground Truth Metrics — HGMM Barnyard Datasets",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "03_gt_metrics.png"))


# ── Plot 4: cluster quality ───────────────────────────────────────────────────

def plot_cluster_quality(df, out_dir):
    datasets  = list(df["dataset"].unique())
    pipelines = _pipelines_in(df)
    n_ds = len(datasets)

    fig, axes = plt.subplots(2, n_ds,
                             figsize=(max(3 * n_ds, 8), 8),
                             sharey="row", squeeze=False)

    for di, ds in enumerate(datasets):
        sub   = df[df["dataset"] == ds]
        pipes = [p for p in pipelines if p in sub["pipeline"].values]
        x     = np.arange(len(pipes))

        # ── M3 ARI ──
        ax = axes[0, di]
        vals = [_val(sub, p, "m3_ari") for p in pipes]
        bar_colors = [_color(p) for p in pipes]
        ax.bar(x, vals, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(ds, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels([""] * len(pipes))
        if di == 0:
            ax.set_ylabel("M3 Cluster ARI\n(1=perfectly stable)", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)
        for xi, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        # ── M6 Silhouette ──
        ax = axes[1, di]
        vals = [_val(sub, p, "m6_sil_delta") for p in pipes]
        sil_colors = []
        for v in vals:
            if np.isnan(v):
                sil_colors.append("#cccccc")
            elif v >= 0:
                sil_colors.append("#2ca02c")
            else:
                sil_colors.append("#d62728")
        ax.bar(x, vals, color=sil_colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(pipes, rotation=40, ha="right", fontsize=8)
        if di == 0:
            ax.set_ylabel("M6 Silhouette Δ\n(↑ = better separation)", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)

    row_labels = ["M3: Cluster ARI", "M6: Silhouette Δ"]
    for ri, rl in enumerate(row_labels):
        axes[ri, 0].annotate(rl, xy=(-0.25, 0.5), xycoords="axes fraction",
                             fontsize=9, rotation=90, va="center", color="gray")

    fig.suptitle("Cluster Quality — M3 ARI & M6 Silhouette",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "04_cluster_quality.png"))


# ── Plot 5: radar chart ───────────────────────────────────────────────────────

_RADAR_CATS = [
    ("m2_fc_ratio",      True,  "M2\nMarker FC"),
    ("m3_ari",           True,  "M3 ARI"),
    ("m4_entropy_delta", True,  "M4\nEntropy"),
    ("gt_mae",           False, "GT MAE"),
    ("gt_pearson",       True,  "GT\nPearson"),
    ("m6_sil_delta",     True,  "M6\nSilhouette"),
    ("m8_rank_delta",    True,  "M8\nMkRank"),
]


def plot_radar(df, out_dir):
    datasets  = list(df["dataset"].unique())
    pipelines = _pipelines_in(df)

    pipe_scores = {p: [] for p in pipelines}

    for col, hb, _ in _RADAR_CATS:
        cat_scores = {p: [] for p in pipelines}
        for ds in datasets:
            sub = df[df["dataset"] == ds]
            raw = {p: _val(sub, p, col) for p in pipelines
                   if not np.isnan(_val(sub, p, col))}
            if not raw:
                continue
            norm = _normalise(raw, higher_better=hb)
            for p, v in norm.items():
                cat_scores[p].append(v)

        for p in pipelines:
            scores = cat_scores[p]
            pipe_scores[p].append(np.nanmean(scores) if scores else np.nan)

    cat_labels = [nm for _, _, nm in _RADAR_CATS]
    N = len(_RADAR_CATS)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for p in pipelines:
        scores = pipe_scores[p]
        if all(np.isnan(s) for s in scores):
            continue
        values = [s if not np.isnan(s) else 0.0 for s in scores]
        values += values[:1]
        ax.plot(angles, values, linewidth=2.0, linestyle="solid",
                label=p, color=_color(p), alpha=0.9)
        ax.fill(angles, values, color=_color(p), alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="gray")
    ax.grid(color="gray", alpha=0.3, linewidth=0.8)
    ax.spines["polar"].set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=9, framealpha=0.85)
    ax.set_title("Pipeline Performance Radar\n(normalised, averaged across datasets)",
                 fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "05_radar_chart.png"))


# ── Plot 6 & 7: downstream embedding + DE — all datasets × all pipelines ─────

_DS_MARKERS = {
    "toy_pbmc":       {"T_cell": ["CD3D", "CD3E"], "B_cell": ["CD79A", "MS4A1"],
                       "Monocyte": ["LYZ", "CD14"]},
    "pbmc_10k":       {"T_cell": ["CD3D", "CD3E"], "B_cell": ["CD79A", "MS4A1"],
                       "Monocyte": ["LYZ", "CD14"]},
    "hgmm":           {"human": ["ACTB", "GAPDH"], "mouse": ["Actb", "Gapdh"]},
    "fetal_liver":    {"Erythroid": ["HBB", "HBA2"], "Other": ["CD3D", "LYZ"]},
    "rep1_zenodo_gt": {"T_cell": ["CD3D", "CD3E"], "B_cell": ["CD79A", "MS4A1"],
                       "Monocyte": ["LYZ", "CD14"]},
}

_ALL_PIPELINES = ["baseline", "upg-auto", "upg-doublet",
                  "upg-iterative", "upg-decontx", "upg-genehet"]

# DE is run only for datasets small enough to densify in memory (<= 5 k cells)
_DE_MAX_CELLS = 5_000


def plot_downstream_analysis(out_dir):
    """t-SNE embeddings + DE for all available datasets × all pipelines.

    Only pipelines that succeed are rendered.  Datasets where every pipeline
    fails are skipped entirely — no FAILED panels are written to disk.
    """
    try:
        from benchmark_full import (
            get_corrected_matrix_toy_pbmc, get_corrected_matrix_pbmc10k,
            get_corrected_matrix_hgmm, get_corrected_matrix_fetal_liver,
            get_corrected_matrix_rep1_zenodo_gt, _AVAIL,
        )
    except Exception as e:
        print(f"  [skip] downstream plots — import failed: {e}")
        return

    from SoupX.downstream import run_downstream, plot_embedding, plot_top_de_genes

    _GETTERS = {
        "toy_pbmc":       get_corrected_matrix_toy_pbmc,
        "pbmc_10k":       get_corrected_matrix_pbmc10k,
        "hgmm":           get_corrected_matrix_hgmm,
        "fetal_liver":    get_corrected_matrix_fetal_liver,
        "rep1_zenodo_gt": get_corrected_matrix_rep1_zenodo_gt,
    }

    for ds, getter in _GETTERS.items():
        if not _AVAIL.get(ds, lambda: False)():
            print(f"  [skip] {ds} — data not found")
            continue

        print(f"\n  [downstream] {ds} ...")

        # ── collect results for every pipeline first ──────────────────────────
        pipe_results = {}   # pipe -> (cor, gene_names, cls_arr, dn_dict)
        for pipe in _ALL_PIPELINES:
            print(f"    {pipe} ...")
            try:
                toc, cor, gene_names, cls_arr = getter(pipeline=pipe)
            except Exception as e:
                print(f"    [skip] {pipe}: {e}")
                continue

            n_cells = cor.shape[1] if hasattr(cor, "shape") else len(cls_arr)
            run_de  = n_cells <= _DE_MAX_CELLS
            try:
                dn = run_downstream(cor, gene_names, cluster_labels=cls_arr,
                                    embedding="tsne", clustering=None, run_de=run_de)
                pipe_results[pipe] = (cor, gene_names, cls_arr, dn)
            except Exception as e:
                print(f"    [warn] downstream failed for {ds}/{pipe}: {e}")
                continue

        if not pipe_results:
            print(f"  [skip] {ds} — all pipelines failed, no plots generated")
            continue

        working = list(pipe_results.keys())
        n_w     = len(working)

        # ── Plot 06: embeddings (only working pipelines) ──────────────────────
        fig_emb, axes_emb = plt.subplots(1, n_w, figsize=(4 * n_w, 4.5), squeeze=False)
        axes_emb = axes_emb[0]
        for ax_e, pipe in zip(axes_emb, working):
            cor, gene_names, cls_arr, dn = pipe_results[pipe]
            if dn.get("embedding") is not None:
                plot_embedding(dn["embedding"], dn["cluster_labels"],
                               title=pipe, ax=ax_e, point_size=3)
            else:
                ax_e.set_title(pipe, fontsize=8)
        fig_emb.suptitle(f"{ds} — t-SNE embeddings (corrected)", fontsize=12)
        plt.tight_layout()
        _save(fig_emb, os.path.join(out_dir, f"06_{ds}_embeddings.png"))

        # ── Plot 07: DE genes (only pipelines with actual DE results) ─────────
        de_pipes = [
            (pipe, pipe_results[pipe][3].get("de_results"))
            for pipe in working
            if pipe_results[pipe][3].get("de_results") is not None
            and not pipe_results[pipe][3]["de_results"].empty
        ]
        if de_pipes:
            n_de = len(de_pipes)
            fig_de, axes_de = plt.subplots(1, n_de,
                                            figsize=(4 * n_de, 6), squeeze=False)
            axes_de = axes_de[0]
            for ax_d, (pipe, de) in zip(axes_de, de_pipes):
                plot_top_de_genes(de, n_genes=3, ax=ax_d)
                ax_d.set_title(pipe, fontsize=8)
            fig_de.suptitle(f"{ds} — Top DE genes per cluster", fontsize=12)
            plt.tight_layout()
            _save(fig_de, os.path.join(out_dir, f"07_{ds}_de_genes.png"))
        else:
            n_cells_sample = (pipe_results[working[0]][0].shape[1]
                              if hasattr(pipe_results[working[0]][0], "shape") else "?")
            print(f"  [skip] 07_{ds}_de_genes — DE skipped "
                  f"(n_cells={n_cells_sample:,} > {_DE_MAX_CELLS:,} limit)"
                  if isinstance(n_cells_sample, int)
                  else f"  [skip] 07_{ds}_de_genes — no DE results")


# ── Plot 8 & 9: SoupX diagnostic plots — all available datasets ──────────────

def plot_soupx_diagnostics(out_dir):
    """Soup-correlation and marker-distribution plots for all available datasets."""
    try:
        from benchmark_full import (
            get_fitted_sc_toy_pbmc, get_fitted_sc_pbmc10k,
            get_fitted_sc_fetal_liver, _AVAIL,
        )
    except Exception as e:
        print(f"  [skip] soupx diagnostic plots — import failed: {e}")
        return

    from SoupX.plot import plot_soup_correlation, plot_marker_distribution

    _SC_GETTERS = {
        "toy_pbmc":    get_fitted_sc_toy_pbmc,
        "pbmc_10k":    get_fitted_sc_pbmc10k,
        "fetal_liver": get_fitted_sc_fetal_liver,
    }

    for ds, sc_getter in _SC_GETTERS.items():
        if not _AVAIL.get(ds, lambda: False)():
            print(f"  [skip] {ds} — data not found")
            continue

        print(f"  [soupx-diag] Fitting upg-auto on {ds} ...")
        try:
            sc = sc_getter()
        except Exception as e:
            print(f"  [skip] {ds} — fit failed: {e}")
            continue

        corr_path = os.path.join(out_dir, f"08_{ds}_soup_correlation.png")
        try:
            fig = plot_soup_correlation(sc, save_path=corr_path)
            if fig is not None:
                plt.close(fig)
            print(f"  Saved → {corr_path}")
        except Exception as e:
            print(f"  [warn] soup_correlation failed for {ds}: {e}")

        gene_lists = {k: v for k, v in _DS_MARKERS.get(ds, {}).items()
                      if any(g in list(sc.genes) for g in v)}
        if gene_lists:
            mkr_path = os.path.join(out_dir, f"09_{ds}_marker_distribution.png")
            try:
                fig = plot_marker_distribution(sc, non_expressed_gene_list=gene_lists,
                                               save_path=mkr_path)
                if fig is not None:
                    plt.close(fig)
                print(f"  Saved → {mkr_path}")
            except Exception as e:
                print(f"  [warn] marker_distribution failed for {ds}: {e}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_or_run(args):
    if args.csv:
        df = pd.read_csv(args.csv)
        print(f"  Loaded {len(df)} rows from {args.csv}")
        return df

    from benchmark_full import (
        benchmark_toy_pbmc, benchmark_pbmc10k, benchmark_hgmm,
        benchmark_fetal_liver, benchmark_rep1_zenodo_gt,
        tune_all_pipelines_hgmm, tune_all_pipelines_rep1_zenodo_gt,
        entries_to_dataframe, _AVAIL,
    )
    _RUNNERS = {
        "toy_pbmc":       benchmark_toy_pbmc,
        "pbmc_10k":       benchmark_pbmc10k,
        "hgmm":           benchmark_hgmm,
        "fetal_liver":    benchmark_fetal_liver,
        "rep1_zenodo_gt": benchmark_rep1_zenodo_gt,
    }

    to_run = args.datasets or [k for k, c in _AVAIL.items() if c()]

    hgmm_tuned_params = None
    if "hgmm" in to_run and _AVAIL["hgmm"]():
        print("\n  ── hgmm: running grid search ──")
        try:
            hgmm_tuned_params, _ = tune_all_pipelines_hgmm()
        except Exception:
            import traceback
            print("  WARNING: hgmm tuning failed — using defaults")
            traceback.print_exc()

    rep1_tuned_params = None
    if "rep1_zenodo_gt" in to_run and _AVAIL["rep1_zenodo_gt"]():
        print("\n  ── rep1_zenodo_gt: running grid search ──")
        try:
            rep1_tuned_params, _ = tune_all_pipelines_rep1_zenodo_gt()
        except Exception:
            import traceback
            print("  WARNING: rep1_zenodo_gt tuning failed — using defaults")
            traceback.print_exc()

    all_entries = []
    for key in to_run:
        if not _AVAIL.get(key, lambda: False)():
            print(f"  SKIP {key} — data not found")
            continue
        print(f"  Running {key} ...")
        try:
            if key == "hgmm":
                entries = benchmark_hgmm(skip_decontx=args.skip_decontx,
                                         tuned_params=hgmm_tuned_params)
            elif key == "rep1_zenodo_gt":
                entries = benchmark_rep1_zenodo_gt(skip_decontx=args.skip_decontx,
                                                   tuned_params=rep1_tuned_params)
            else:
                entries = _RUNNERS[key](skip_decontx=args.skip_decontx)
            all_entries.extend(entries)
        except Exception:
            import traceback
            print(f"  ERROR in {key}:")
            traceback.print_exc()

    if not all_entries:
        raise RuntimeError("No benchmark entries produced — check datasets are available.")

    df = entries_to_dataframe(all_entries)
    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"  Results saved to: {args.save_csv}")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate SoupX benchmark visualizations."
    )
    parser.add_argument("--csv", default=None, metavar="PATH",
                        help="Load pre-saved benchmark CSV instead of running benchmark")
    parser.add_argument("--datasets", nargs="+",
                        choices=["toy_pbmc", "pbmc_10k", "hgmm", "fetal_liver",
                                 "rep1_zenodo_gt"],
                        default=None)
    parser.add_argument("--skip-decontx", action="store_true")
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "plots"),
                        metavar="DIR")
    parser.add_argument("--save-csv", default=None, metavar="PATH",
                        help="Save benchmark results to CSV (only used when --csv not given)")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Also generate downstream embedding and SoupX diagnostic plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading benchmark data ...")
    df = load_or_run(args)

    print(f"\nGenerating plots → {args.out_dir}/")
    plot_rho_comparison(df, args.out_dir)
    plot_metric_heatmap(df, args.out_dir)
    plot_gt_metrics(df, args.out_dir)
    plot_cluster_quality(df, args.out_dir)
    plot_radar(df, args.out_dir)
    n_plots = 5

    if args.diagnostics or not args.csv:
        plot_downstream_analysis(args.out_dir)
        plot_soupx_diagnostics(args.out_dir)
        n_plots = 9

    print(f"\nDone. Up to {n_plots} plots saved.")


if __name__ == "__main__":
    main()
