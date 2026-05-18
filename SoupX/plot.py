import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
import warnings


def plot_soup_correlation(sc, save_path=None):
    """
    Scatter plot comparing aggregate cell expression profile vs soup profile.

    Parameters
    ----------
    sc : SoupChannel

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    cell_profile = np.array(sc.toc.sum(axis=1)).flatten().astype(float)
    total = cell_profile.sum()
    if total <= 0:
        raise ValueError("Aggregate cell expression is all-zero; cannot plot correlation.")
    cell_profile = cell_profile / total
    soup_profile = sc.soup_profile['est'].values.astype(float)

    # Restrict scatter to genes expressed in both profiles; log10(0) = -Inf breaks axes.
    pos_mask = (cell_profile > 0) & (soup_profile > 0)
    if not pos_mask.any():
        raise ValueError("No genes with positive expression in both cell and soup profiles.")
    log_cell = np.log10(cell_profile[pos_mask])
    log_soup = np.log10(soup_profile[pos_mask])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(log_cell, log_soup, alpha=0.3, s=5, c='black')
    lim = [min(log_cell.min(), log_soup.min()) - 0.2,
           max(log_cell.max(), log_soup.max()) + 0.2]
    ax.plot(lim, lim, 'k-', linewidth=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('log10(Aggregate cell Expression)')
    ax.set_ylabel('log10(Soup Expression)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_marker_distribution(sc, non_expressed_gene_list=None, max_cells=150,
                               tfidf_min=1.0, save_path=None, **kwargs):
    """
    Violin plot of observed/expected expression ratio for marker gene sets.

    For each gene set, shows the distribution of the ratio of observed counts
    to expected counts if the cell were pure soup. A ratio near the global
    contamination fraction indicates non-expression; much higher ratios indicate
    genuine expression.

    Parameters
    ----------
    sc : SoupChannel
    non_expressed_gene_list : dict or None
        {set_name: [genes]}. If None, top cluster markers are used (exploratory only).
    max_cells : int
        Maximum cells to show as individual points.
    tfidf_min : float
        TF-IDF cut-off when auto-discovering markers.
    **kwargs
        Passed to estimate_non_expressing_cells().

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from .estimation import estimate_non_expressing_cells, calculate_contamination_fraction
    from .markers import quick_markers

    if non_expressed_gene_list is None:
        print("No gene lists provided, attempting to find and plot cluster marker genes.")
        if 'clusters' not in sc.meta_data.columns:
            raise ValueError("No clusters set. Run set_clusters() or provide non_expressed_gene_list.")
        mrks = quick_markers(sc.toc, sc.meta_data['clusters'].values,
                              genes=list(sc.genes), n=np.iinfo(int).max)
        mrks = mrks.sort_values(['gene', 'tfidf'], ascending=[True, False])
        mrks = mrks.drop_duplicates('gene', keep='first').sort_values('tfidf', ascending=False)
        mrks = mrks[mrks['tfidf'] > tfidf_min]
        print(f"Found {len(mrks)} marker genes")
        mrks = mrks.sort_values('soupExp' if 'soupExp' in mrks.columns else 'tfidf', ascending=False)
        top_genes = list(mrks['gene'].head(20))
        non_expressed_gene_list = {g: [g] for g in top_genes}
        print("NOTE: Do NOT use these auto-discovered genes directly for contamination estimation.")

    if not isinstance(non_expressed_gene_list, dict):
        raise TypeError("non_expressed_gene_list must be a dict of {name: [genes]}.")

    null_mat = estimate_non_expressing_cells(sc, non_expressed_gene_list, **kwargs)

    gene_idx = {g: i for i, g in enumerate(sc.genes)}
    toc_arr = sc.toc.tocsc()
    nUMIs = sc.meta_data['nUMIs'].values
    soup_est = sc.soup_profile['est'].values

    # Compute observed/expected ratio for each gene set, per cell
    records = []
    for set_name, gene_list in non_expressed_gene_list.items():
        valid_genes = [g for g in gene_list if g in gene_idx]
        if not valid_genes:
            continue
        g_idx = [gene_idx[g] for g in valid_genes]
        soup_sum = soup_est[g_idx].sum()
        if soup_sum == 0:
            continue
        obs = np.array(toc_arr[np.ix_(g_idx, np.arange(len(sc.cells)))].sum(axis=0)).flatten()
        obs_frac = obs / np.maximum(nUMIs, 1)
        ratio = obs_frac / soup_sum
        for ci, cell in enumerate(sc.cells):
            records.append({'MarkerGroup': set_name, 'Barcode': cell, 'Values': ratio[ci]})

    df = pd.DataFrame(records)
    df = df[df['Values'] > 0]
    df['nUMIs'] = sc.meta_data['nUMIs'].reindex(df['Barcode']).values

    # Expected soup counts
    exp_cnts_dict = {}
    for set_name, gene_list in non_expressed_gene_list.items():
        valid = [g for g in gene_list if g in gene_idx]
        if not valid:
            continue
        g_idx = [gene_idx[g] for g in valid]
        exp_cnts_dict[set_name] = nUMIs * soup_est[g_idx].sum()

    _cell_to_idx = {c: i for i, c in enumerate(sc.cells)}
    df['expCnts'] = df.apply(
        lambda r: exp_cnts_dict[r['MarkerGroup']][_cell_to_idx[r['Barcode']]]
        if r['MarkerGroup'] in exp_cnts_dict and r['Barcode'] in _cell_to_idx else 1.0,
        axis=1
    )

    # Compute global rho per marker group
    global_rhos = {}
    for set_name, gene_list in non_expressed_gene_list.items():
        use_col = null_mat[set_name] if set_name in null_mat.columns else pd.Series(True, index=sc.cells)
        if use_col.sum() == 0:
            global_rhos[set_name] = np.nan
            continue
        try:
            tmp = calculate_contamination_fraction(
                sc, {set_name: gene_list},
                null_mat[[set_name]],
                verbose=False, force_accept=True
            )
            global_rhos[set_name] = float(tmp.meta_data['rho'].iloc[0])
        except Exception:
            global_rhos[set_name] = np.nan

    set_names = list(non_expressed_gene_list.keys())
    keep_cells = np.random.choice(list(sc.cells), size=min(len(sc.cells), max_cells), replace=False)

    fig, ax = plt.subplots(figsize=(max(6, len(set_names) * 1.5), 5))

    for xi, set_name in enumerate(set_names):
        sub = df[df['MarkerGroup'] == set_name]
        vals = np.log10(sub['Values'].replace(0, np.nan).dropna())
        if len(vals) == 0:
            continue
        # Violin
        parts = ax.violinplot(vals, positions=[xi], showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('#cccccc')
            pc.set_alpha(0.6)
        # Jitter for subset of cells
        sub_show = sub[sub['Barcode'].isin(keep_cells)]
        jitter = np.random.uniform(-0.2, 0.2, len(sub_show))
        sizes = np.log10(np.maximum(sub_show['expCnts'].values, 1)) * 5
        ax.scatter(xi + jitter, np.log10(np.maximum(sub_show['Values'].values, 1e-6)),
                   s=np.clip(sizes, 2, 30), alpha=0.4, c='black')
        # Global rho line
        if set_name in global_rhos and not np.isnan(global_rhos[set_name]):
            ax.plot([xi - 0.4, xi + 0.4],
                    [np.log10(global_rhos[set_name])] * 2, 'r-', linewidth=1.5)

    ax.set_xticks(range(len(set_names)))
    ax.set_xticklabels(set_names, rotation=90)
    ax.set_ylabel('log10(observed/expected)')
    ax.set_xlabel('Marker group')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_marker_map(sc, gene_set, dr=None, rat_lims=(-2, 2), fdr=0.05,
                    use_to_est=None, point_size=20, na_point_size=2,
                    save_path=None):
    """
    UMAP/tSNE map coloured by observed/expected expression ratio for a gene set.

    Significant enrichment (Poisson FDR < fdr) is outlined in green.

    Parameters
    ----------
    sc : SoupChannel
    gene_set : str or list of str
        Gene(s) to aggregate and visualise.
    dr : pd.DataFrame, optional
        Dimension reduction coordinates (cells x 2). If None, uses sc.DR.
    rat_lims : tuple
        (min, max) for log10(obs/exp) colour scale.
    fdr : float
        Significance threshold for marking enriched cells.
    use_to_est : array-like, optional
        Boolean mask (length n_cells) to colour cells instead of Poisson test.
    point_size : float
    na_point_size : float

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if isinstance(gene_set, str):
        gene_set = [gene_set]

    # Resolve DR
    if dr is None:
        if sc.DR is None:
            raise ValueError("No dimension reduction found. Set sc.DR or pass dr explicitly.")
        dr = sc.meta_data[sc.DR].copy()
    else:
        dr = pd.DataFrame(dr)
        if not all(c in sc.cells for c in dr.index):
            dr.index = sc.cells[:len(dr)]

    dr = dr.iloc[:, :2].copy()
    dr.columns = ['RD1', 'RD2']

    gene_idx = {g: i for i, g in enumerate(sc.genes)}
    valid = [g for g in gene_set if g in gene_idx]
    if not valid:
        raise ValueError(f"None of {gene_set} found in sc.genes.")
    g_idx = [gene_idx[g] for g in valid]

    toc_csc = sc.toc.tocsc()
    obs = np.array(toc_csc[g_idx, :].sum(axis=0)).flatten()
    soup_sum = sc.soup_profile['est'].values[g_idx].sum()
    exp = sc.meta_data['nUMIs'].values * soup_sum

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(exp > 0, obs / exp, 0.0)
        log_ratio = np.where(ratio > 0, np.log10(ratio), np.nan)

    log_ratio = np.clip(log_ratio, rat_lims[0], rat_lims[1])
    log_ratio[ratio == 0] = np.nan

    # Significance
    if use_to_est is not None:
        sig = np.asarray(use_to_est, dtype=bool)
    else:
        pvals = scipy.stats.poisson.sf(obs - 1, np.maximum(exp, 1e-100))
        from statsmodels.stats.multitest import multipletests
        sig = multipletests(pvals, method='fdr_bh')[0] if pvals.min() < 1 else pvals < fdr

    # Reindex to dr
    cell_list = list(sc.cells)
    log_ratio_aligned = pd.Series(log_ratio, index=sc.cells).reindex(dr.index).values
    sig_aligned = pd.Series(sig, index=sc.cells).reindex(dr.index).values

    # Colour scale: blue → white → red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'bwr_soup', ['#0000cc', '#ffffff', '#cc0000']
    )
    norm = plt.Normalize(vmin=rat_lims[0], vmax=rat_lims[1])

    fig, ax = plt.subplots(figsize=(6, 5))

    # NA points (zero expression)
    na_mask = np.isnan(log_ratio_aligned)
    edge_na = np.where(sig_aligned[na_mask], '#009933', 'black')
    ax.scatter(dr.iloc[na_mask, 0], dr.iloc[na_mask, 1],
               c='lightgrey', s=na_point_size, linewidths=0, zorder=1)

    # Expressed points
    ok_mask = ~na_mask
    edge_ok = np.where(sig_aligned[ok_mask], '#009933', 'black')
    sc_plot = ax.scatter(
        dr.iloc[ok_mask, 0], dr.iloc[ok_mask, 1],
        c=log_ratio_aligned[ok_mask],
        cmap=cmap, norm=norm,
        s=point_size,
        linewidths=0.4,
        edgecolors=edge_ok,
        zorder=2
    )
    plt.colorbar(sc_plot, ax=ax, label='log10(obs/expected)')
    ax.set_xlabel('ReducedDim1')
    ax.set_ylabel('ReducedDim2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return fig


def plot_change_map(sc, cleaned_matrix, gene_set, dr=None,
                    data_type='soupFrac', log_data=False, point_size=2,
                    save_path=None):
    """
    Visualise the change in expression before and after soup correction.

    Parameters
    ----------
    sc : SoupChannel
    cleaned_matrix : scipy.sparse matrix
        Output of adjust_counts().
    gene_set : str or list of str
        Gene(s) to aggregate.
    dr : pd.DataFrame, optional
        Dimension reduction (cells x 2). If None, uses sc.DR.
    data_type : str
        'soupFrac' (default): fraction of expression identified as soup.
        'binary': expressed (>0) or not.
        'counts': raw count values.
    log_data : bool
        Log10-transform values (ignored for binary).
    point_size : float

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if data_type == 'binary':
        log_data = False

    if isinstance(gene_set, str):
        gene_set = [gene_set]

    # Resolve DR
    if dr is None:
        if sc.DR is None:
            raise ValueError("No dimension reduction found. Set sc.DR or pass dr explicitly.")
        dr = sc.meta_data[sc.DR].copy()
    else:
        dr = pd.DataFrame(dr)
        if not all(c in sc.cells for c in dr.index):
            dr.index = sc.cells[:len(dr)]

    dr = dr.iloc[:, :2].copy()
    dr.columns = ['RD1', 'RD2']

    gene_idx = {g: i for i, g in enumerate(sc.genes)}
    valid = [g for g in gene_set if g in gene_idx]
    if not valid:
        raise ValueError(f"None of {gene_set} found in sc.genes.")
    g_idx = [gene_idx[g] for g in valid]

    # Reindex to DR cells
    cell_pos = [list(sc.cells).index(c) for c in dr.index if c in list(sc.cells)]

    old = np.array(sc.toc.tocsc()[np.ix_(g_idx, cell_pos)].sum(axis=0)).flatten()
    new = np.array(cleaned_matrix.tocsc()[np.ix_(g_idx, cell_pos)].sum(axis=0)).flatten()

    orange_cmap = plt.get_cmap('YlOrBr')

    if data_type == 'soupFrac':
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_change = np.where(old > 0, (old - new) / old, np.nan)
        if log_data:
            rel_change = np.log10(np.maximum(rel_change, 1e-10))
            z_lims = (-2, 0)
            label = 'log10(SoupFrac)'
        else:
            z_lims = (0, 1)
            label = 'SoupFrac'
        rel_change = np.clip(rel_change, z_lims[0], z_lims[1])

        fig, ax = plt.subplots(figsize=(6, 5))
        sc_plot = ax.scatter(dr['RD1'], dr['RD2'],
                             c=rel_change, cmap=orange_cmap,
                             vmin=z_lims[0], vmax=z_lims[1],
                             s=point_size, linewidths=0)
        plt.colorbar(sc_plot, ax=ax, label=label)
        ax.set_title('Change in expression due to soup correction')
        ax.set_xlabel('ReducedDim1')
        ax.set_ylabel('ReducedDim2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        return fig

    else:
        # Side-by-side before/after
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        panels = [('Uncorrected', old), ('Corrected', new)]

        for ax, (title, vals) in zip(axes, panels):
            if data_type == 'binary':
                data = (vals > 0).astype(float)
                cmap = plt.get_cmap('RdBu')
                z_lims = (None, None)
            else:
                data = vals.astype(float)
                if log_data:
                    data = np.log10(np.maximum(data, 1e-10))
                z_lims = (None, None)
                cmap = orange_cmap

            sc_plot = ax.scatter(dr['RD1'], dr['RD2'],
                                  c=data, cmap=cmap,
                                  vmin=z_lims[0], vmax=z_lims[1],
                                  s=point_size, linewidths=0)
            plt.colorbar(sc_plot, ax=ax, label='geneSet')
            ax.set_title(title)
            ax.set_xlabel('ReducedDim1')
            ax.set_ylabel('ReducedDim2')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.suptitle('Comparison of before and after correction')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        return fig
