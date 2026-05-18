import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
import warnings

from .set_properties import set_contamination_fraction


def estimate_non_expressing_cells(sc, non_expressed_gene_list, clusters=None,
                                   maximum_contamination=1.0, fdr=0.05):
    """
    Identify cells that genuinely do not express each gene set.

    Uses a Poisson test to flag clusters where any cell shows expression
    significantly above the maximum expected from soup alone.

    Parameters
    ----------
    sc : SoupChannel
    non_expressed_gene_list : dict
        {set_name: [gene1, gene2, ...]} mapping.
    clusters : pd.Series or None or False
        Cell-to-cluster mapping. None = use sc.meta_data['clusters'],
        False = treat each cell as its own cluster.
    maximum_contamination : float
        Upper bound on expected contamination. Higher = more cells excluded.
    fdr : float
        FDR threshold for the Poisson test.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame (cells x gene_sets). True = safe to use for estimation.
    """
    from statsmodels.stats.multitest import multipletests

    if not isinstance(non_expressed_gene_list, dict):
        raise TypeError("non_expressed_gene_list must be a dict of {name: [genes]}.")

    # Resolve clusters
    if clusters is None:
        if 'clusters' in sc.meta_data.columns:
            clusters = sc.meta_data['clusters']
        else:
            clusters = False

    if clusters is False:
        clusters = pd.Series(sc.cells, index=sc.cells)
    else:
        clusters = pd.Series(clusters)

    if not all(c in clusters.index for c in sc.cells):
        raise ValueError("clusters must contain an entry for every cell.")

    # Map cells to cluster IDs
    cell_clusters = clusters.reindex(sc.cells).astype(str)
    unique_clusters = sorted(cell_clusters.unique())
    cluster_to_cells = {cl: sc.cells[cell_clusters == cl] for cl in unique_clusters}

    gene_idx = {g: i for i, g in enumerate(sc.genes)}
    nUMIs = sc.meta_data['nUMIs'].values  # (n_cells,)
    soup_est = sc.soup_profile['est'].values  # (n_genes,)

    set_names = list(non_expressed_gene_list.keys())
    n_sets = len(set_names)

    # Per-cell count and expected count for each gene set
    # cnt[cell_i, set_j] = sum of counts for gene set j in cell i
    # exp[cell_i, set_j] = nUMIs[cell_i] * maximum_contamination * sum(soup_est for set j)
    cnt = np.zeros((len(sc.cells), n_sets))
    exp = np.zeros((len(sc.cells), n_sets))

    toc_csc = sc.toc.tocsc()
    cell_idx_map = {c: i for i, c in enumerate(sc.cells)}

    for j, (set_name, gene_list) in enumerate(non_expressed_gene_list.items()):
        valid_genes = [g for g in gene_list if g in gene_idx]
        if not valid_genes:
            continue
        g_idx = [gene_idx[g] for g in valid_genes]
        set_counts = np.array(toc_csc[g_idx, :].sum(axis=0)).flatten()
        soup_frac = soup_est[g_idx].sum()

        cnt[:, j] = set_counts
        exp[:, j] = nUMIs * maximum_contamination * soup_frac

    # Cluster-level: for each cluster, a cell "passes" for a gene set if
    # no cell in the cluster shows significant overexpression.
    # Test: ppois(cnt-1, exp, lower.tail=FALSE) → P(X >= cnt)
    # If any cell in the cluster has a small p-value → exclude entire cluster.
    pvals = scipy.stats.poisson.sf(cnt - 1, exp)   # (n_cells, n_sets)
    pvals = np.clip(pvals, 0, 1)

    # FDR correct per gene set
    qvals = np.ones_like(pvals)
    for j in range(n_sets):
        pv = pvals[:, j]
        if pv.min() < 1.0:
            qvals[:, j] = multipletests(pv, method='fdr_bh')[1]

    # For each cluster, find minimum q across cells → cluster passes if min_q >= fdr
    clust_pass = {}
    for cl, cells_in_cl in cluster_to_cells.items():
        c_idx = [cell_idx_map[c] for c in cells_in_cl]
        min_q = qvals[c_idx, :].min(axis=0)   # (n_sets,)
        clust_pass[cl] = min_q >= fdr          # True = no significant expression

    # Expand to cell level
    result = pd.DataFrame(False, index=sc.cells, columns=set_names)
    for cl, cells_in_cl in cluster_to_cells.items():
        result.loc[cells_in_cl, :] = clust_pass[cl]

    total_passing = result.values.sum()
    if total_passing == 0:
        warnings.warn(
            "No non-expressing cells identified. Consider setting clusters=False, "
            "increasing maximum_contamination and/or fdr."
        )
    elif total_passing < 100:
        warnings.warn(
            f"Fewer than 100 non-expressing cells identified ({total_passing}). "
            "Contamination estimate may be inaccurate."
        )

    return result


def estimate_cell_rho(sc, soup_quantile=0.9, prior_rho=None, prior_std=0.1,
                       inplace=False):
    """
    Refine contamination fraction to per-cell estimates via empirical Bayes.

    Uses a Gamma-Poisson conjugate model:
      Prior:      rho ~ Gamma(alpha, 1/beta)  with mean = prior_rho
      Likelihood: O_i ~ Poisson(rho * E_i)   where O_i = observed counts of
                  high-soup genes in cell i, E_i = nUMI_i * sum(soup_frac for
                  those genes)
      Posterior mean: rho_i = (alpha + O_i) / (beta + E_i)

    High-UMI cells are dominated by their own data; low-UMI cells shrink
    toward the prior mean.  Requires rho to already be set (call
    auto_est_cont or calculate_contamination_fraction first).

    Parameters
    ----------
    sc : SoupChannel
    soup_quantile : float
        Percentile cutoff on soup fraction: only genes above this quantile
        are used as signal for per-cell estimation (default 0.5 = top half).
    prior_rho : float, optional
        Prior mean contamination fraction.  If None, the mean of the already-
        stored rho values is used.
    prior_std : float
        Prior standard deviation (default 0.1).  Smaller = stronger shrinkage
        toward prior_rho.
    inplace : bool

    Returns
    -------
    SoupChannel
        With sc.meta_data['rho'] replaced by a per-cell array.
    """
    if not inplace:
        sc = sc.copy()

    if 'rho' not in sc.meta_data.columns:
        raise ValueError(
            "rho must be set before calling estimate_cell_rho. "
            "Call auto_est_cont() or calculate_contamination_fraction() first."
        )
    if sc.soup_profile is None:
        raise ValueError("soup_profile must be set before calling estimate_cell_rho.")
    if prior_std <= 0:
        raise ValueError("prior_std must be positive.")

    global_rho = float(np.mean(sc.meta_data['rho'].values))
    if prior_rho is None:
        prior_rho = global_rho

    if prior_std < 0.01:
        _alpha_approx = (prior_rho / prior_std) ** 2
        warnings.warn(
            f"prior_std={prior_std:.4g} is very small; clipping to 0.01 to avoid "
            f"degenerate Gamma prior (alpha_prior ≈ {_alpha_approx:.0f}).",
            UserWarning, stacklevel=2,
        )
        prior_std = 0.01

    soup_frac = sc.soup_profile['est'].values
    soup_threshold = float(np.quantile(soup_frac, soup_quantile))
    soup_mask = soup_frac >= soup_threshold
    soup_frac_sum = float(soup_frac[soup_mask].sum())

    if soup_frac_sum <= 0:
        warnings.warn("No soup marker genes found for per-cell rho estimation; "
                      "rho values unchanged.")
        return sc

    # Observed counts of soup-marker genes per cell
    toc_csc = sc.toc.tocsc()
    O = np.array(toc_csc[soup_mask, :].sum(axis=0)).flatten().astype(float)

    # Expected counts under rho = 1
    nUMIs = sc.meta_data['nUMIs'].values.astype(float)
    E = nUMIs * soup_frac_sum

    # Gamma prior: mean = prior_rho, std = prior_std
    #   Gamma(alpha, scale=1/beta): mean = alpha/beta, var = alpha/beta^2
    alpha_prior = (prior_rho / prior_std) ** 2
    beta_prior = prior_rho / (prior_std ** 2)

    # Posterior mean: E[rho | O_i] = (alpha + O_i) / (beta + E_i)
    # Guard: beta_prior=0 when prior_rho=0; E[i]=0 when nUMIs[i]=0 — avoid 0/0.
    rho_cell = (alpha_prior + O) / np.maximum(beta_prior + E, 1e-12)
    rho_cell = np.clip(rho_cell, 0.0, 1.0)

    sc.meta_data['rho'] = rho_cell
    return sc


def estimate_decontx_rho(sc, prior_rho=None, n_iter=100, tol=1e-4, tol_ll=1e-6,
                          verbose=False, inplace=False):
    """
    Per-cell contamination via DecontX-style Dirichlet-Multinomial EM.

    Models each cell's count vector as a two-component mixture:
      x_i ~ Multinomial(n_i, theta_i * pi_soup + (1 - theta_i) * phi_i)
    where theta_i is the per-cell contamination fraction and phi_i is the
    cell's native expression profile.  Both are jointly estimated via EM.
    Unlike the GLM approach, this uses the full per-gene expression profile
    of each cell directly — no marker genes or cluster labels required.

    Requires sc.soup_profile to be set (call estimate_soup first).

    Parameters
    ----------
    sc : SoupChannel
    prior_rho : float or None
        Initial contamination guess for all cells.  If None, uses the mean
        of sc.meta_data['rho'] if already set, else 0.05.
    n_iter : int
        Maximum EM iterations (default 100).
    tol : float
        Parameter-delta convergence threshold on max |Δtheta_i| (default 1e-4).
    tol_ll : float
        Relative log-likelihood convergence threshold (default 1e-6). EM also
        stops when |LL_t - LL_{t-1}| / |LL_{t-1}| < tol_ll, preventing
        max-iter exhaustion when theta oscillates near a fixed point.
    verbose : bool
    inplace : bool

    Returns
    -------
    SoupChannel
        With sc.meta_data['rho'] set to a per-cell contamination array.
    """
    if not inplace:
        sc = sc.copy()

    if sc.soup_profile is None:
        raise ValueError("soup_profile must be set before calling estimate_decontx_rho.")

    if prior_rho is None:
        if 'rho' in sc.meta_data.columns:
            prior_rho = float(np.mean(sc.meta_data['rho'].values))
        else:
            prior_rho = 0.05
    prior_rho = float(np.clip(prior_rho, 1e-4, 1.0 - 1e-4))

    theta = _decontx_em(sc, prior_rho=prior_rho, n_iter=n_iter, tol=tol,
                        tol_ll=tol_ll, verbose=verbose)
    sc.meta_data['rho'] = theta

    if verbose:
        print(f"DecontX: mean rho={float(theta.mean()):.4f}, "
              f"std={float(theta.std()):.4f}, "
              f"range=[{float(theta.min()):.4f}, {float(theta.max()):.4f}]")

    return sc


def calculate_contamination_fraction(sc, non_expressed_gene_list, use_to_est,
                                      verbose=True, force_accept=False,
                                      cell_level_rho=False, cell_rho_method=None,
                                      inplace=False):
    """
    Estimate global contamination fraction using known non-expressing gene sets.

    Fits a Poisson GLM: counts ~ offset(log(expected_soup_counts)).

    Parameters
    ----------
    sc : SoupChannel
    non_expressed_gene_list : dict
        {set_name: [genes]} — gene sets assumed non-expressed in some cells.
    use_to_est : pd.DataFrame
        Boolean (cells x set_names) from estimate_non_expressing_cells().
    verbose : bool
    force_accept : bool
    cell_level_rho : bool
        If True, refine the global rho estimate to per-cell values using
        estimate_cell_rho() after the GLM fit (default False).
        Ignored when cell_rho_method is set.
    cell_rho_method : str or None
        None (default): global rho only (or use cell_level_rho flag).
        'empirical_bayes': per-cell rho via Gamma-Poisson conjugate shrinkage
        (same as cell_level_rho=True).
        'glm': per-cell rho from a Poisson GLM that includes log(nUMI) as a
        covariate.  Cells with high total UMI counts get lower proportional
        contamination.  Sets sc.meta_data['rho'] to a per-cell array.
        'decontx': per-cell rho via DecontX-style Dirichlet-Multinomial EM.
        Models each cell as a mixture of ambient (soup) and native expression;
        EM jointly estimates the per-cell contamination fraction theta_i and
        native profile phi_i.  More accurate than 'glm' on heterogeneous data.
    inplace : bool

    Returns
    -------
    SoupChannel
        With rho, rhoLow, rhoHigh set in meta_data and fit stored in sc.fit.
    """
    import statsmodels.api as sm

    if not inplace:
        sc = sc.copy()

    if not isinstance(non_expressed_gene_list, dict):
        raise TypeError("non_expressed_gene_list must be a dict of {name: [genes]}.")
    if use_to_est.values.sum() == 0:
        raise ValueError("No cells specified as acceptable for estimation.")

    gene_idx = {g: i for i, g in enumerate(sc.genes)}
    nUMIs = sc.meta_data['nUMIs'].values
    soup_est = sc.soup_profile['est'].values
    toc_csc = sc.toc.tocsc()

    records = []
    for i, (set_name, gene_list) in enumerate(non_expressed_gene_list.items()):
        valid_genes = [g for g in gene_list if g in gene_idx]
        if not valid_genes:
            continue
        g_idx = [gene_idx[g] for g in valid_genes]
        soup_frac = float(soup_est[g_idx].sum())

        # Cells to use for this gene set
        use_cells = use_to_est.index[use_to_est.iloc[:, i]].tolist()
        if not use_cells:
            continue

        cell_pos_map = {c: i for i, c in enumerate(sc.cells)}
        cell_pos = [cell_pos_map[c] for c in use_cells]
        counts_per_cell = np.array(toc_csc[np.ix_(g_idx, cell_pos)].sum(axis=0)).flatten()
        nUMIs_use = nUMIs[cell_pos]
        exp_soup = nUMIs_use * soup_frac

        for cnt, exp, cell in zip(counts_per_cell, exp_soup, use_cells):
            records.append({
                'cell': cell,
                'gene_set': set_name,
                'soup_frac': soup_frac,
                'counts': cnt,
                'nUMIs': exp / soup_frac if soup_frac > 0 else 0,
                'expSoupCnts': exp,
            })

    df = pd.DataFrame(records)
    if len(df) == 0 or df['expSoupCnts'].sum() == 0:
        raise ValueError("No valid data for GLM fitting.")

    if cell_rho_method == 'glm':
        rho, rho_low, rho_high, result, rho_per_cell = _fit_poisson_glm_numi(
            df, sc, verbose
        )
        sc = set_contamination_fraction(sc, rho, force_accept=force_accept, inplace=True)
        sc.meta_data['rhoLow'] = rho_low
        sc.meta_data['rhoHigh'] = rho_high
        sc.meta_data['rho'] = rho_per_cell
        sc.fit = result
    else:
        # Poisson GLM: log(E[counts]) = intercept + log(expSoupCnts)
        X = np.ones((len(df), 1))
        offset = np.log(np.maximum(df['expSoupCnts'].values, 1e-10))

        try:
            glm = sm.GLM(df['counts'].values, X,
                          family=sm.families.Poisson(link=sm.families.links.Log()),
                          offset=offset)
            result = glm.fit(disp=False)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Poisson GLM failed (singular matrix): {exc}. "
                "Check for degenerate or zero expSoupCnts values."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Poisson GLM fitting failed: {exc}") from exc

        rho = float(np.exp(result.params[0]))
        ci = np.asarray(result.conf_int())
        rho_low = float(np.exp(ci[0, 0]))
        rho_high = float(np.exp(ci[0, 1]))

        if verbose:
            print(f"Estimated global contamination fraction of {rho * 100:.2f}%")

        sc = set_contamination_fraction(sc, rho, force_accept=force_accept, inplace=True)
        sc.meta_data['rhoLow'] = rho_low
        sc.meta_data['rhoHigh'] = rho_high
        sc.fit = result

        if cell_rho_method == 'decontx':
            sc = estimate_decontx_rho(sc, prior_rho=rho, inplace=True)
        elif cell_level_rho or cell_rho_method == 'empirical_bayes':
            sc = estimate_cell_rho(sc, prior_rho=rho, inplace=True)

    return sc


def auto_est_cont(sc, top_markers=None, tfidf_min=1.0, soup_quantile=0.90,
                   max_markers=100, contamination_range=(0.01, 0.8),
                   rho_max_fdr=0.2, prior_rho=0.05, prior_rho_std_dev=0.10,
                   do_plot=True, force_accept=False, cell_level_rho=False,
                   cell_rho_method=None, verbose=True, inplace=False):
    """
    Automatically estimate the contamination fraction.

    Uses TF-IDF marker genes that are also highly expressed in the soup.
    Estimates rho per cluster/gene pair, then aggregates via a Bayesian
    posterior (gamma prior) to find the global MAP estimate.

    Parameters
    ----------
    sc : SoupChannel
        Must have clusters set (call set_clusters first).
    top_markers : pd.DataFrame, optional
        Pre-computed markers. If None, computed via quick_markers().
    tfidf_min : float
        Minimum TF-IDF score for a marker gene.
    soup_quantile : float
        Minimum soup expression quantile for a marker gene.
    max_markers : int
        Maximum number of marker genes to use.
    contamination_range : tuple
        (min_rho, max_rho) — constrains the search.
    rho_max_fdr : float
        FDR threshold passed to estimate_non_expressing_cells.
    prior_rho : float
        Mode of the gamma prior on rho.
    prior_rho_std_dev : float
        Std dev of the gamma prior on rho.
    do_plot : bool
        Whether to plot the posterior density.
    force_accept : bool
    cell_level_rho : bool
        If True, refine the global MAP rho to per-cell estimates via
        estimate_cell_rho() after global estimation (default False).
        Ignored when cell_rho_method is set.
    cell_rho_method : str or None
        None (default): use cell_level_rho flag behaviour.
        'empirical_bayes': per-cell rho via Gamma-Poisson conjugate shrinkage.
        'glm': per-cell rho from a Poisson GLM with log(nUMI) covariate fitted
        on the same marker genes used for global estimation.
        'decontx': per-cell rho via DecontX-style Dirichlet-Multinomial EM
        (see estimate_decontx_rho).  Does not require marker genes.
    verbose : bool
    inplace : bool

    Returns
    -------
    SoupChannel
        With global contamination fraction set and sc.fit populated.
    """
    from .markers import quick_markers

    if not inplace:
        sc = sc.copy()

    if 'clusters' not in sc.meta_data.columns:
        raise ValueError("Clustering must be set first. Call set_clusters().")

    clusters = sc.meta_data['clusters'].astype(str)
    unique_clusters = sorted(clusters.unique())

    # ── Step 1: collapse to cluster level ────────────────────────────────────
    n_genes = len(sc.genes)
    n_clusters = len(unique_clusters)
    clust_to_idx = {cl: i for i, cl in enumerate(unique_clusters)}

    clust_counts = np.zeros((n_genes, n_clusters))
    clust_nUMIs = np.zeros(n_clusters)
    toc_csc = sc.toc.tocsc()

    for cl, ci in clust_to_idx.items():
        mask = (clusters == cl).values
        clust_counts[:, ci] = np.array(toc_csc[:, mask].sum(axis=1)).flatten()
        clust_nUMIs[ci] = sc.meta_data.loc[mask, 'nUMIs'].sum()

    clust_counts_df = pd.DataFrame(
        clust_counts, index=sc.genes, columns=unique_clusters
    )
    clust_meta = pd.DataFrame({'nUMIs': clust_nUMIs}, index=unique_clusters)

    # ── Step 2: find marker genes ─────────────────────────────────────────────
    soup_prof = sc.soup_profile.sort_values('est', ascending=False)
    soup_min = float(np.quantile(soup_prof['est'], soup_quantile))

    if top_markers is None:
        mrks = quick_markers(sc.toc, clusters.values, genes=list(sc.genes), n=np.iinfo(int).max)
        # Keep only most specific entry per gene
        if mrks.empty:
            mrks = pd.DataFrame(columns=['gene', 'cluster', 'tfidf'])
        else:
            mrks = mrks.sort_values(['gene', 'tfidf'], ascending=[True, False])
            mrks = mrks.drop_duplicates('gene', keep='first')
            mrks = mrks.sort_values('tfidf', ascending=False)
            mrks = mrks[mrks['tfidf'] > tfidf_min]
    else:
        mrks = top_markers

    # Filter to genes highly expressed in soup
    soup_genes = set(soup_prof.index[soup_prof['est'] > soup_min])
    filt = mrks[mrks['gene'].isin(soup_genes)]
    tgts = list(filt['gene'].head(max_markers))

    if verbose:
        print(
            f"{len(mrks)} genes passed tf-idf cut-off and {len(filt)} soup quantile filter. "
            f"Taking the top {len(tgts)}."
        )

    if len(tgts) == 0:
        raise ValueError(
            "No plausible marker genes found. Is the channel low complexity? "
            "If not, reduce tfidf_min or soup_quantile."
        )
    if len(tgts) < 10:
        warnings.warn(
            "Fewer than 10 marker genes found. Consider reducing tfidf_min or soup_quantile."
        )

    # ── Step 3: get non-expressing cell estimates per cluster/gene ────────────
    tgts_dict = {g: [g] for g in tgts}
    ute = estimate_non_expressing_cells(
        sc, tgts_dict,
        maximum_contamination=max(contamination_range),
        fdr=rho_max_fdr
    )
    # ute is (n_cells x n_tgts). Collapse to cluster level using any().
    #
    # Conservative-any semantics (inherited from R SoupX): a cluster is
    # excluded from contamination estimation for gene g if ANY cell in it
    # passes the non-expressing test.  This is intentionally conservative —
    # we prefer false negatives (fewer usable gene-cluster pairs) over false
    # positives (using a cluster whose aggregate counts include cells that
    # genuinely express g).  The consequence is that cluster-level observed
    # counts used downstream include ALL cells in the cluster, even those that
    # passed the non-expressing filter; the filter only gates which clusters
    # are trusted for estimation, not which cells are summed.
    ute_clust = pd.DataFrame(False, index=unique_clusters, columns=tgts)
    for cl in unique_clusters:
        mask = (clusters == cl).values
        ute_clust.loc[cl] = ute.loc[mask].any(axis=0).values

    # ── Step 4: compute observed / expected rho per gene × cluster ────────────
    soup_est = sc.soup_profile.loc[tgts, 'est'].values   # (n_tgts,)
    # expCnts[gene, cluster] = soupEst[gene] * nUMIs[cluster]
    exp_cnts = np.outer(soup_est, clust_nUMIs)           # (n_tgts, n_clusters)
    obs_cnts = clust_counts_df.loc[tgts].values          # (n_tgts, n_clusters)

    # pass_non_exp[gene, cluster] = ute_clust[cluster, gene]
    pass_non_exp = ute_clust[tgts].T.values              # (n_tgts, n_clusters)

    # p-value: ppois(obs, exp*max_cont, lower.tail=TRUE)
    pp = scipy.stats.poisson.cdf(obs_cnts, exp_cnts * max(contamination_range))
    from statsmodels.stats.multitest import multipletests
    qq = np.array([
        multipletests(pp[g, :], method='fdr_bh')[1]
        for g in range(len(tgts))
    ])  # (n_tgts, n_clusters)

    with np.errstate(divide='ignore', invalid='ignore'):
        rhos = np.where(exp_cnts > 0, obs_cnts / exp_cnts, np.inf)

    # Build flat records
    records = []
    tgt_gene_arr = np.array(tgts)
    mrks_indexed = mrks.set_index('gene') if 'gene' in mrks.columns else mrks
    soup_exp_arr = sc.soup_profile.loc[tgts, 'est'].values

    for gi, gene in enumerate(tgts):
        tfidf_val = float(mrks_indexed.loc[gene, 'tfidf']) if gene in mrks_indexed.index else np.nan
        for ci, cl in enumerate(unique_clusters):
            rho_idx = int(scipy.stats.rankdata(rhos[gi, :])[ci])
            records.append({
                'gene': gene,
                'cluster': cl,
                'passNonExp': bool(pass_non_exp[gi, ci]),
                'rhoEst': float(rhos[gi, ci]),
                'rhoIdx': rho_idx,
                'obsCnt': float(obs_cnts[gi, ci]),
                'expCnt': float(exp_cnts[gi, ci]),
                'isExpressedFDR': float(qq[gi, ci]),
                'tfidf': tfidf_val,
                'soupExp': float(soup_exp_arr[gi]),
                'useEst': bool(pass_non_exp[gi, ci]) and float(exp_cnts[gi, ci]) > 0,
            })

    dd = pd.DataFrame(records)

    if dd['useEst'].sum() < 10:
        warnings.warn(
            "Fewer than 10 independent estimates; rho estimation may be unstable. "
            "Consider reducing tfidf_min or increasing soup_quantile."
        )
    if verbose:
        print(f"Using {dd['useEst'].sum()} independent estimates of rho.")

    # ── Step 5: Bayesian posterior aggregation ────────────────────────────────
    # Gamma prior: mode=prior_rho, std=prior_rho_std_dev
    # Guard: mode parameterization valid only when std_dev < rho (CV < 1).
    if prior_rho_std_dev >= prior_rho:
        _clamped = 0.99 * prior_rho
        warnings.warn(
            f"prior_rho_std_dev ({prior_rho_std_dev:.4g}) >= prior_rho ({prior_rho:.4g}); "
            f"clamping to {_clamped:.4g} to avoid degenerate Gamma prior.",
            UserWarning, stacklevel=2,
        )
        prior_rho_std_dev = _clamped
    v2 = (prior_rho_std_dev / prior_rho) ** 2
    k = 1.0 + (v2 ** -2) / 2.0 * (1.0 + np.sqrt(1.0 + 4.0 * v2))
    theta = prior_rho / (k - 1.0)

    rho_probes = np.arange(0.0, 1.001, 0.001)
    use_dd = dd[dd['useEst']]

    if len(use_dd) == 0:
        raise ValueError(
            "All gene×cluster estimates have useEst=False — posterior has no data. "
            "Reduce tfidf_min, rho_max_fdr, or soup_quantile to allow more estimates through."
        )

    obs_arr = use_dd['obsCnt'].values
    exp_arr = use_dd['expCnt'].values
    obs_int = np.round(obs_arr).astype(int)

    # Joint posterior: log p(rho | data) = log prior(rho) + Σ_i log p(O_i | rho·E_i).
    # Product of independent Poisson likelihoods × Gamma prior — proper Bayesian update.
    # Old code used np.mean(gamma.pdf(...)) which is a mixture, not a joint posterior.
    log_posterior = np.array([
        scipy.stats.gamma.logpdf(rho, a=k, scale=theta) +
        np.sum(scipy.stats.poisson.logpmf(
            obs_int,
            np.maximum(rho * exp_arr, 1e-300)
        ))
        for rho in rho_probes
    ])
    log_posterior -= log_posterior.max()          # numerical stability before exp
    posterior = np.exp(log_posterior)
    _step = float(rho_probes[1] - rho_probes[0])
    _norm = posterior.sum() * _step
    if _norm > 0:
        posterior /= _norm                        # normalise to proper density

    prior_curve = scipy.stats.gamma.pdf(rho_probes, a=k, scale=theta)

    # MAP within contamination_range
    in_range = (rho_probes >= contamination_range[0]) & (rho_probes <= contamination_range[1])
    rho_est = float(rho_probes[in_range][np.argmax(posterior[in_range])])
    half_max = posterior[in_range].max() / 2.0
    fwhm_rhos = rho_probes[in_range][posterior[in_range] >= half_max]
    rho_fwhm = (float(fwhm_rhos.min()), float(fwhm_rhos.max())) if len(fwhm_rhos) > 0 else (rho_est, rho_est)

    # Warn when MAP is pinned at a contamination_range boundary — posterior may be cut off.
    if rho_est <= contamination_range[0] + 0.001:
        warnings.warn(
            f"MAP rho ({rho_est:.3f}) is at the lower boundary of contamination_range "
            f"{contamination_range}. Posterior may peak below the allowed range; "
            "consider lowering contamination_range[0].",
            UserWarning, stacklevel=2,
        )
    elif rho_est >= contamination_range[1] - 0.001:
        warnings.warn(
            f"MAP rho ({rho_est:.3f}) is at the upper boundary of contamination_range "
            f"{contamination_range}. Posterior may peak above the allowed range; "
            "consider raising contamination_range[1].",
            UserWarning, stacklevel=2,
        )

    if verbose:
        print(f"Estimated global rho of {rho_est:.2f}")

    # ── Confidence intervals on per-record rho ────────────────────────────────
    alpha = 0.025
    dd['rhoHigh'] = [
        scipy.stats.gamma.ppf(1 - alpha, a=x + 1) / e if e > 0 else np.inf
        for x, e in zip(dd['obsCnt'], dd['expCnt'])
    ]
    dd['rhoLow'] = [
        (0.0 if x == 0 else scipy.stats.gamma.ppf(alpha, a=x) / e) if e > 0 else 0.0
        for x, e in zip(dd['obsCnt'], dd['expCnt'])
    ]

    if do_plot:
        _plot_posterior(rho_probes, posterior, prior_curve, rho_est, rho_fwhm,
                        prior_rho, prior_rho_std_dev)

    sc.fit = {
        'dd': dd,
        'prior_rho': prior_rho,
        'prior_rho_std_dev': prior_rho_std_dev,
        'posterior': posterior,
        'rho_est': rho_est,
        'rho_fwhm': rho_fwhm,
        'markers_used': mrks,
    }

    sc = set_contamination_fraction(sc, rho_est, force_accept=force_accept, inplace=True)

    if cell_rho_method == 'glm':
        sc = _fit_glm_cell_rho_from_markers(sc, tgts, verbose)
    elif cell_rho_method == 'decontx':
        sc = estimate_decontx_rho(sc, prior_rho=rho_est, inplace=True)
    elif cell_level_rho or cell_rho_method == 'empirical_bayes':
        sc = estimate_cell_rho(sc, prior_rho=rho_est, inplace=True)

    return sc


def _fit_poisson_glm_numi(df, sc, verbose):
    """
    Poisson GLM with log(nUMI) covariate → per-cell rho array.

    Model:  log(E[counts]) = beta0 + beta1*log(nUMI) + log(expSoupCnts)
    Interpretation: rho_i = exp(beta0) * nUMI_i^beta1
    Negative beta1 → higher-UMI cells have lower proportional contamination.

    Returns (rho_global, rho_low, rho_high, result, rho_per_cell)
    where rho_global is the geometric-mean rho across all cells.
    """
    import statsmodels.api as sm

    log_numi_df = np.log(np.maximum(df['nUMIs'].values, 1.0))
    X = np.column_stack([np.ones(len(df)), log_numi_df])
    offset = np.log(np.maximum(df['expSoupCnts'].values, 1e-10))

    glm = sm.GLM(df['counts'].values, X,
                  family=sm.families.Poisson(link=sm.families.links.Log()),
                  offset=offset)
    result = glm.fit(disp=False)

    beta0, beta1 = float(result.params[0]), float(result.params[1])

    nUMIs_all = sc.meta_data['nUMIs'].values.astype(float)
    log_numi_all = np.log(np.maximum(nUMIs_all, 1.0))
    rho_per_cell = np.clip(np.exp(beta0 + beta1 * log_numi_all), 0.0, 1.0)

    rho_global = float(np.exp(np.mean(np.log(np.maximum(rho_per_cell, 1e-10)))))

    # CI via delta method evaluated at the mean log(nUMI)
    ci = np.asarray(result.conf_int())
    mean_log_numi = float(log_numi_all.mean())
    rho_low = float(np.clip(np.exp(ci[0, 0] + ci[1, 0] * mean_log_numi), 0.0, 1.0))
    rho_high = float(np.clip(np.exp(ci[0, 1] + ci[1, 1] * mean_log_numi), 0.0, 1.0))

    if verbose:
        print(
            f"GLM cell-level rho: baseline exp(beta0)={np.exp(beta0):.4f}, "
            f"nUMI effect beta1={beta1:.4f}. "
            f"Geometric mean rho={rho_global * 100:.2f}%"
        )

    return rho_global, rho_low, rho_high, result, rho_per_cell


def _fit_glm_cell_rho_from_markers(sc, tgts, verbose):
    """
    Poisson GLM with log(nUMI) covariate for auto_est_cont's per-cell rho.

    Uses the same marker genes (tgts) that drove the global MAP estimate.
    Collapses all markers into one observation per cell (sum of counts /
    sum of expected soup fractions), then fits the nUMI-covariate GLM.
    Falls back to empirical Bayes if the GLM fails.
    """
    import statsmodels.api as sm

    gene_idx = {g: i for i, g in enumerate(sc.genes)}
    valid_tgts = [g for g in tgts if g in gene_idx]
    if not valid_tgts:
        warnings.warn("No valid marker genes for GLM cell rho; falling back to empirical Bayes.")
        return estimate_cell_rho(sc, inplace=True)

    soup_est = sc.soup_profile['est'].values
    g_idx = [gene_idx[g] for g in valid_tgts]
    soup_frac_sum = float(soup_est[g_idx].sum())
    if soup_frac_sum <= 0:
        warnings.warn("Marker soup fraction is zero; falling back to empirical Bayes.")
        return estimate_cell_rho(sc, inplace=True)

    toc_csc = sc.toc.tocsc()
    obs = np.array(toc_csc[g_idx, :].sum(axis=0)).flatten().astype(float)  # (n_cells,)
    nUMIs = sc.meta_data['nUMIs'].values.astype(float)
    exp_soup = nUMIs * soup_frac_sum                                         # (n_cells,)

    valid = exp_soup > 0
    if valid.sum() < 2:
        warnings.warn("Too few valid cells for GLM cell rho; falling back to empirical Bayes.")
        return estimate_cell_rho(sc, inplace=True)

    log_numi = np.log(np.maximum(nUMIs[valid], 1.0))
    X = np.column_stack([np.ones(int(valid.sum())), log_numi])
    offset = np.log(np.maximum(exp_soup[valid], 1e-10))

    try:
        glm = sm.GLM(obs[valid], X,
                      family=sm.families.Poisson(link=sm.families.links.Log()),
                      offset=offset)
        result = glm.fit(disp=False)
    except (np.linalg.LinAlgError, ValueError) as exc:
        warnings.warn(
            f"GLM cell rho fitting failed ({type(exc).__name__}: {exc}); "
            "falling back to empirical Bayes.",
            UserWarning, stacklevel=2,
        )
        return estimate_cell_rho(sc, inplace=True)
    except Exception as exc:
        exc_msg = str(exc).lower()
        if any(kw in exc_msg for kw in ('singular', 'convergence', 'perfect separation')):
            warnings.warn(
                f"GLM cell rho fitting failed ({type(exc).__name__}: {exc}); "
                "falling back to empirical Bayes.",
                UserWarning, stacklevel=2,
            )
            return estimate_cell_rho(sc, inplace=True)
        raise

    beta0, beta1 = float(result.params[0]), float(result.params[1])
    log_numi_all = np.log(np.maximum(nUMIs, 1.0))
    rho_per_cell = np.clip(np.exp(beta0 + beta1 * log_numi_all), 0.0, 1.0)

    if verbose:
        rho_geo = float(np.exp(np.mean(np.log(np.maximum(rho_per_cell, 1e-10)))))
        print(
            f"GLM cell-level rho (markers): baseline={np.exp(beta0):.4f}, "
            f"nUMI effect={beta1:.4f}, geometric mean rho={rho_geo * 100:.2f}%"
        )

    sc.meta_data['rho'] = rho_per_cell
    return sc


def _decontx_em(sc, prior_rho=0.05, n_iter=100, tol=1e-4, tol_ll=1e-6,
                verbose=False):
    """
    Simple DecontX-style Dirichlet-Multinomial EM — no LDA topics.

    WHEN TO USE vs. _decontx_lda_em (decontx.py)
    ---------------------------------------------
    This function (no-topics):
      - phi_i is estimated independently per cell from its own counts.
        Rare cell types with few UMIs get poor phi estimates, but that is
        acceptable here because this function is used only for refinement
        *after* a cluster-level rho has already been set by auto_est_cont or
        calculate_contamination_fraction.
      - O(nnz) per EM iteration; converges in ~20-50 iterations.
      - float64 throughout.
      - Called by: estimate_decontx_rho, estimate_cell_rho(cell_rho_method='decontx').
      - Max iterations default: 100.

    _decontx_lda_em (decontx.py):
      - phi_i = Pi_topics[i] @ Beta — shared LDA topics let rare cell types
        borrow strength across the dataset.  This is the full Yang et al. 2020
        DecontX model.
      - Significantly more expensive: O(nnz × n_topics) per iteration.
      - float32 for memory efficiency.
      - Called by: run_decontx only.
      - Max iterations default: 500.

    The two implementations intentionally share no code because their priors,
    float precision, convergence criteria, and E/M-step mathematics diverge
    once topics are introduced.

    Model
    -----
    x_i ~ Multinomial(n_i, theta_i * pi + (1 - theta_i) * phi_i)
      theta_i  — per-cell contamination fraction
      pi       — soup profile (fixed, known)
      phi_i    — native expression profile (estimated per cell, no topics)

    Beta(alpha_theta, beta_theta) prior on theta, symmetric Dirichlet on phi.
    Works at non-zero positions of X only — no dense (n_cells × n_genes) matrix.

    Returns
    -------
    theta : np.ndarray, shape (n_cells,)
    """
    pi = sc.soup_profile['est'].values.astype(float)
    pi = pi / (pi.sum() + 1e-10)
    n_genes = len(pi)

    X_csr  = sc.toc.T.tocsr().astype(float)         # (n_cells, n_genes) sparse
    n_cells = X_csr.shape[0]
    n_umis  = np.asarray(X_csr.sum(axis=1)).flatten()

    X_coo  = X_csr.tocoo()
    rows   = X_coo.row.astype(np.int32)
    cols   = X_coo.col.astype(np.int32)
    x_vals = X_coo.data.astype(float)
    pi_nz  = pi[cols]

    # Beta prior centered at prior_rho with concentration 10 (weak)
    conc = 10.0
    alpha_theta = prior_rho * conc
    beta_theta  = (1.0 - prior_rho) * conc
    alpha_phi   = 1.0 / n_genes                     # symmetric Dirichlet

    eps   = 1e-10
    Theta = np.full(n_cells, prior_rho)
    Phi_nz = x_vals / np.maximum(n_umis[rows], 1.0) # phi at non-zero positions

    delta   = np.inf
    ll      = -np.inf
    prev_ll = -np.inf

    for it in range(n_iter):
        Theta_prev = Theta.copy()

        # E-step at non-zero positions
        theta_nz = Theta[rows]
        amb_nz   = theta_nz * pi_nz
        nat_nz   = (1.0 - theta_nz) * Phi_nz
        mix_nz   = amb_nz + nat_nz + eps
        r_nz     = amb_nz / mix_nz

        prev_ll = ll
        ll = float(np.dot(x_vals, np.log(mix_nz)))

        N_nz = x_vals * (1.0 - r_nz)

        # M-step: Theta
        A_total = np.bincount(rows, weights=x_vals * r_nz, minlength=n_cells)
        Theta   = (A_total + alpha_theta) / (n_umis + alpha_theta + beta_theta)
        Theta   = np.clip(Theta, 0.0, 1.0)

        # M-step: Phi at non-zero positions (zero positions contribute alpha_phi
        # to N_sum via n_genes * alpha_phi term but have zero native counts)
        N_sum_per_cell = (
            np.bincount(rows, weights=N_nz, minlength=n_cells)
            + n_genes * alpha_phi
        )
        Phi_nz = (N_nz + alpha_phi) / np.maximum(N_sum_per_cell[rows], eps)

        delta  = float(np.max(np.abs(Theta - Theta_prev)))
        rel_ll = (abs(ll - prev_ll) / (abs(prev_ll) + eps)
                  if np.isfinite(prev_ll) else np.inf)

        if delta < tol or (it > 0 and rel_ll < tol_ll):
            if verbose:
                reason = "Δθ" if delta < tol else "rel-LL"
                print(f"DecontX EM converged at iteration {it + 1} "
                      f"({reason}: delta={delta:.2e}, rel_ll={rel_ll:.2e})")
            break
    else:
        if verbose:
            print(f"DecontX EM: max_iter={n_iter} reached "
                  f"(final delta={delta:.2e}, rel_ll={rel_ll:.2e})")

    return Theta


def _plot_posterior(rho_probes, posterior, prior_curve, rho_est, rho_fwhm,
                    prior_rho, prior_rho_std_dev):
    """
    Plot the posterior and prior density curves for the contamination fraction.

    :param rho_probes: Grid of rho values in [0, 1].
    :type rho_probes: np.ndarray
    :param posterior: Normalised posterior density evaluated at rho_probes.
    :type posterior: np.ndarray
    :param prior_curve: Prior (Gamma) density evaluated at rho_probes.
    :type prior_curve: np.ndarray
    :param rho_est: MAP estimate of rho (vertical red line).
    :type rho_est: float
    :param rho_fwhm: (low, high) full-width at half-maximum of the posterior.
    :type rho_fwhm: tuple
    :param prior_rho: Mode of the Gamma prior (shown in legend).
    :type prior_rho: float
    :param prior_rho_std_dev: Standard deviation of the Gamma prior.
    :type prior_rho_std_dev: float
    :return: The matplotlib Figure.
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rho_probes, posterior, 'b-', label=f'posterior rho {rho_est:.2g} ({rho_fwhm[0]:.2g},{rho_fwhm[1]:.2g})')
    ax.plot(rho_probes, prior_curve, 'k--', label=f'prior rho {prior_rho:.2g}(+/-{prior_rho_std_dev:.2g})')
    ax.axvline(rho_est, color='red', label='rho max')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Contamination Fraction')
    ax.set_ylabel('Probability Density')
    ax.legend(loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    return fig
