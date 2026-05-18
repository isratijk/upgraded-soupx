import numpy as np
import pandas as pd
import scipy.stats
import warnings


def estimate_soup(sc, soup_range=(0, 100), method='fixed', alpha=0.05,
                  keep_droplets=False, inplace=False):
    """
    Estimate the soup (ambient RNA) expression profile from empty droplets.

    Parameters
    ----------
    sc : SoupChannel
        Must have tod and n_drop_umis populated.
    soup_range : tuple of (float, float)
        (lower, upper): droplets with UMI count strictly inside this range
        define the candidate pool.
    method : str
        'fixed' (default): use all droplets in soup_range directly.
        'statistical': seed ambient profile from the lowest-UMI droplets in
        the range, then apply a two-sided Poisson test to accept/reject
        higher-UMI droplets before adding them to the profile.  This
        excludes low-quality cells that accidentally fall in the range.
        'emptydrops': emptyDrops-style selection using a quasi-Poisson
        (negative binomial) overdispersion model.  Seeds the ambient profile
        from the lowest-UMI barcodes, estimates per-gene overdispersion from
        those seeds, then for each candidate barcode computes an
        overdispersion-corrected Pearson chi-squared statistic and applies
        Benjamini-Hochberg FDR correction.  Barcodes that fail to reject the
        ambient null (FDR-adjusted p > alpha) are retained as empty droplets.
        Produces a more accurate soup profile than 'statistical' when the
        data are overdispersed.
    alpha : float
        Significance level for statistical/emptydrops methods (default 0.05).
        Droplets with two-sided Poisson p > alpha (statistical) or
        FDR-adjusted p > alpha (emptydrops) are kept as ambient.
    keep_droplets : bool
        If False (default), tod is set to None after estimation to save memory.
    inplace : bool
        Modify sc in place. Default False returns a modified copy.

    Returns
    -------
    SoupChannel
        Updated with soup_profile set.
    """
    if not inplace:
        sc = sc.copy()

    if sc.tod is None:
        raise ValueError(
            "sc.tod is None. Cannot estimate soup without the full droplet table."
        )

    umis = sc.n_drop_umis.values
    lower, upper = float(soup_range[0]), float(soup_range[1])

    if method == 'fixed':
        w = np.where((umis > lower) & (umis <= upper))[0]
        if len(w) == 0:
            raise ValueError(
                f"No droplets found in soup_range {soup_range}. "
                "Check that your raw (unfiltered) matrix contains empty droplets."
            )
        soup_counts = np.array(sc.tod[:, w].sum(axis=1)).flatten().astype(float)

    elif method == 'statistical':
        soup_counts, n_accepted = _statistical_soup(sc, umis, lower, upper, alpha)
        if n_accepted == 0:
            warnings.warn(
                "Statistical method found no ambient barcodes; "
                "falling back to all barcodes in soup_range."
            )
            w = np.where((umis > lower) & (umis <= upper))[0]
            if len(w) == 0:
                raise ValueError(
                    f"No droplets found in soup_range {soup_range}."
                )
            soup_counts = np.array(sc.tod[:, w].sum(axis=1)).flatten().astype(float)

    elif method == 'emptydrops':
        soup_counts, n_accepted = _emptydrops_nb_soup(sc, umis, lower, upper, alpha)
        if n_accepted == 0:
            warnings.warn(
                "emptydrops method found no ambient barcodes; "
                "falling back to all barcodes in soup_range."
            )
            w = np.where((umis > lower) & (umis <= upper))[0]
            if len(w) == 0:
                raise ValueError(
                    f"No droplets found in soup_range {soup_range}."
                )
            soup_counts = np.array(sc.tod[:, w].sum(axis=1)).flatten().astype(float)

    else:
        raise ValueError(
            f"Unknown method {method!r}. Use 'fixed', 'statistical', or 'emptydrops'."
        )

    total = soup_counts.sum()
    if total == 0:
        raise ValueError("Soup profile is all zeros — no counts in selected droplets.")

    sc.soup_profile = pd.DataFrame(
        {'est': soup_counts / total, 'counts': soup_counts},
        index=sc.genes
    )

    if not keep_droplets:
        sc.tod = None

    return sc


def _statistical_soup(sc, umis, lower, upper, alpha):
    """
    Select empty droplets for soup profile using ambient similarity testing.

    Algorithm
    ---------
    1. Seed barcodes: UMI in (lower, seed_max] where seed_max = lower + 10% of range.
       These are taken as ground-truth ambient and define the initial profile.
    2. Test barcodes: UMI in (seed_max, upper).
       For each, run a vectorised two-sided Poisson test on the top ambient-signal
       genes. Accept those not significantly different from ambient (p > alpha).
    3. Return aggregated counts from seed + accepted test barcodes.

    Returns
    -------
    (soup_counts, n_accepted) where soup_counts is a (n_genes,) float array.
    """
    seed_max = lower + max(1.0, (upper - lower) * 0.1)
    seed_idx = np.where((umis > lower) & (umis <= seed_max))[0]

    if len(seed_idx) == 0:
        # Fallback: use lower half of range as seed
        mid = (lower + upper) / 2.0
        seed_idx = np.where((umis > lower) & (umis <= mid))[0]
        if len(seed_idx) == 0:
            return np.zeros(sc.tod.shape[0], dtype=float), 0

    # Initial ambient profile from seed barcodes
    seed_counts = np.array(sc.tod[:, seed_idx].sum(axis=1)).flatten().astype(float)
    seed_total = seed_counts.sum()
    if seed_total == 0:
        return seed_counts, 0
    ambient_profile = seed_counts / seed_total

    # Test barcodes in (seed_max, upper)
    test_idx = np.where((umis > seed_max) & (umis <= upper))[0]
    accepted_test_mask = np.ones(len(test_idx), dtype=bool)  # default: accept all

    if len(test_idx) > 0:
        tod_test = sc.tod[:, test_idx]
        tod_arr = tod_test.toarray().astype(float) if hasattr(tod_test, 'toarray') \
            else np.asarray(tod_test, dtype=float)  # (n_genes, n_test)

        n_total = tod_arr.sum(axis=0)  # (n_test,)

        # Top ambient-signal genes as Poisson test signal
        n_nonzero_ambient = int((ambient_profile > 0).sum())
        n_top = max(1, min(50, n_nonzero_ambient))
        top_idx = np.argsort(ambient_profile)[-n_top:]
        top_frac = float(ambient_profile[top_idx].sum())

        if top_frac > 0:
            obs_top = tod_arr[top_idx, :].sum(axis=0)       # (n_test,)
            exp_top = n_total * top_frac                     # (n_test,)

            # Two-sided Poisson test: P(X >= obs) and P(X <= obs)
            exp_safe = np.maximum(exp_top, 1e-10)
            p_upper = scipy.stats.poisson.sf(obs_top - 1, exp_safe)   # P(X >= obs)
            p_lower = scipy.stats.poisson.cdf(obs_top, exp_safe)       # P(X <= obs)
            p_two = np.minimum(2.0 * np.minimum(p_upper, p_lower), 1.0)

            # Accept if not significantly different from ambient, or too few UMIs
            accepted_test_mask = (p_two > alpha) | (n_total < 5)

    accepted_idx = np.concatenate([seed_idx, test_idx[accepted_test_mask]])
    soup_counts = np.array(sc.tod[:, accepted_idx].sum(axis=1)).flatten().astype(float)
    return soup_counts, len(accepted_idx)


def _emptydrops_nb_soup(sc, umis, lower, upper, alpha):
    """
    emptyDrops-style empty droplet selection via quasi-Poisson (NB) model.

    Algorithm
    ---------
    1. Seed: lowest-UMI barcodes in (lower, seed_max] define ambient profile.
    2. Overdispersion: estimate quasi-Poisson scale factor (phi) from seed
       barcodes via the Pearson estimator.  phi >= 1 (Poisson if = 1).
    3. Test: for each candidate barcode in (seed_max, upper), compute an
       overdispersion-corrected Pearson chi-squared vs. the ambient profile.
    4. BH FDR correction across all candidates.
    5. Accept barcodes where FDR-adjusted p > alpha (consistent with ambient).

    Returns
    -------
    (soup_counts, n_accepted)
    """
    from statsmodels.stats.multitest import multipletests

    seed_max = lower + max(1.0, (upper - lower) * 0.1)
    seed_idx = np.where((umis > lower) & (umis <= seed_max))[0]
    if len(seed_idx) == 0:
        mid = (lower + upper) / 2.0
        seed_idx = np.where((umis > lower) & (umis <= mid))[0]
        if len(seed_idx) == 0:
            return np.zeros(sc.tod.shape[0], dtype=float), 0

    seed_mat = sc.tod[:, seed_idx]
    seed_counts = np.array(seed_mat.sum(axis=1)).flatten().astype(float)
    seed_total = seed_counts.sum()
    if seed_total == 0:
        return seed_counts, 0
    ambient_profile = seed_counts / seed_total          # (n_genes,)

    # Quasi-Poisson overdispersion from seed barcodes
    seed_arr = seed_mat.toarray().astype(float)         # (n_genes, n_seed)
    seed_umis_f = umis[seed_idx].astype(float)
    mu_seed = np.outer(ambient_profile, seed_umis_f)    # (n_genes, n_seed)

    nonzero = ambient_profile > 0
    df_genes = max(1, int(nonzero.sum()) - 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        pearson_per_seed = np.where(
            mu_seed[nonzero, :] > 0,
            (seed_arr[nonzero, :] - mu_seed[nonzero, :]) ** 2 / mu_seed[nonzero, :],
            0.0
        ).sum(axis=0)                                   # (n_seed,)

    scale = float(np.nanmedian(pearson_per_seed / df_genes))
    if not np.isfinite(scale):
        warnings.warn(
            "All-NaN overdispersion estimates in emptydrops; "
            "falling back to Poisson model (scale=1).",
            UserWarning, stacklevel=3,
        )
        scale = 1.0
    scale = max(1.0, scale)                             # quasi-Poisson: scale >= 1

    test_idx = np.where((umis > seed_max) & (umis <= upper))[0]
    if len(test_idx) == 0:
        return seed_counts, len(seed_idx)

    tod_test = sc.tod[:, test_idx]
    tod_arr = tod_test.toarray().astype(float)          # (n_genes, n_test)
    test_umis_f = umis[test_idx].astype(float)

    mu_test = np.outer(ambient_profile, test_umis_f)    # (n_genes, n_test)

    with np.errstate(divide='ignore', invalid='ignore'):
        chi2_raw = np.where(
            mu_test[nonzero, :] > 0,
            (tod_arr[nonzero, :] - mu_test[nonzero, :]) ** 2 / mu_test[nonzero, :],
            0.0
        ).sum(axis=0)                                   # (n_test,)

    chi2_adj = chi2_raw / scale
    p_vals = scipy.stats.chi2.sf(chi2_adj, df=df_genes)
    nan_pval = ~np.isfinite(p_vals)
    if nan_pval.any():
        warnings.warn(
            f"{nan_pval.sum()} NaN p-values in emptydrops chi2 test; "
            "treating as non-significant (ambient).",
            UserWarning, stacklevel=3,
        )
        p_vals = np.where(nan_pval, 1.0, p_vals)

    if len(p_vals) > 0 and p_vals.min() < 1.0:
        _, p_adj, _, _ = multipletests(p_vals, method='fdr_bh')
    else:
        p_adj = p_vals.copy()

    accepted_mask = p_adj > alpha
    accepted_idx = np.concatenate([seed_idx, test_idx[accepted_mask]])
    soup_counts = np.array(sc.tod[:, accepted_idx].sum(axis=1)).flatten().astype(float)
    return soup_counts, len(accepted_idx)


def _chi2_ambient_pval(obs_counts, ambient_profile, n_total):
    """
    Chi-squared goodness-of-fit p-value for a single barcode vs. ambient profile.

    Tests whether obs_counts is consistent with Multinomial(n_total, ambient_profile).
    Handles sparsity by merging genes with expected count < 5 into an 'other' bin.

    Parameters
    ----------
    obs_counts : np.ndarray, shape (n_genes,)
    ambient_profile : np.ndarray, shape (n_genes,)  (sums to 1)
    n_total : float  total UMI count for this barcode

    Returns
    -------
    float  p-value in [0, 1]; returns 1.0 when there is insufficient data to test.
    """
    expected = ambient_profile * n_total
    nonzero = (expected > 0) | (obs_counts > 0)
    obs = obs_counts[nonzero].astype(float)
    exp = expected[nonzero].astype(float)

    large_mask = exp >= 5.0
    if large_mask.sum() < 2:
        return 1.0  # not enough bins → conservative accept

    obs_large = obs[large_mask]
    exp_large = exp[large_mask]
    obs_other = obs[~large_mask].sum()
    exp_other = exp[~large_mask].sum()

    if exp_other > 0:
        obs_final = np.append(obs_large, obs_other)
        exp_final = np.append(exp_large, exp_other)
    else:
        obs_final = obs_large
        exp_final = exp_large

    df = len(obs_final) - 1
    if df < 1:
        return 1.0

    chi2 = float(np.sum((obs_final - exp_final) ** 2 / exp_final))
    return float(scipy.stats.chi2.sf(chi2, df=df))
