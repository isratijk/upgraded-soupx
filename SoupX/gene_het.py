"""
Gene-level soup heterogeneity for DecontX.

Standard DecontX uses the raw soup profile (normalized empty-droplet counts)
as a fixed mixture component.  In reality, some genes are uniquely ambient
(e.g. haemoglobin in a non-erythroid experiment) while others appear equally
in soup and cells, making them uninformative for separating contamination from
native expression.

This module reweights the soup profile BEFORE the LDA EM to amplify truly
ambient genes and suppress ambiguous ones:

    enrichment_g = log1p(soup_share_g / cell_share_g)   [clipped]

The reweighted profile is then used as the fixed pi in run_decontx, improving
the model's ability to separate contamination from native expression in datasets
where ambient soup substantially overlaps with cellular expression.
"""

import copy
import warnings

import numpy as np
import scipy.sparse


def compute_gene_enrichment(sc, log_smooth=True, min_weight=0.5, max_weight=2.0):
    """
    Per-gene soup enrichment factor relative to mean corrected cellular expression.

    Parameters
    ----------
    sc : SoupChannel
        Must have soup_profile and toc.
    log_smooth : bool
        Apply log1p to the ratio to prevent extreme values (default True).
    min_weight, max_weight : float
        Clip bounds on the final enrichment factor.

    Returns
    -------
    np.ndarray shape (n_genes,): enrichment weight per gene
    """
    soup = sc.soup_profile['est'].values.astype(float)
    soup_norm = soup / (soup.sum() + 1e-10)

    cell_mean = np.asarray(sc.toc.mean(axis=1)).flatten().astype(float)
    cell_norm = cell_mean / (cell_mean.sum() + 1e-10)

    ratio = soup_norm / (cell_norm + 1e-10)
    if log_smooth:
        ratio = np.log1p(ratio)

    return np.clip(ratio, min_weight, max_weight)


def reweight_soup_profile(sc, log_smooth=True, min_weight=0.5, max_weight=2.0,
                           inplace=False):
    """
    Reweight sc.soup_profile by per-gene soup-vs-cell enrichment.

    Parameters
    ----------
    sc : SoupChannel
    log_smooth, min_weight, max_weight : passed to compute_gene_enrichment
    inplace : bool

    Returns
    -------
    SoupChannel with modified soup_profile (original unchanged unless inplace=True)
    """
    if sc.soup_profile is None:
        raise ValueError("soup_profile must be set before calling reweight_soup_profile.")

    if not inplace:
        sc = copy.deepcopy(sc)

    enrichment = compute_gene_enrichment(
        sc, log_smooth=log_smooth, min_weight=min_weight, max_weight=max_weight
    )

    orig = sc.soup_profile['est'].values.astype(float)
    new_soup = orig * enrichment
    new_soup = np.maximum(new_soup, 0)
    new_soup /= (new_soup.sum() + 1e-10)

    sc.soup_profile = sc.soup_profile.copy()
    sc.soup_profile['est'] = new_soup
    return sc


def run_decontx_genehet(sc, n_topics=None, n_iter=300, n_hvg=2000, soup_top_q=0.9,
                         pca_init=True, inner_iter=1, tol_theta=1e-4, tol_param=1e-5,
                         log_smooth=True, min_weight=0.5, max_weight=1.5,
                         prior_rho=None, verbose=False, exclude_mt=False):
    """
    DecontX with gene-level soup profile reweighting (gene-het variant).

    Amplifies truly ambient genes (high soup / low cell expression) and
    suppresses ambiguous genes before the LDA EM.  This improves separation
    of contamination from native expression, especially when soup ~= cells.

    Parameters
    ----------
    sc : SoupChannel
        Should have clusters and rho from auto_est_cont (warm-start prior).
    n_topics : int or None
        LDA topics.  None = max(2, n_unique_clusters).
    n_iter, n_hvg, soup_top_q, pca_init, inner_iter, tol_theta, tol_param,
    prior_rho, verbose, exclude_mt : forwarded to run_decontx
    log_smooth, min_weight, max_weight : forwarded to reweight_soup_profile

    Returns
    -------
    SoupChannel with per-cell rho from gene-het DecontX EM
    """
    from .decontx import run_decontx

    sc_weighted = reweight_soup_profile(
        sc, log_smooth=log_smooth, min_weight=min_weight,
        max_weight=max_weight, inplace=False
    )

    if n_topics is None:
        if 'clusters' in sc_weighted.meta_data.columns:
            n_topics = max(2, int(sc_weighted.meta_data['clusters'].nunique()))
        else:
            n_topics = 5

    return run_decontx(
        sc_weighted,
        n_topics=n_topics, n_iter=n_iter, n_hvg=n_hvg,
        soup_top_q=soup_top_q, pca_init=pca_init,
        inner_iter=inner_iter, tol_theta=tol_theta, tol_param=tol_param,
        prior_rho=prior_rho, verbose=verbose, exclude_mt=exclude_mt,
    )
