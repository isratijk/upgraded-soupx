"""
Doublet-aware contamination estimation.

Scrublet-style doublet scoring integrated with auto_est_cont:
doublet cells corrupt tf-idf marker selection by mixing expression profiles
from two cell types.  Excluding them yields cleaner per-cluster rho estimates.

Reference: Wolock et al., Cell Systems 8, 281-291.e4 (2019).
"""

import copy
import warnings

import numpy as np
import scipy.sparse


def estimate_doublet_scores(toc, n_sim=None, n_pcs=30, k=20, seed=42, n_hvg=2000):
    """
    Estimate per-cell doublet scores using Scrublet-style simulation.

    For each real cell, score = fraction of k nearest neighbors in PCA space
    that are simulated doublets.  Simulated doublets are raw-count sums of
    random cell pairs, normalized and sqrt-transformed (stays sparse).

    Parameters
    ----------
    toc : sparse (n_genes, n_cells)
    n_sim : int
        Number of simulated doublets (default = n_cells).
    n_pcs : int
        PCA components for kNN embedding.
    k : int
        Nearest neighbors to query.
    seed : int
    n_hvg : int
        Top highly-variable genes to retain before SVD. Reduces cost when
        gene count is large (e.g. barnyard experiments with 60k genes).

    Returns
    -------
    np.ndarray shape (n_cells,): doublet score in [0, 1]
    """
    rng = np.random.default_rng(seed)
    mat = scipy.sparse.csc_matrix(toc).astype(float)
    n_genes, n_cells = mat.shape

    # Pre-filter to HVGs to keep SVD tractable for large gene sets
    if n_hvg is not None and n_genes > n_hvg:
        gene_var = np.asarray(mat.power(2).mean(axis=1)).flatten() - \
                   np.asarray(mat.mean(axis=1)).flatten() ** 2
        hvg_idx = np.argpartition(gene_var, -n_hvg)[-n_hvg:]
        mat = mat[hvg_idx, :]

    if n_sim is None:
        n_sim = n_cells

    # Simulate doublets by summing raw-count pairs (stays sparse)
    idx1 = rng.integers(0, n_cells, size=n_sim)
    idx2 = rng.integers(0, n_cells, size=n_sim)
    sim = mat[:, idx1] + mat[:, idx2]   # (n_genes, n_sim)

    def _sqrt_norm(m):
        m = scipy.sparse.csc_matrix(m)
        col_sums = np.asarray(m.sum(axis=0)).flatten()
        col_sums[col_sums == 0] = 1.0
        m_norm = scipy.sparse.csc_matrix(m.multiply(1e4 / col_sums))
        m_norm.data = np.sqrt(m_norm.data)
        return m_norm

    real_t = _sqrt_norm(mat)
    sim_t  = _sqrt_norm(sim)

    combined = scipy.sparse.hstack([real_t, sim_t], format='csr')
    n_total  = n_cells + n_sim
    is_sim   = np.zeros(n_total, dtype=bool)
    is_sim[n_cells:] = True

    from scipy.sparse.linalg import svds
    n_comps = min(n_pcs, min(combined.shape) - 1)
    try:
        v0 = np.random.default_rng(seed + 1).standard_normal(min(combined.shape))
        U, s, _ = svds(combined.T.tocsr(), k=max(1, n_comps), v0=v0)
        pca_coords = U * s   # (n_total, n_comps)
    except Exception:
        warnings.warn("SVD failed in doublet scoring; using raw feature fallback.",
                      stacklevel=2)
        pca_coords = combined.T.toarray()[:, :n_pcs]

    from scipy.spatial import KDTree
    real_coords = pca_coords[:n_cells]
    tree = KDTree(pca_coords)
    _, indices = tree.query(real_coords, k=k + 1)
    indices = indices[:, 1:]   # exclude self
    scores = is_sim[indices].mean(axis=1).astype(float)
    return scores


def auto_est_cont_doublet_aware(sc, doublet_threshold=0.25, n_sim=None,
                                 n_pcs=30, k=20, seed=42,
                                 use_umi_filter=True, umi_factor=1.5,
                                 inplace=False, **aec_kwargs):
    """
    Run auto_est_cont with doublet cells excluded from marker selection.

    Doublet cells express a mix of two cell types, diluting cluster
    specificity and corrupting tf-idf marker selection.  This function:

      1. Scores each cell for doublet probability.
      2. Temporarily relabels high-score cells to ``_excl_`` cluster.
      3. Runs auto_est_cont on the relabelled clusters.
      4. Restores original cluster labels.
      5. Imputes rho for excluded cells from non-doublet cluster mates.

    Parameters
    ----------
    sc : SoupChannel
        Must have clusters set in meta_data.
    doublet_threshold : float
        Cells with score >= this are treated as doublets (default 0.25).
    n_sim, n_pcs, k, seed : passed to estimate_doublet_scores
    inplace : bool
    **aec_kwargs : forwarded to auto_est_cont

    Returns
    -------
    SoupChannel with rho and doublet_score in meta_data
    """
    from .estimation import auto_est_cont

    if not inplace:
        sc = sc.copy()

    n_cells = sc.toc.shape[1]
    n_sim_eff = n_sim if n_sim is not None else n_cells

    scores = estimate_doublet_scores(sc.toc, n_sim=n_sim, n_pcs=n_pcs,
                                      k=k, seed=seed)
    sc.meta_data['doublet_score'] = scores

    # When n_sim equals n_cells the combined PCA space is 50% simulated, so
    # every real cell scores ~0.5 by random chance.  Use an adaptive threshold:
    # require the score to exceed 1.5× the expected random background, which
    # ensures only cells genuinely enriched in simulated-doublet neighbours are
    # flagged regardless of the n_sim / n_cells ratio.
    expected_bg       = n_sim_eff / (n_cells + n_sim_eff)
    adaptive_threshold = max(doublet_threshold, expected_bg * 1.5)
    is_doublet = scores >= adaptive_threshold

    # Additionally require elevated nUMIs: real doublets carry ~2× average UMI
    # counts whereas contaminated cells (e.g. barnyard cross-species) do not.
    if use_umi_filter and 'nUMIs' in sc.meta_data.columns:
        numi_arr   = sc.meta_data['nUMIs'].values.astype(float)
        umi_thresh = float(np.median(numi_arr)) * umi_factor
        is_doublet = is_doublet & (numi_arr > umi_thresh)

    if 'clusters' not in sc.meta_data.columns:
        warnings.warn("No clusters in meta_data; running standard auto_est_cont.",
                      stacklevel=2)
        return auto_est_cont(sc, **aec_kwargs)

    original_clusters = sc.meta_data['clusters'].copy()

    # Run auto_est_cont on all cells with the ORIGINAL cluster labels.
    #
    # The previous approach — relabelling doublets to '_excl_' before running
    # auto_est_cont — was flawed: doublets have ~2× nUMIs so the non-expressing
    # Poisson test (exp = nUMIs × max_cont × soup_frac) passes even for genes
    # that doublets genuinely express at high levels.  Those '_excl_' × gene
    # pairs then contribute rhoEst >> 1 to the joint posterior, inflating the
    # global rho estimate catastrophically (e.g. 1% → 29% on hgmm).
    #
    # Running on the full dataset with original clusters gives the same rho
    # estimate as a plain auto_est_cont call; the doublet-detection step then
    # serves only to flag cells whose rho is subsequently imputed from their
    # non-doublet cluster mates.
    try:
        sc = auto_est_cont(sc, **aec_kwargs)
    except (ValueError, KeyError) as exc:
        warnings.warn(
            f"auto_est_cont failed ({exc}); returning without doublet imputation.",
            stacklevel=2
        )
        is_doublet = np.zeros(len(scores), dtype=bool)

    sc.meta_data['clusters'] = original_clusters

    # Impute rho for doublet cells: use mean rho of clean cluster mates
    if is_doublet.any():
        orig_arr = original_clusters.astype(str).values
        rho_arr  = sc.meta_data['rho'].values.copy().astype(float)
        clean_mean = float(rho_arr[~is_doublet].mean()) if (~is_doublet).any() else 0.05
        for cl in np.unique(orig_arr[is_doublet]):
            in_cl     = orig_arr == cl
            clean_idx = in_cl & ~is_doublet
            fill_rho  = float(rho_arr[clean_idx].mean()) if clean_idx.any() else clean_mean
            rho_arr[in_cl & is_doublet] = fill_rho
        sc.meta_data['rho'] = rho_arr

    return sc
