"""
DecontX: Bayesian Dirichlet-multinomial decontamination with LDA topics.

Two-component mixture model per cell:
  x_i ~ Multinomial(n_i, theta_i * pi + (1 - theta_i) * phi_i)

where theta_i is the per-cell contamination fraction and phi_i is the
native expression profile estimated as a mixture of K LDA topics shared
across all cells.  Sharing topics lets rare cell types borrow strength
from similar cells instead of relying only on their own sparse counts.

Reference: Yang S et al., Genome Biol 21, 289 (2020).
"""

import numpy as np
import scipy.sparse
import warnings


# ─── Soup profile quality guard ───────────────────────────────────────────────

def _check_soup_profile(sc):
    """
    Warn when the soup profile has properties that will bias rho estimates.

    Two checks:
    1. MT genes > 5% of soup.  MT RNA leaks from damaged cells into every
       droplet; it is ubiquitous in real cells too, so the model mistakes
       genuine mitochondrial expression for contamination.
    2. ≥5 of the top-10 soup genes are also in the top 10% of mean cell
       expression.  When soup ≈ cells, the model cannot distinguish ambient
       signal from native expression and drives rho toward zero.
    """
    soup_est = sc.soup_profile['est'].values
    genes    = sc.soup_profile.index

    mt_mask = genes.str.match(r'^[Mm][Tt]-')
    mt_frac = float(soup_est[mt_mask].sum())
    if mt_frac > 0.05:
        warnings.warn(
            f"MT genes account for {mt_frac*100:.1f}% of the soup profile. "
            "MT-dominated soup produces unreliable rho estimates — "
            "pass exclude_mt=True to zero MT genes from the soup before EM.",
            UserWarning, stacklevel=3,
        )

    top10_soup_idx  = set(np.argsort(soup_est)[::-1][:10])
    mean_expr       = np.asarray(sc.toc.mean(axis=1)).flatten()
    threshold_90pct = float(np.quantile(mean_expr, 0.90))
    top10pct_idx    = set(np.where(mean_expr >= threshold_90pct)[0])
    overlap_idx     = top10_soup_idx & top10pct_idx
    if len(overlap_idx) >= 5:
        overlap_names = ", ".join(str(genes[i]) for i in sorted(overlap_idx))
        warnings.warn(
            f"{len(overlap_idx)}/10 top soup genes are also in the top 10% of "
            f"mean cell expression ({overlap_names}). "
            "Soup ≈ cells — the model cannot separate contamination from native "
            "expression; near-zero rho estimates are likely.",
            UserWarning, stacklevel=3,
        )


def run_decontx(sc, n_topics=20, n_iter=500,
                tol_theta=1e-4, tol_param=1e-5, tol_ll=1e-6,
                prior_rho=None, n_hvg=None, soup_top_q=0.9, delta_rate=None,
                seed=42, pca_init=True, inner_iter=1, phi_chunk=200_000,
                exclude_mt=False, verbose=True, inplace=False):
    """
    Full DecontX: Bayesian Dirichlet-multinomial decontamination.

    Models each cell's count vector as a mixture of ambient (soup) and native
    expression.  Native expression is modelled via K LDA topics shared across
    all cells, so rare cell types benefit from the expression patterns of
    nearby cells in topic space rather than relying on their own sparse counts.

    Sets sc.meta_data['rho'] to per-cell contamination fractions and stores
    topic proportions in sc.meta_data['decontx_topic_0'] … ['decontx_topic_K-1'].
    After calling this function, run adjust_counts(sc) as normal.

    Parameters
    ----------
    sc : SoupChannel
        Must have soup_profile set (call estimate_soup / load_10x first).
    n_topics : int
        Number of LDA topics for native expression (default 20).
    n_iter : int
        Maximum EM iterations (default 500).
    tol_theta : float
        Convergence threshold for max |ΔΘ_i| across cells (default 1e-4).
        Max-based: one outlier cell can delay convergence; tune upward if needed.
    tol_param : float
        Convergence threshold for mean |ΔBeta| and mean |ΔPi_topics| (default
        1e-5).  Mean-based, appropriate for the shared topic parameters.
        Convergence is declared when BOTH tol_theta AND tol_param are satisfied
        (or when the rel-LL criterion fires).
    tol_ll : float
        Relative log-likelihood convergence threshold (default 1e-6). EM also
        stops when |LL_t - LL_{t-1}| / |LL_{t-1}| < tol_ll, preventing
        max-iter exhaustion when Θ oscillates near a true fixed point.
    prior_rho : float or None
        Initial contamination fraction. None = use mean of sc.meta_data['rho']
        if already set, else 0.05.
    n_hvg : int or None
        Number of highly variable genes for EM (default None = use all genes).
        The sparse EM never densifies the full n_cells × n_genes matrix, so
        memory scales with nnz (non-zeros in the count matrix), not n_genes.
        Set to an integer (e.g. 3000) only to reduce nnz-proportional compute
        on very large datasets; information loss applies.
    soup_top_q : float or None
        Genes above this quantile in the soup profile are force-included in
        the HVG set, ensuring ambient markers are always represented (default 0.9).
        Set to None to skip force-inclusion.
    delta_rate : float or None
        Symmetric Dirichlet concentration for topic-gene profiles.
        None = 1 / n_genes (weak, non-informative prior).
    seed : int
        Random seed for reproducibility (default 42).
    pca_init : bool
        If True, initialize topic profiles from the top principal components of
        the normalized count matrix using sparse SVD (default True).
        Converges 2-3x faster than random cell sampling.
    inner_iter : int
        Pi_topics multiplicative update steps per outer EM iteration (default 1).
        1 = standard EM (stable). Higher values accelerate topic-proportion
        convergence but can amplify per-iteration Θ oscillations on noisy data.
    phi_chunk : int
        Non-zeros per chunk when computing Phi at sparse positions (default
        200 000). Limits peak intermediate memory; lower = less RAM, more loops.
    exclude_mt : bool
        If True, zero MT genes (prefix MT- or mt-) out of the soup profile pi
        used by the EM and renormalize.  MT RNA leaks ubiquitously from damaged
        cells; leaving it in biases rho upward in every cell.  The MT counts
        in the cell matrix are kept intact — only the ambient attribution is
        suppressed (default False).
    verbose : bool
    inplace : bool

    Returns
    -------
    SoupChannel
        With sc.meta_data['rho'] as a per-cell array and topic proportions
        stored as additional metadata columns.
    """
    if not inplace:
        sc = sc.copy()

    if sc.soup_profile is None:
        raise ValueError("soup_profile must be set. Call estimate_soup() first.")

    # ── Soup profile quality checks ───────────────────────────────────────────
    _check_soup_profile(sc)

    if prior_rho is None:
        if 'rho' in sc.meta_data.columns:
            prior_rho = sc.meta_data['rho'].values.copy().astype(float)
        else:
            prior_rho = 0.05
    if np.ndim(prior_rho) == 0:
        prior_rho = float(np.clip(prior_rho, 1e-4, 1.0 - 1e-4))
    else:
        prior_rho = np.clip(np.asarray(prior_rho, dtype=float), 1e-4, 1.0 - 1e-4)

    n_genes_full = sc.toc.shape[0]

    # ── Gene selection ────────────────────────────────────────────────────────
    gene_idx = _select_hvg(sc.toc, sc.soup_profile, n_hvg, soup_top_q)
    n_genes_used = len(gene_idx)

    if verbose and n_genes_used < n_genes_full:
        print(f"DecontX: using {n_genes_used} genes "
              f"({n_hvg} HVGs + forced soup top-{int(soup_top_q*100)}% genes)")

    toc_sub = sc.toc.tocsc()[gene_idx, :]
    pi_sub  = sc.soup_profile['est'].values[gene_idx].astype(float)

    # ── MT exclusion from ambient profile ────────────────────────────────────
    if exclude_mt:
        sub_genes = sc.soup_profile.index[gene_idx]
        mt_mask   = sub_genes.str.match(r'^[Mm][Tt]-')
        n_mt      = int(mt_mask.sum())
        pi_sub[mt_mask] = 0.0
        if verbose and n_mt > 0:
            print(f"DecontX: zeroed {n_mt} MT genes from soup pi (renormalizing)")

    pi_sub_sum = pi_sub.sum()
    if pi_sub_sum <= 0:
        raise ValueError(
            "Soup profile sums to zero for selected genes"
            + (" after MT exclusion — all selected genes are MT genes; "
               "increase n_hvg or disable exclude_mt." if exclude_mt
               else "; check that soup_profile is properly normalised.")
        )
    pi_sub /= pi_sub_sum

    if delta_rate is None:
        delta_rate = 1.0 / n_genes_used

    # Memory estimate: sparse EM scales with nnz, not n_cells × n_genes.
    nnz = toc_sub.nnz
    mem_mb = (nnz * 5 + n_genes_used * n_topics * 2 + sc.toc.shape[1] * n_topics) * 4 / 1e6
    if mem_mb > 4000:
        warnings.warn(
            f"run_decontx needs ~{mem_mb:.0f} MB "
            f"(nnz={nnz:,}, {n_genes_used} genes, {n_topics} topics). "
            "Reduce n_hvg or n_topics to lower memory."
        )

    # ── PCA initialisation ────────────────────────────────────────────────────
    init_beta = None
    if pca_init:
        if verbose:
            print("DecontX: computing PCA initialisation for topics …")
        init_beta = _pca_init_beta(toc_sub, n_topics, delta_rate, seed)

    # ── EM ────────────────────────────────────────────────────────────────────
    theta, Pi_topics, ll_history = _decontx_lda_em(
        toc_sub, pi_sub,
        n_topics=n_topics, n_iter=n_iter,
        tol_theta=tol_theta, tol_param=tol_param, tol_ll=tol_ll,
        prior_rho=prior_rho, delta_rate=delta_rate,
        seed=seed, verbose=verbose, init_beta=init_beta,
        inner_iter=inner_iter, phi_chunk=phi_chunk,
    )

    sc.meta_data['rho'] = theta
    for k in range(n_topics):
        sc.meta_data[f'decontx_topic_{k}'] = Pi_topics[:, k]

    total_counts = float(toc_sub.sum())
    final_ll     = float(ll_history[-1]) if len(ll_history) else float('nan')
    final_ppl    = float(np.exp(-final_ll / total_counts)) if total_counts > 0 else float('nan')

    sc.fit = {
        'method': 'decontx',
        'n_topics': n_topics,
        'n_hvg': n_genes_used,
        'prior_rho': prior_rho,
        'n_iter_max': n_iter,
        'tol_theta': tol_theta,
        'tol_param': tol_param,
        'tol_ll': tol_ll,
        'exclude_mt': exclude_mt,
        'pca_init': pca_init,
        'inner_iter': inner_iter,
        'll_history': ll_history,
        'final_ll': final_ll,
        'final_perplexity': final_ppl,
        'n_iter_run': len(ll_history),
        'perplexity_random': float(n_genes_used),
        'perplexity_ratio': float(n_genes_used) / final_ppl if final_ppl > 0 else float('nan'),
    }

    if verbose:
        ppl_random = float(n_genes_used)
        ratio      = ppl_random / final_ppl if final_ppl > 0 else float('nan')
        print(
            f"DecontX: mean rho={float(theta.mean()):.4f}, "
            f"std={float(theta.std()):.4f}, "
            f"range=[{float(theta.min()):.4f}, {float(theta.max()):.4f}]  |  "
            f"LL={final_ll:.4e}  "
            f"perplexity={final_ppl:.1f} "
            f"(random baseline: {ppl_random:.0f}, ratio: {ratio:.1f}×)"
        )

    return sc


# ─── Topic count selection ────────────────────────────────────────────────────

def select_n_topics(sc, n_topics_range=(5, 10, 20, 50), n_iter=150,
                    tol_theta=1e-3, tol_param=1e-4, tol_ll=1e-5,
                    n_hvg=None, soup_top_q=0.9, exclude_mt=False,
                    seed=42, verbose=True):
    """
    Fit DecontX at each value in n_topics_range and report perplexity.

    Use the elbow in the perplexity curve to pick n_topics: choose the
    smallest K where adding more topics stops meaningfully reducing perplexity.
    The random baseline (= n_genes) and the perplexity ratio are printed so the
    absolute scale is interpretable.

    Parameters
    ----------
    sc : SoupChannel
    n_topics_range : sequence of int
        Topic counts to evaluate (default (5, 10, 20, 50)).
    n_iter : int
        Max EM iterations per fit (default 150 — looser than run_decontx to
        keep wall time manageable; increase for noisy data).
    tol_theta, tol_param, tol_ll : float
        Convergence thresholds (defaults are looser than run_decontx).
    n_hvg, soup_top_q, exclude_mt, seed : same as run_decontx.
    verbose : bool
        Print a per-K summary table (default True).

    Returns
    -------
    pandas.DataFrame
        Columns: n_topics, perplexity, perplexity_ratio, mean_rho, n_iter_run.
        Sorted by n_topics.
    """
    import pandas as pd

    rows = []
    for k in sorted(set(n_topics_range)):
        sc_k = run_decontx(
            sc, n_topics=k, n_iter=n_iter,
            tol_theta=tol_theta, tol_param=tol_param, tol_ll=tol_ll,
            n_hvg=n_hvg, soup_top_q=soup_top_q, exclude_mt=exclude_mt,
            seed=seed, verbose=False, inplace=False,
        )
        fit = sc_k.fit
        rows.append({
            'n_topics':        k,
            'perplexity':      fit['final_perplexity'],
            'perplexity_ratio': fit['perplexity_ratio'],
            'mean_rho':        float(sc_k.meta_data['rho'].mean()),
            'n_iter_run':      fit['n_iter_run'],
        })

    result = pd.DataFrame(rows).sort_values('n_topics').reset_index(drop=True)

    if verbose:
        ppl_random = result['perplexity'].iloc[0] * result['perplexity_ratio'].iloc[0]
        print(f"\nDecontX topic selection  (random baseline perplexity: {ppl_random:.0f})")
        print(f"  {'n_topics':>8}  {'perplexity':>12}  {'ratio (×)':>10}  "
              f"{'mean_rho':>9}  {'iters':>6}")
        print("  " + "-" * 55)
        for _, row in result.iterrows():
            print(f"  {int(row.n_topics):>8}  {row.perplexity:>12.1f}  "
                  f"{row.perplexity_ratio:>10.1f}  "
                  f"{row.mean_rho:>9.4f}  {int(row.n_iter_run):>6}")
        print()

    return result


# ─── Gene selection ────────────────────────────────────────────────────────────

def _select_hvg(toc, soup_profile, n_hvg, soup_top_q):
    """
    Return gene indices for EM: top-n_hvg by normalised variance, union with
    top-soup_top_q soup genes (so ambient markers are never filtered out).
    """
    n_genes = toc.shape[0]

    if n_hvg is None or n_hvg >= n_genes:
        return np.arange(n_genes)

    toc_csc = toc.tocsc().astype(float)
    nUMIs   = np.array(toc_csc.sum(axis=0)).flatten()
    toc_norm = toc_csc.multiply(1e4 / np.maximum(nUMIs, 1))

    gene_mean = np.array(toc_norm.mean(axis=1)).flatten()
    gene_sq   = np.array(toc_norm.power(2).mean(axis=1)).flatten()
    gene_var  = gene_sq - gene_mean**2

    hvg_set = set(int(i) for i in np.argsort(gene_var)[::-1][:n_hvg])

    if soup_top_q is not None and soup_profile is not None:
        soup_est = soup_profile['est'].values
        threshold = float(np.quantile(soup_est, soup_top_q))
        soup_set  = set(int(i) for i in np.where(soup_est >= threshold)[0])
        hvg_set   = hvg_set | soup_set

    return np.array(sorted(hvg_set), dtype=int)


# ─── PCA initialisation ────────────────────────────────────────────────────────

def _pca_init_beta(toc, n_topics, delta, seed):
    """
    Initialise topic-gene profiles Beta from sparse SVD of the normalised
    count matrix.  Each principal component captures a coherent expression
    programme, giving topics a biologically grounded starting point and
    cutting convergence time by 2-3x vs random cell sampling.

    Uses scipy.sparse.linalg.svds (ARPACK) — works directly on the sparse
    matrix so no densification is needed.
    """
    from scipy.sparse.linalg import svds

    n_genes, n_cells = toc.shape
    k = min(n_topics, min(n_genes, n_cells) - 1)

    nUMIs    = np.array(toc.sum(axis=0)).flatten()
    toc_norm = toc.multiply(1e4 / np.maximum(nUMIs, 1))  # (n_genes, n_cells)
    X_sp     = toc_norm.T.tocsr()                          # (n_cells, n_genes)

    rng = np.random.default_rng(seed)
    v0  = rng.standard_normal(min(X_sp.shape))

    try:
        _, _, Vt = svds(X_sp, k=k, v0=v0)   # Vt: (k, n_genes)
    except Exception as exc:
        warnings.warn(
            f"PCA initialisation failed ({type(exc).__name__}: {exc}); "
            "falling back to random-cell sampling for topic init.",
            UserWarning, stacklevel=2,
        )
        return None  # fall back to random-cell init in _decontx_lda_em

    # svds returns in ascending singular-value order — reverse
    Vt = Vt[::-1, :]

    # Shift to positive (loadings can be negative) and add Dirichlet floor
    Beta = np.abs(Vt[:n_topics, :]) + delta
    Beta /= Beta.sum(axis=1, keepdims=True)
    return Beta


# ─── Core EM ─────────────────────────────────────────────────────────────────

def _decontx_lda_em(toc, pi, n_topics=20, n_iter=500,
                    tol_theta=1e-4, tol_param=1e-5, tol_ll=1e-6,
                    prior_rho=0.05, delta_rate=None, seed=42, verbose=False,
                    init_beta=None, inner_iter=5, phi_chunk=200_000):
    """
    Full DecontX Dirichlet-multinomial + LDA EM (Yang et al. 2020).

    WHEN TO USE vs. _decontx_em (estimation.py)
    --------------------------------------------
    This function (LDA topics):
      - Phi_i = Pi_topics[i] @ Beta — K shared topics let rare cell types
        borrow strength from similar cells.  Appropriate when contamination
        estimation IS the primary pipeline step (no prior cluster rho).
      - Expensive: O(nnz × n_topics) per iteration; defaults to 500 iterations.
      - float32 for memory efficiency with 100K-cell datasets.
      - Called by: run_decontx only.

    _decontx_em (estimation.py):
      - phi_i estimated independently per cell; no topic sharing.
      - Cheaper (O(nnz)), float64, 100 iterations.
      - Intended as a refinement step after auto_est_cont has set a prior rho.
      - Called by: estimate_decontx_rho, estimate_cell_rho.

    The two implementations intentionally share no code because their priors,
    float precision, E/M-step math, and convergence criteria all differ once
    LDA topic variables are introduced.

    Model
    -----
    For cell i with count vector x_i (n_genes):
      x_i ~ Multinomial(n_i, theta_i * pi + (1 - theta_i) * Phi_i)
      Phi_i = Pi_topics[i] @ Beta          (LDA native profile)
      Pi_topics[i] ~ Dirichlet(alpha_phi)  (topic proportions per cell)
      Beta[k]      ~ Dirichlet(delta)      (topic-gene profile)
      theta_i      ~ Beta(alpha_theta, beta_theta)

    Memory
    ------
    Never materializes the full (n_cells × n_genes) Phi or W matrices.
    Phi is computed only at the non-zero positions of X (chunked to control
    peak memory), and W is kept as a sparse matrix with X's sparsity pattern.
    Peak memory scales with nnz(X) × n_topics, not n_cells × n_genes.

    Convergence
    -----------
    Declared when BOTH of these hold:
      max(|ΔTheta|) < tol_theta   — worst-cell Θ change (max-based)
      mean(|ΔBeta|) < tol_param   — shared topic-gene profiles (mean-based)
      mean(|ΔPi|)   < tol_param   — per-cell topic proportions (mean-based)
    OR when the relative LL change falls below tol_ll (guards against Θ
    oscillation that never crosses tol_theta).
    Using separate thresholds avoids the scale mismatch of the old joint-max
    criterion where Θ (max) always dominated Beta/Pi (mean).
    Pi_topics is updated inner_iter times per outer step with the E-step W
    held fixed (cheap: only n_cells × n_topics arithmetic).

    Parameters
    ----------
    toc       : scipy.sparse (n_genes, n_cells)
    pi        : np.ndarray (n_genes,), sums to 1
    inner_iter: int
        Pi_topics update steps per outer EM iteration (default 1).
    phi_chunk : int
        Non-zeros per chunk for Phi computation (default 200 000).
    init_beta : np.ndarray (n_topics, n_genes) or None

    Returns
    -------
    theta     : np.ndarray (n_cells,)
    Pi_topics : np.ndarray (n_cells, n_topics)
    """
    # ── Data setup (keep X sparse) ────────────────────────────────────────────
    X_csr  = toc.T.tocsr().astype(np.float32)          # (n_cells, n_genes)
    n_cells, n_genes = X_csr.shape
    n_umis = np.asarray(X_csr.sum(axis=1)).flatten().astype(np.float32)

    X_coo  = X_csr.tocoo()
    rows   = X_coo.row.astype(np.int32)
    cols   = X_coo.col.astype(np.int32)
    x_vals = X_coo.data.astype(np.float32)
    pi_nz  = pi.astype(np.float32)[cols]               # soup profile at each nnz

    if delta_rate is None:
        delta_rate = 1.0 / n_genes

    # Clip prior_rho away from boundaries: at 0 or 1 the Beta prior degenerates
    # (alpha_theta or beta_theta → 0), causing numeric instability in the Theta update.
    # Accepts scalar or (n_cells,) array — array path gives per-cell Beta priors.
    if np.ndim(prior_rho) == 0:
        _pr = np.full(n_cells, float(prior_rho), dtype=np.float32)
    else:
        _pr = np.clip(np.asarray(prior_rho, dtype=np.float32),
                      np.float32(1e-4), np.float32(1.0 - 1e-4))
    alpha_theta = (_pr * 10.0)
    beta_theta  = ((1.0 - _pr) * 10.0)
    alpha_phi   = np.float32(1.0 / n_topics)
    delta_f     = np.float32(delta_rate)
    eps         = np.float32(1e-10)
    rng = np.random.default_rng(seed)

    # ── Initialise Beta ───────────────────────────────────────────────────────
    if init_beta is not None and init_beta.shape == (n_topics, n_genes):
        Beta = init_beta.astype(np.float32)
    else:
        n_sample = min(n_topics, n_cells)
        idx = rng.choice(n_cells, size=n_sample, replace=False)
        Beta_init = X_csr[idx].toarray().astype(np.float32) + float(delta_rate)
        Beta_init /= Beta_init.sum(axis=1, keepdims=True) + eps
        if n_sample < n_topics:
            extra = rng.dirichlet(
                np.ones(n_genes) * float(delta_rate * n_genes),
                size=n_topics - n_sample
            ).astype(np.float32)
            Beta_init = np.vstack([Beta_init, extra])
        Beta = Beta_init[:n_topics]

    Pi_topics = np.full((n_cells, n_topics), 1.0 / n_topics, dtype=np.float32)
    Theta     = _pr.copy()

    d_theta = d_beta = d_pi = np.inf
    ll = prev_ll = -np.inf
    ll_history = []

    total_counts = float(x_vals.sum())

    for it in range(n_iter):
        Theta_prev     = Theta.copy()
        Beta_prev      = Beta.copy()
        Pi_topics_prev = Pi_topics.copy()

        # ── E-step: Phi only at non-zero positions, chunked ───────────────────
        # Avoids materializing the full (n_cells × n_genes) Phi matrix.
        # Each chunk computes (chunk_size, n_topics) · (n_topics, chunk_size).T
        Phi_nz = np.empty(len(rows), dtype=np.float32)
        for s in range(0, len(rows), phi_chunk):
            e = min(s + phi_chunk, len(rows))
            Phi_nz[s:e] = (Pi_topics[rows[s:e]] * Beta[:, cols[s:e]].T).sum(axis=1)

        # Floor Phi_nz to float32 machine-eps scale: prevents W_nz explosion when
        # Theta≈0 and Phi_nz underflows to a denormal (W_nz = N_nz / float32_tiny → ∞).
        Phi_nz   = np.maximum(Phi_nz, np.float32(1e-7))
        theta_nz = Theta[rows]
        amb_nz   = theta_nz * pi_nz
        nat_nz   = (1.0 - theta_nz) * Phi_nz
        mix_nz   = amb_nz + nat_nz + eps
        r_nz     = np.clip(amb_nz / mix_nz,
                           np.float32(0.0), np.float32(1.0))  # ambient responsibility at nnz
        N_nz     = x_vals * (1.0 - r_nz)
        W_nz     = N_nz / (Phi_nz + eps)

        # ── Log-likelihood: observed data LL at non-zero positions ────────────
        # L = Σ_j x_j · log(θ_{row_j}·π_{col_j} + (1−θ_{row_j})·Φ_nz_j)
        #   = Σ_j x_j · log(mix_nz_j)
        # Normalised as per-count log-likelihood (perplexity = exp(-ll_per_count)).
        # Free to compute: mix_nz already available from E-step.
        prev_ll = ll
        ll = float(np.dot(x_vals.astype(float), np.log(mix_nz.astype(float))))
        ll_per_count = ll / total_counts if total_counts > 0 else 0.0
        perplexity   = float(np.exp(-ll_per_count))
        ll_history.append(ll)

        # Sparse W has exactly X's sparsity pattern — no dense expansion needed
        W_sp = scipy.sparse.csr_matrix(
            (W_nz, (rows, cols)), shape=(n_cells, n_genes), dtype=np.float32
        )

        # ── Pi_topics M-step: inner_iter updates with fixed WBeta ────────────
        # WBeta = W_sp @ Beta.T is (n_cells, n_topics); cheap to reuse.
        # Multiple steps drive Pi_topics toward its fixed point before Beta moves.
        WBeta = W_sp @ Beta.T                         # (n_cells, n_topics)
        for _ in range(inner_iter):
            new_Pi    = alpha_phi + Pi_topics * WBeta
            Pi_topics = new_Pi / (new_Pi.sum(axis=1, keepdims=True) + eps)

        # ── Beta M-step ───────────────────────────────────────────────────────
        # W_sp.T (n_genes, n_cells) @ Pi_topics (n_cells, n_topics) → (n_genes, n_topics)
        PiW      = np.asarray(W_sp.T @ Pi_topics).T.astype(np.float32)  # (n_topics, n_genes)
        new_Beta = delta_f + Beta * PiW
        Beta     = new_Beta / (new_Beta.sum(axis=1, keepdims=True) + eps)

        # ── Theta M-step ──────────────────────────────────────────────────────
        A_total = np.bincount(
            rows, weights=(x_vals * r_nz).astype(float), minlength=n_cells
        ).astype(np.float32)
        Theta = (A_total + alpha_theta) / (n_umis + alpha_theta + beta_theta)
        Theta = np.clip(Theta, 0.0, 1.0)

        # ── Convergence: separate thresholds for Θ (max) vs Beta/Pi (mean) ────
        d_theta = float(np.max(np.abs(Theta - Theta_prev)))
        d_beta  = float(np.mean(np.abs(Beta - Beta_prev)))
        d_pi    = float(np.mean(np.abs(Pi_topics - Pi_topics_prev)))

        if verbose and (it + 1) % 25 == 0:
            print(f"  iter {it+1:3d}: LL={ll:.4e}  perplexity={perplexity:.4f}  "
                  f"Δθ={d_theta:.2e}/{tol_theta:.0e}  "
                  f"ΔBeta={d_beta:.2e}/{tol_param:.0e}  "
                  f"ΔPi={d_pi:.2e}/{tol_param:.0e}  "
                  f"mean_θ={float(Theta.mean()):.4f}")

        rel_ll      = (abs(ll - prev_ll) / (abs(prev_ll) + 1e-10)
                       if np.isfinite(prev_ll) else np.inf)
        param_conv  = (d_theta < tol_theta and d_beta < tol_param and d_pi < tol_param)
        ll_conv     = it > 0 and rel_ll < tol_ll

        if param_conv or ll_conv:
            if verbose:
                reason = "Δ-param" if param_conv else "rel-LL"
                print(f"DecontX LDA EM converged at iter {it+1} ({reason}): "
                      f"LL={ll:.4e}  perplexity={perplexity:.4f}  "
                      f"Δθ={d_theta:.2e} ΔBeta={d_beta:.2e} ΔPi={d_pi:.2e}")
            break
    else:
        if verbose:
            print(f"DecontX LDA EM reached max_iter={n_iter} "
                  f"(LL={ll:.4e}  perplexity={perplexity:.4f}  "
                  f"Δθ={d_theta:.2e} ΔBeta={d_beta:.2e} ΔPi={d_pi:.2e})")

    return Theta.astype(float), Pi_topics.astype(float), np.array(ll_history)
