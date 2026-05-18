import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats

from .utils import alloc, expand_clusters, init_prog_bar


def adjust_counts(sc, clusters=None, method='subtraction', round_to_int=False,
                   verbose=1, tol=1e-3):
    """
    Remove ambient RNA contamination from the count matrix.

    Parameters
    ----------
    sc : SoupChannel
        Must have rho set in meta_data (call set_contamination_fraction or
        auto_est_cont first).
    clusters : None, False, or array-like
        None = auto-load from sc.meta_data['clusters'].
        False = no clustering; adjust at single-cell level.
        array = explicit cluster labels (length n_cells).
    method : str
        'subtraction' (default) or 'multinomial'.
    round_to_int : bool
        Stochastically round result to integers.
    verbose : int
        0 = silent, 1 = basic, 2 = chatty, 3 = debug.
    tol : float
        Convergence tolerance (subtraction method internal).

    Returns
    -------
    scipy.sparse.csc_matrix
        Corrected (genes x cells) count matrix, same shape as sc.toc.
    """
    if 'rho' not in sc.meta_data.columns:
        raise ValueError("Contamination fraction (rho) must be set before adjusting counts.")

    # Normalize rho to a 1-D per-cell array. set_contamination_fraction stores a
    # scalar Python float via pandas column assignment; numpy may surface this as a
    # 0-d or length-1 array, causing IndexError when indexing by cell position.
    _rho_raw = np.asarray(sc.meta_data['rho'], dtype=float).flatten()
    if len(_rho_raw) != sc.toc.shape[1]:
        sc.meta_data['rho'] = np.full(sc.toc.shape[1], float(_rho_raw.flat[0]))

    if method not in ('subtraction', 'multinomial', 'soupOnly'):
        raise ValueError(f"Unknown method: {method!r}. Use 'subtraction', 'multinomial', or 'soupOnly'.")

    # ── Resolve cluster specification ─────────────────────────────────────────
    if clusters is None:
        if 'clusters' in sc.meta_data.columns:
            clusters = sc.meta_data['clusters'].astype(str).values
        else:
            if verbose > 0:
                print("Clustering data not found. Adjusting counts at cell level. "
                      "Results will improve with cluster information.")
            clusters = False

    if clusters is not False:
        clusters = np.asarray(clusters, dtype=str)
        if len(clusters) != len(sc.cells):
            raise ValueError("clusters must have length equal to number of cells.")

    # ── Cluster-aware path ────────────────────────────────────────────────────
    # soupOnly tests each cell independently; cluster aggregation would hide the
    # per-cell Poisson signal, so it always runs at single-cell level.
    if clusters is not False and method != 'soupOnly':
        unique_clusters = sorted(np.unique(clusters))
        n_genes = len(sc.genes)
        n_clusters = len(unique_clusters)
        toc_csc = sc.toc.tocsc()

        # Aggregate counts and rho to cluster level
        clust_counts = np.zeros((n_genes, n_clusters))
        clust_nUMIs = np.zeros(n_clusters)
        clust_rho = np.zeros(n_clusters)

        for ci, cl in enumerate(unique_clusters):
            mask = clusters == cl
            clust_counts[:, ci] = np.array(toc_csc[:, mask].sum(axis=1)).flatten()
            nUMIs_cl = sc.meta_data['nUMIs'].values[mask]
            rho_cl = sc.meta_data['rho'].values[mask]
            clust_nUMIs[ci] = nUMIs_cl.sum()
            clust_rho[ci] = (rho_cl * nUMIs_cl).sum() / clust_nUMIs[ci] if clust_nUMIs[ci] > 0 else 0.0

        # Build a minimal SoupChannel-like object for cluster-level adjustment
        clust_sc = _ClusterProxy(
            toc=scipy.sparse.csc_matrix(clust_counts),
            soup_profile=sc.soup_profile,
            meta_data=pd.DataFrame(
                {'nUMIs': clust_nUMIs, 'rho': clust_rho},
                index=pd.Index(unique_clusters)
            ),
            genes=sc.genes,
            cells=pd.Index(unique_clusters),
        )

        # Adjust at cluster level (no further clustering)
        corrected_clust = adjust_counts(
            clust_sc, clusters=False, method=method,
            round_to_int=False, verbose=verbose, tol=tol
        )

        # soup counts at cluster level
        soup_clust = scipy.sparse.csc_matrix(clust_counts) - corrected_clust
        soup_clust_df = pd.DataFrame(
            soup_clust.toarray(), index=sc.genes, columns=pd.Index(unique_clusters)
        )

        # cell-level weights for expand_clusters
        cell_weights = (sc.meta_data['nUMIs'].values * sc.meta_data['rho'].values)

        # Expand cluster-level soup counts to cell level
        soup_cells = expand_clusters(
            soup_clust_df, sc.toc, clusters, cell_weights, verbose=verbose
        )

        out = sc.toc.tocsc() - soup_cells

    else:
        # ── Single-cell level adjustment ──────────────────────────────────────
        if method == 'subtraction':
            out = _subtraction(sc, verbose)

        elif method == 'multinomial':
            out = _multinomial(sc, verbose)

        elif method == 'soupOnly':
            out = _soup_only(sc, verbose)

    # ── Optional stochastic rounding ──────────────────────────────────────────
    if round_to_int:
        if verbose > 1:
            print("Rounding to integers.")
        out_coo = out.tocoo().astype(float)
        floor_vals = np.floor(out_coo.data)
        frac_vals = out_coo.data - floor_vals
        out_coo.data = floor_vals + np.random.binomial(1, np.clip(frac_vals, 0, 1))
        out = out_coo.tocsc()

    return out.tocsc()


# ─── Subtraction method ───────────────────────────────────────────────────────

def _subtraction(sc, verbose):
    """
    Iterative weighted subtraction of expected soup counts.

    Distributes each cell's expected soup count (rho × nUMI) across genes
    proportional to the soup profile, subject to per-gene count caps.

    :param sc: SoupChannel with rho and soup_profile set.
    :type sc: SoupChannel
    :param verbose: Verbosity level (0 = silent).
    :type verbose: int
    :return: Corrected count matrix (genes × cells).
    :rtype: scipy.sparse.csc_matrix
    """
    toc = sc.toc.tocoo().astype(float)
    soup_frac = sc.soup_profile['est'].values          # (n_genes,)
    exp_soup_cnts = sc.meta_data['nUMIs'].values * sc.meta_data['rho'].values  # (n_cells,)

    data = toc.data.copy()
    rows = toc.row
    cols = toc.col

    # Group non-zero entries by cell (column)
    sort_idx = np.argsort(cols, kind='stable')
    s_rows = rows[sort_idx]
    s_cols = cols[sort_idx]
    s_data = data[sort_idx]

    unique_cols, col_starts = np.unique(s_cols, return_index=True)
    col_ends = np.append(col_starts[1:], len(s_cols))

    n_capped = 0
    for i, j in enumerate(unique_cols):
        start, end = col_starts[i], col_ends[i]
        gene_idx = s_rows[start:end]
        counts = s_data[start:end].copy()
        ws = soup_frac[gene_idx]
        allocated = alloc(exp_soup_cnts[j], counts, ws)
        if exp_soup_cnts[j] - allocated.sum() > 1e-9:
            n_capped += 1
        s_data[start:end] = counts - allocated

    if n_capped > 0:
        warnings.warn(
            f"{n_capped} cell(s) had expected soup exceeding allocatable counts "
            "(rho * nUMIs > expressible soup). Unallocatable soup was discarded; "
            "corrected counts are biased high for those cells. "
            "Consider checking whether rho is overestimated or soup profile has low "
            "overlap with cell expression.",
            stacklevel=3,
        )

    # Restore original order and drop non-positives
    unsort = np.argsort(sort_idx)
    data = s_data[unsort]

    pos = data > 0
    out = scipy.sparse.csc_matrix(
        (data[pos], (rows[pos], cols[pos])),
        shape=sc.toc.shape
    )
    return out


# ─── Multinomial method ───────────────────────────────────────────────────────

def _multinomial(sc, verbose):
    """
    Greedy swap optimisation to maximise multinomial likelihood.
    Sparse-aware: inner loop operates only on non-zero genes per cell,
    reducing per-iteration work from O(n_genes) to O(n_expressed).
    Initialised with the subtraction method result.
    """
    if verbose > 1:
        print("Initialising with subtraction method.")

    fit_init = _subtraction(sc, verbose=0)
    fit_init_coo = fit_init.tocoo().astype(float)
    _floor = np.floor(fit_init_coo.data)
    _frac = fit_init_coo.data - _floor
    fit_init_coo.data = _floor + np.random.binomial(1, np.clip(_frac, 0, 1))
    fit_init_round = fit_init_coo.tocsc()

    soup_frac = sc.soup_profile['est'].values    # (n_genes,)
    # Pre-compute log probs once — avoids repeated log() calls inside the per-cell loop.
    log_ps = np.log(np.maximum(soup_frac, 1e-300))
    n_genes = sc.toc.shape[0]
    n_cells = sc.toc.shape[1]
    toc_csc = sc.toc.tocsc()
    fit_init_csc = fit_init_round.tocsc()

    if verbose > 0:
        print(f"Fitting multinomial distribution to {n_cells} cells/clusters.")
        pb = init_prog_bar(n_cells)

    # Accumulate sparse output as COO triplets — avoids dense column_stack at the end.
    out_rows = []
    out_cols_idx = []
    out_data = []

    for i in range(n_cells):
        if verbose > 0:
            pb.update(1)

        tgt_n = int(round(sc.meta_data['rho'].values[i] * sc.meta_data['nUMIs'].values[i]))

        # Densify only this cell's column (O(n_genes) once per cell).
        col_full = toc_csc[:, i].toarray().flatten().astype(float)
        nz_genes = np.where(col_full > 0)[0]

        if len(nz_genes) == 0 or tgt_n == 0:
            # No expressed genes or no soup to remove — keep as-is.
            if len(nz_genes) > 0:
                out_rows.append(nz_genes)
                out_cols_idx.append(np.full(len(nz_genes), i, dtype=int))
                out_data.append(col_full[nz_genes])
            continue

        # Work on the expressed-gene subset only.
        # All greedy-swap iterations are O(n_expressed), not O(n_genes).
        lims = col_full[nz_genes]
        # Guard: rho > 1 from upstream bug → tgt_n > lims.sum() → loop increments forever.
        tgt_n = min(tgt_n, int(lims.sum()))

        init_full = fit_init_csc[:, i].toarray().flatten().astype(float)
        fit = lims - init_full[nz_genes]
        fit = np.clip(fit, 0, lims)

        ps_nz = log_ps[nz_genes]  # log probs restricted to non-zero genes

        while True:
            increasable = fit < lims
            decreasable = fit > 0

            if not increasable.any():
                break

            with np.errstate(divide='ignore', invalid='ignore'):
                del_inc = np.where(increasable, ps_nz - np.log(fit + 1), -np.inf)
                del_dec = np.where(decreasable, -ps_nz + np.log(np.maximum(fit, 1e-300)), -np.inf)

            w_inc_all = np.where(increasable & (del_inc == del_inc[increasable].max()))[0]
            w_dec_all = (np.where(decreasable & (del_dec == del_dec[decreasable].max()))[0]
                         if decreasable.any() else np.array([], dtype=int))

            w_inc = w_inc_all[np.random.randint(len(w_inc_all))]
            w_dec = w_dec_all[np.random.randint(len(w_dec_all))] if len(w_dec_all) > 0 else None

            cur_n = int(fit.sum())

            if cur_n < tgt_n:
                fit[w_inc] += 1
            elif cur_n > tgt_n:
                if w_dec is not None:
                    fit[w_dec] -= 1
                else:
                    break
            else:
                if w_dec is None:
                    break
                del_tot = del_inc[w_inc] + del_dec[w_dec]
                if del_tot == 0:
                    # Ambiguous: distribute evenly among tied candidates
                    zero_bucket = np.unique(np.concatenate([w_inc_all, w_dec_all]))
                    fit[w_dec_all] -= 1
                    fit[zero_bucket] += len(w_dec_all) / len(zero_bucket)
                    break
                elif del_tot < 0:
                    break
                else:
                    fit[w_inc] += 1
                    fit[w_dec] -= 1

        # fit = soup counts for nz_genes; corrected = lims - fit
        corrected_nz = lims - fit
        nz_corr = corrected_nz > 0
        if nz_corr.any():
            out_rows.append(nz_genes[nz_corr])
            out_cols_idx.append(np.full(int(nz_corr.sum()), i, dtype=int))
            out_data.append(corrected_nz[nz_corr])

    if verbose > 0:
        pb.close()

    if out_rows:
        rows = np.concatenate(out_rows)
        cols = np.concatenate(out_cols_idx)
        data = np.concatenate(out_data)
        return scipy.sparse.csc_matrix((data, (rows, cols)), shape=(n_genes, n_cells))
    return scipy.sparse.csc_matrix((n_genes, n_cells), dtype=float)


# ─── soupOnly method ─────────────────────────────────────────────────────────

def _soup_only(sc, verbose, pval_threshold=0.05):
    """
    Conservative adjustment: remove counts only where they are consistent with
    pure soup origin (R SoupX method='soupOnly').

    For each cell, genes are evaluated in ascending per-gene p-value order
    (most soup-like first).  A Fisher's combined test gates whether the gene
    and the cumulative removal so far are jointly consistent with soup:

        per-gene p  : P(X >= count  | mu_gene)          Poisson upper tail
        cumulative p: P(X >= cumul  | rho * nUMI)       Poisson upper tail
        Fisher stat : chi2.sf(-2 * log(p_gene * p_cumul), df=4)

    When the combined statistic falls below pval_threshold the gene's soup
    contribution (min(count, mu)) is removed and the loop continues.  The
    first gene that fails the combined test stops further removal for that
    cell — sequential stopping prevents over-removal.

    Removal amount is min(count, mu) in all cases, preserving genuine
    expression above the expected soup level.
    """
    soup_frac = sc.soup_profile['est'].values        # (n_genes,)
    rho       = sc.meta_data['rho'].values           # (n_cells,)
    nUMIs     = sc.meta_data['nUMIs'].values         # (n_cells,)

    toc_coo = sc.toc.tocoo().astype(float)
    rows = toc_coo.row
    cols = toc_coo.col
    data = toc_coo.data.copy()

    mu_nz = soup_frac[rows] * rho[cols] * nUMIs[cols]  # expected soup per entry

    # Per-entry Poisson upper-tail p-value: large = consistent with pure soup.
    p_vals = scipy.stats.poisson.sf(
        np.floor(data).astype(int) - 1,
        np.maximum(mu_nz, 1e-300),
    )

    # Sort entries: group by cell (ascending col), then ascending p-value within cell
    # so the most soup-like genes are tested first.
    order = np.lexsort((p_vals, cols))
    cell_groups: dict = {}
    for idx in order:
        cell_groups.setdefault(int(cols[idx]), []).append(idx)

    remove = np.zeros(len(data), dtype=float)

    for ci, indices in cell_groups.items():
        lam_total = float(nUMIs[ci] * rho[ci])   # expected total soup for this cell
        cumul_removed = 0.0
        for idx in indices:
            to_remove = min(data[idx], mu_nz[idx])
            # Cumulative p-value: P(X >= amount removed so far | lam_total).
            # Uses amount BEFORE adding current gene so the first gene only
            # depends on its own per-gene p-value (cumul_removed=0 → p_cumul=1).
            p_cumul = scipy.stats.poisson.sf(
                int(max(cumul_removed - 1, -1)), lam_total
            )
            combined_p = p_vals[idx] * p_cumul
            fisher_q = scipy.stats.chi2.sf(-2.0 * np.log(combined_p + 1e-300), df=4)
            if fisher_q < pval_threshold:
                remove[idx] = to_remove
                cumul_removed += to_remove
            else:
                break  # sequential stopping: first failure ends removal for this cell

    remaining = data - remove

    if verbose > 0:
        n_adjusted = int((remove > 0).sum())
        print(f"soupOnly: adjusted {n_adjusted} of {len(data)} non-zero entries "
              f"(pval_threshold={pval_threshold}).")

    pos = remaining > 0
    return scipy.sparse.csc_matrix(
        (remaining[pos], (rows[pos], cols[pos])),
        shape=sc.toc.shape,
        dtype=float,
    )


# ─── Internal proxy class ─────────────────────────────────────────────────────

@dataclass
class _ClusterProxy:
    """
    Minimal SoupChannel stand-in for the cluster-level adjust_counts recursive call.

    Using a dataclass makes every required attribute explicit: if adjust_counts ever
    accesses a new SoupChannel attribute, adding it here is a one-line fix and the
    omission surfaces immediately as a TypeError at construction time — not as an
    AttributeError deep inside a correction run.
    """
    toc: scipy.sparse.csc_matrix
    soup_profile: pd.DataFrame
    meta_data: pd.DataFrame
    genes: pd.Index
    cells: pd.Index

    def copy(self):
        import copy
        return copy.deepcopy(self)
