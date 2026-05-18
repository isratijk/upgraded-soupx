import numpy as np
import scipy.sparse
import pandas as pd


def alloc(tgt, bucket_lims, ws=None):
    """
    Distribute tgt counts across buckets subject to per-bucket caps and weights.

    Direct port of R SoupX alloc(). Distributes tgt proportional to ws but
    never exceeding bucket_lims for any bucket.

    Parameters
    ----------
    tgt : float
        Total amount to distribute.
    bucket_lims : array-like
        Maximum value for each bucket.
    ws : array-like, optional
        Weights for each bucket. Uniform if None.

    Returns
    -------
    np.ndarray
        Allocated values, same length as bucket_lims.
    """
    bucket_lims = np.asarray(bucket_lims, dtype=float)
    n = len(bucket_lims)
    if ws is None:
        ws = np.ones(n) / n
    ws = np.asarray(ws, dtype=float)
    # Sanitize: NaN or negative weights treated as zero
    ws = np.where(np.isfinite(ws) & (ws >= 0), ws, 0.0)

    # Normalise weights
    ws_sum = ws.sum()
    if ws_sum <= 0:
        return np.zeros(n)
    ws = ws / ws_sum

    # Fast path: no bucket hits its limit
    if np.all(tgt * ws <= bucket_lims):
        return tgt * ws

    # Order by the threshold at which each bucket fills up (bucket_lim / weight).
    # Use stable sort so all-inf ties (zero-weight buckets) are deterministic.
    with np.errstate(divide='ignore'):
        ratio = np.where(ws > 0, bucket_lims / ws, np.inf)
    o = np.argsort(ratio, kind='stable')
    w = ws[o]
    y = bucket_lims[o]

    # k[i] = tgt value at which bucket i fills up given allocation order
    cw = np.concatenate([[0.0], np.cumsum(w[:-1])])
    cy = np.concatenate([[0.0], np.cumsum(y[:-1])])
    with np.errstate(divide='ignore', invalid='ignore'):
        k = np.where(w > 0, y / w * (1.0 - cw) + cy, np.inf)

    # Buckets that are completely full
    b = k <= tgt
    resid = tgt - np.sum(y[b])
    remaining_w = 1.0 - np.sum(w[b])
    if remaining_w <= 0:
        # All weight consumed; fill everything to limit
        out = y.copy()
    else:
        w_remain = w / remaining_w
        out = np.where(b, y, resid * w_remain)

    # Restore original order
    result = np.empty(n)
    result[o] = out
    return result


def expand_clusters(clust_soup_cnts, cell_obs_cnts, clusters, cell_weights, verbose=1):
    """
    Expand cluster-level soup count estimates to individual cells.

    Parameters
    ----------
    clust_soup_cnts : pd.DataFrame
        (n_genes x n_clusters) soup counts per cluster. Columns = cluster IDs.
    cell_obs_cnts : scipy.sparse matrix
        (n_genes x n_cells) observed counts.
    clusters : array-like
        Cluster assignment for each cell (length n_cells), same order as
        columns of cell_obs_cnts.
    cell_weights : array-like
        Per-cell weight used to distribute soup within cluster (typically nUMIs*rho).
    verbose : int
        Verbosity level.

    Returns
    -------
    scipy.sparse.csc_matrix
        (n_genes x n_cells) estimated soup counts per cell.
    """
    ws = np.asarray(cell_weights, dtype=float)
    clusters = np.asarray(clusters, dtype=str)
    n_genes, n_cells = cell_obs_cnts.shape
    cluster_names = list(clust_soup_cnts.columns)

    if verbose > 0:
        print(f"Expanding counts from {len(cluster_names)} clusters to {n_cells} cells.")

    all_rows = []
    all_cols = []
    all_data = []

    cell_obs_csc = cell_obs_cnts.tocsc()

    for j, clust in enumerate(cluster_names):
        if verbose > 1:
            print(f"Expanding cluster {clust}")

        w_cells = np.where(clusters == str(clust))[0]
        if len(w_cells) == 0:
            continue

        ww = ws[w_cells]
        ww = np.where(np.isfinite(ww) & (ww >= 0), ww, 0.0)  # sanitize NaN/neg weights
        ww_sum = ww.sum()
        if ww_sum <= 0:
            continue
        ww = ww / ww_sum

        # Slice observation matrix to cells in this cluster
        lims = cell_obs_csc[:, w_cells].tocoo().astype(float)

        n_soup = np.asarray(clust_soup_cnts.iloc[:, j]).flatten()
        row_sums = np.asarray(cell_obs_csc[:, w_cells].sum(axis=1)).flatten()

        # Case 1: zero soup for gene → zero out those entries.
        # np.isin replaces the Python set-membership list comprehension.
        zero_soup_genes = np.where(n_soup == 0)[0]
        if len(zero_soup_genes):
            zero_mask = np.isin(lims.row, zero_soup_genes)
            lims.data[zero_mask] = 0.0

        # Cases 3/4: genes where 0 < n_soup < row_sum of lims
        active_genes = np.where((n_soup > 0) & (n_soup < row_sums))[0]
        if len(active_genes) > 0:
            active_mask = np.isin(lims.row, active_genes)
            active_idx = np.where(active_mask)[0]

            # Group entries by gene using np.unique — avoids Python dict/setdefault loop.
            active_rows = lims.row[active_idx]
            unique_active_genes, inverse_idx = np.unique(active_rows, return_inverse=True)
            for k, gene_idx in enumerate(unique_active_genes):
                e = active_idx[inverse_idx == k]
                local_cols = lims.col[e]
                lims.data[e] = alloc(n_soup[gene_idx], lims.data[e], ww[local_cols])

        # Collect non-zero results, mapping local col → global cell index
        valid = lims.data > 0
        if valid.any():
            all_rows.append(lims.row[valid])
            all_cols.append(w_cells[lims.col[valid]])
            all_data.append(lims.data[valid])

    if not all_rows:
        return scipy.sparse.csc_matrix((n_genes, n_cells), dtype=float)

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    data = np.concatenate(all_data)

    return scipy.sparse.csc_matrix((data, (rows, cols)), shape=(n_genes, n_cells))


def init_prog_bar(total):
    """
    Return a tqdm progress bar if tqdm is installed, otherwise a no-op stub.

    :param total: Total number of steps for the progress bar.
    :type total: int
    :return: A tqdm progress bar object (or a no-op stub with update/close methods).
    :rtype: tqdm.tqdm or _Noop
    """
    try:
        from tqdm import tqdm
        return tqdm(total=total)
    except ImportError:
        class _Noop:
            def update(self, n=1): pass
            def close(self): pass
        return _Noop()
