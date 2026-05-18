"""
Downstream analysis on corrected count matrices.

All functions accept the (genes × cells) sparse matrix returned by adjust_counts().
Optional heavy dependencies (umap-learn, leidenalg, python-igraph) are imported
lazily — the rest of the pipeline works without them.

Typical usage
-------------
>>> from SoupX import load_10x, auto_est_cont, adjust_counts
>>> from SoupX.downstream import run_downstream, plot_embedding
>>> sc  = auto_est_cont(load_10x('path/to/cellranger'))
>>> out = adjust_counts(sc)                        # genes × cells sparse
>>> dn  = run_downstream(out, sc.genes.tolist())
>>> plot_embedding(dn['embedding'], dn['cluster_labels'], title='UMAP — corrected')
"""

import warnings
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.stats import ranksums


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_dense(matrix):
    """
    Convert a sparse or dense matrix to a float64 numpy array.

    :param matrix: Input matrix (genes × cells). Sparse or dense.
    :type matrix: scipy.sparse matrix or np.ndarray
    :return: Dense float64 numpy array with same shape.
    :rtype: np.ndarray
    """
    if scipy.sparse.issparse(matrix):
        return matrix.toarray().astype(float)
    return np.asarray(matrix, dtype=float)


def normalize_log1p(matrix, target_sum=1e4):
    """
    Library-size normalise then log1p-transform.

    Parameters
    ----------
    matrix : (genes × cells) sparse or dense
    target_sum : counts per cell after scaling

    Returns
    -------
    ndarray (cells × genes), float64
    """
    mat = _to_dense(matrix).T          # → cells × genes
    totals = mat.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    return np.log1p(mat / totals * target_sum)


# ── PCA ───────────────────────────────────────────────────────────────────────

def run_pca(matrix, gene_names, n_components=50, n_top_genes=2000):
    """
    PCA on normalised, HVG-filtered count matrix.

    Parameters
    ----------
    matrix : (genes × cells) sparse or dense — output of adjust_counts()
    gene_names : array-like, length = n_genes
    n_components : PCA components to compute
    n_top_genes : highly-variable genes to keep before PCA

    Returns
    -------
    dict
        'embedding'      : ndarray (n_cells × n_components)
        'variance_ratio' : ndarray (n_components,)
        'gene_names_used': ndarray of gene names kept for PCA
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    gene_names = np.asarray(gene_names)
    mat = normalize_log1p(matrix)                  # cells × genes

    # Highly-variable genes by variance
    if mat.shape[1] > n_top_genes:
        var = mat.var(axis=0)
        top_idx = np.argsort(var)[-n_top_genes:]
        mat = mat[:, top_idx]
        used_genes = gene_names[top_idx]
    else:
        used_genes = gene_names

    mat_scaled = StandardScaler().fit_transform(mat)
    n_comp = min(n_components, mat_scaled.shape[0] - 1, mat_scaled.shape[1] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    embedding = pca.fit_transform(mat_scaled)

    return {
        'embedding':       embedding,
        'variance_ratio':  pca.explained_variance_ratio_,
        'gene_names_used': used_genes,
    }


# ── Embeddings ────────────────────────────────────────────────────────────────

def run_umap(pca_result, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    UMAP 2-D embedding from a PCA result dict or embedding array.

    Requires ``umap-learn``: ``pip install umap-learn``.

    :param pca_result: Either a dict (output of run_pca) with key ``'embedding'``,
        or a plain ndarray (n_cells × n_components).
    :type pca_result: dict or np.ndarray
    :param n_neighbors: Number of nearest neighbours for UMAP graph construction.
    :type n_neighbors: int
    :param min_dist: Minimum distance between points in the 2-D embedding.
    :type min_dist: float
    :param random_state: Random seed for reproducibility.
    :type random_state: int
    :return: 2-D UMAP coordinates, shape (n_cells, 2).
    :rtype: np.ndarray
    :raises ImportError: If umap-learn is not installed.
    """
    try:
        import umap as umap_mod
    except ImportError:
        raise ImportError("umap-learn required: pip install umap-learn")

    emb = pca_result['embedding'] if isinstance(pca_result, dict) else pca_result
    reducer = umap_mod.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state
    )
    return reducer.fit_transform(emb)


def run_tsne(pca_result, n_components=2, perplexity=30, random_state=42):
    """
    t-SNE 2-D embedding from a PCA result dict or embedding array.

    Uses scikit-learn ``TSNE``; no extra install required beyond the core extras.

    :param pca_result: Either a dict (output of run_pca) with key ``'embedding'``,
        or a plain ndarray (n_cells × n_components).
    :type pca_result: dict or np.ndarray
    :param n_components: Dimensionality of the t-SNE output (usually 2).
    :type n_components: int
    :param perplexity: Perplexity parameter. Clamped to ``n_cells // 3`` automatically.
    :type perplexity: float
    :param random_state: Random seed for reproducibility.
    :type random_state: int
    :return: t-SNE coordinates, shape (n_cells, n_components).
    :rtype: np.ndarray
    """
    from sklearn.manifold import TSNE

    emb = pca_result['embedding'] if isinstance(pca_result, dict) else pca_result
    safe_perp = min(perplexity, max(1, emb.shape[0] // 3))
    tsne = TSNE(
        n_components=n_components, perplexity=safe_perp, random_state=random_state
    )
    return tsne.fit_transform(emb[:, :min(50, emb.shape[1])])


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_leiden(pca_result, resolution=0.5, n_neighbors=15):
    """
    Leiden community detection on a k-NN graph built from a PCA embedding.

    Requires ``leidenalg`` and ``python-igraph``:
    ``pip install leidenalg python-igraph``.

    :param pca_result: Either a dict (output of run_pca) with key ``'embedding'``,
        or a plain ndarray (n_cells × n_components).
    :type pca_result: dict or np.ndarray
    :param resolution: Leiden resolution parameter. Higher = more, smaller clusters.
    :type resolution: float
    :param n_neighbors: Number of nearest neighbours for the k-NN graph.
    :type n_neighbors: int
    :return: Cluster labels as string array, shape (n_cells,).
    :rtype: np.ndarray
    :raises ImportError: If leidenalg or python-igraph is not installed.
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        raise ImportError(
            "Leiden clustering requires: pip install leidenalg python-igraph"
        )
    from sklearn.neighbors import NearestNeighbors

    emb = pca_result['embedding'] if isinstance(pca_result, dict) else pca_result
    n_cells = emb.shape[0]
    k = min(n_neighbors, n_cells - 1)

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    distances, indices = nbrs.kneighbors(emb)

    edges, weights = [], []
    for i, (nbr_idx, nbr_dist) in enumerate(zip(indices[:, 1:], distances[:, 1:])):
        for j, d in zip(nbr_idx, nbr_dist):
            edges.append((i, int(j)))
            weights.append(float(1.0 / (1.0 + d)))

    g = ig.Graph(n=n_cells, edges=edges, directed=False)
    g.es['weight'] = weights
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights='weight', resolution_parameter=resolution, seed=42,
    )
    return np.array(partition.membership).astype(str)


def cluster_kmeans(pca_result, n_clusters=10, random_state=42):
    """
    k-means clustering on a PCA embedding.

    Uses scikit-learn ``KMeans``; no extra install required beyond the core extras.

    :param pca_result: Either a dict (output of run_pca) with key ``'embedding'``,
        or a plain ndarray (n_cells × n_components).
    :type pca_result: dict or np.ndarray
    :param n_clusters: Number of clusters. Clamped to ``n_cells - 1`` automatically.
    :type n_clusters: int
    :param random_state: Random seed for reproducibility.
    :type random_state: int
    :return: Cluster labels as string array, shape (n_cells,).
    :rtype: np.ndarray
    """
    from sklearn.cluster import KMeans

    emb = pca_result['embedding'] if isinstance(pca_result, dict) else pca_result
    n_clusters = min(n_clusters, emb.shape[0] - 1)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(emb).astype(str)


# ── Differential Expression ───────────────────────────────────────────────────

def differential_expression(matrix, gene_names, cluster_labels,
                             min_cells=5, top_n=20, log2fc_thresh=0.25):
    """
    One-vs-rest Wilcoxon rank-sum DE per cluster.

    Parameters
    ----------
    matrix : (genes × cells) sparse or dense
    gene_names : array-like, length = n_genes
    cluster_labels : array-like, length = n_cells
    min_cells : minimum cells per cluster to run DE
    top_n : top N genes per cluster to return
    log2fc_thresh : pre-filter genes below this log2FC

    Returns
    -------
    pd.DataFrame
        columns: cluster, gene, statistic, pvalue, log2fc, rank
    """
    gene_names = np.asarray(gene_names)
    labels = np.asarray(cluster_labels)
    mat = _to_dense(matrix)    # genes × cells

    records = []
    for cl in np.unique(labels):
        in_mask  = labels == cl
        out_mask = ~in_mask
        if in_mask.sum() < min_cells or out_mask.sum() < min_cells:
            continue

        in_mat  = mat[:, in_mask]
        out_mat = mat[:, out_mask]

        eps = 1e-9
        log2fc = np.log2((in_mat.mean(axis=1) + eps) / (out_mat.mean(axis=1) + eps))
        candidate = np.where(log2fc > log2fc_thresh)[0]

        stats = np.zeros(len(gene_names))
        pvals = np.ones(len(gene_names))
        for gi in candidate:
            stat, pv = ranksums(in_mat[gi], out_mat[gi])
            stats[gi] = stat
            pvals[gi] = pv

        for rank, gi in enumerate(np.argsort(-stats)[:top_n]):
            if pvals[gi] < 0.05:
                records.append({
                    'cluster':   cl,
                    'gene':      gene_names[gi],
                    'statistic': float(stats[gi]),
                    'pvalue':    float(pvals[gi]),
                    'log2fc':    float(log2fc[gi]),
                    'rank':      rank + 1,
                })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=['cluster', 'gene', 'statistic', 'pvalue', 'log2fc', 'rank']
    )


# ── Cell-type Scoring ─────────────────────────────────────────────────────────

def score_cell_types(matrix, gene_names, marker_dict):
    """
    Score each cell against marker gene sets (mean log-normalised expression).

    Parameters
    ----------
    matrix : (genes × cells) sparse or dense
    gene_names : array-like
    marker_dict : dict  cell_type -> list of marker gene names

    Returns
    -------
    pd.DataFrame  (n_cells × n_cell_types)
    """
    gene_names = np.asarray(gene_names)
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    norm = normalize_log1p(matrix)                 # cells × genes

    scores = {}
    for cell_type, markers in marker_dict.items():
        present = [gene_idx[g] for g in markers if g in gene_idx]
        if not present:
            warnings.warn(f"score_cell_types: no markers for '{cell_type}' found in gene_names")
            scores[cell_type] = np.zeros(norm.shape[0])
        else:
            scores[cell_type] = norm[:, present].mean(axis=1)

    return pd.DataFrame(scores)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_embedding(embedding, labels=None, title='', ax=None, point_size=8):
    """
    Scatter plot of 2-D embedding coloured by labels.

    Parameters
    ----------
    embedding : ndarray (n_cells × 2)
    labels : array-like or None — colour categories
    title : str
    ax : matplotlib Axes or None
    point_size : marker size

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    emb = np.asarray(embedding)

    if labels is not None:
        labels = np.asarray(labels)
        unique = np.unique(labels)
        cmap = plt.colormaps.get_cmap('tab20').resampled(len(unique))
        for i, lab in enumerate(unique):
            mask = labels == lab
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=[cmap(i)], s=point_size, alpha=0.7,
                label=str(lab), rasterized=True,
            )
        ax.legend(
            markerscale=3, fontsize=7, loc='best',
            framealpha=0.5, ncol=max(1, len(unique) // 15),
        )
    else:
        ax.scatter(emb[:, 0], emb[:, 1], s=point_size, alpha=0.7, rasterized=True)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def plot_top_de_genes(de_df, n_genes=10, ax=None):
    """
    Horizontal bar chart of top DE genes per cluster (by statistic).

    Parameters
    ----------
    de_df : pd.DataFrame from differential_expression()
    n_genes : genes per cluster to show
    ax : matplotlib Axes or None

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if de_df.empty:
        warnings.warn("plot_top_de_genes: DE DataFrame is empty")
        return ax

    clusters = de_df['cluster'].unique()
    fig_height = max(4, len(clusters) * n_genes * 0.25)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, fig_height))

    top = (
        de_df.sort_values('statistic', ascending=False)
             .groupby('cluster')
             .head(n_genes)
    )
    top = top.sort_values(['cluster', 'statistic'], ascending=[True, False])
    labels = [f"{row.cluster}:{row.gene}" for _, row in top.iterrows()]

    ax.barh(labels, top['log2fc'].values, color='steelblue', alpha=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('log2 FC (vs rest)')
    ax.set_title('Top DE genes per cluster', fontsize=11)
    ax.invert_yaxis()
    return ax


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_downstream(corrected_matrix, gene_names,
                   cluster_labels=None,
                   n_pca=50,
                   embedding='umap',
                   clustering='auto',
                   n_clusters=10,
                   leiden_resolution=0.5,
                   marker_dict=None,
                   run_de=True):
    """
    Run the full downstream analysis pipeline on a corrected count matrix.

    Parameters
    ----------
    corrected_matrix : (genes × cells) sparse — output of adjust_counts()
    gene_names : array-like, length = n_genes
    cluster_labels : optional pre-computed labels (skips clustering step)
    n_pca : PCA components
    embedding : 'umap' | 'tsne' | None
        'umap' falls back to 'tsne' if umap-learn is not installed
    clustering : 'leiden' | 'kmeans' | 'auto' | None
        'auto' tries leiden, falls back to kmeans
    n_clusters : k for kmeans
    leiden_resolution : resolution parameter for leiden
    marker_dict : optional dict  cell_type -> [genes]  for cell-type scoring
    run_de : run differential expression

    Returns
    -------
    dict with keys:
        'pca'              – result dict from run_pca()
        'embedding'        – ndarray (n_cells × 2) or None
        'embedding_method' – 'umap' | 'tsne' | None
        'cluster_labels'   – ndarray (n_cells,) str or None
        'cluster_method'   – 'leiden' | 'kmeans' | 'provided' | None
        'de_results'       – pd.DataFrame or None
        'cell_type_scores' – pd.DataFrame or None
    """
    results = {}

    # 1. PCA ─────────────────────────────────────────────────────────────────
    print("  [downstream] PCA ...")
    pca_result = run_pca(corrected_matrix, gene_names, n_components=n_pca)
    results['pca'] = pca_result

    # 2. Embedding ────────────────────────────────────────────────────────────
    if embedding == 'umap':
        try:
            print("  [downstream] UMAP ...")
            results['embedding'] = run_umap(pca_result)
            results['embedding_method'] = 'umap'
        except ImportError as e:
            warnings.warn(f"UMAP skipped ({e}). Falling back to t-SNE.")
            print("  [downstream] t-SNE (fallback) ...")
            results['embedding'] = run_tsne(pca_result)
            results['embedding_method'] = 'tsne'
    elif embedding == 'tsne':
        print("  [downstream] t-SNE ...")
        results['embedding'] = run_tsne(pca_result)
        results['embedding_method'] = 'tsne'
    else:
        results['embedding'] = None
        results['embedding_method'] = None

    # 3. Clustering ───────────────────────────────────────────────────────────
    if cluster_labels is not None:
        results['cluster_labels'] = np.asarray(cluster_labels).astype(str)
        results['cluster_method'] = 'provided'
    elif clustering in ('leiden', 'auto'):
        try:
            print("  [downstream] Leiden clustering ...")
            results['cluster_labels'] = cluster_leiden(
                pca_result, resolution=leiden_resolution
            )
            results['cluster_method'] = 'leiden'
        except ImportError as e:
            if clustering == 'leiden':
                raise
            warnings.warn(f"Leiden skipped ({e}). Falling back to k-means.")
            print(f"  [downstream] k-means (k={n_clusters}) ...")
            results['cluster_labels'] = cluster_kmeans(pca_result, n_clusters=n_clusters)
            results['cluster_method'] = 'kmeans'
    elif clustering == 'kmeans':
        print(f"  [downstream] k-means (k={n_clusters}) ...")
        results['cluster_labels'] = cluster_kmeans(pca_result, n_clusters=n_clusters)
        results['cluster_method'] = 'kmeans'
    else:
        results['cluster_labels'] = None
        results['cluster_method'] = None

    # 4. Differential expression ──────────────────────────────────────────────
    if run_de and results.get('cluster_labels') is not None:
        print("  [downstream] Differential expression ...")
        results['de_results'] = differential_expression(
            corrected_matrix, gene_names, results['cluster_labels']
        )
    else:
        results['de_results'] = None

    # 5. Cell-type scoring ────────────────────────────────────────────────────
    if marker_dict is not None:
        print("  [downstream] Cell-type scoring ...")
        results['cell_type_scores'] = score_cell_types(
            corrected_matrix, gene_names, marker_dict
        )
    else:
        results['cell_type_scores'] = None

    n_cl = (len(np.unique(results['cluster_labels']))
            if results['cluster_labels'] is not None else 0)
    n_cells = (corrected_matrix.shape[1]
               if hasattr(corrected_matrix, 'shape') else '?')
    print(f"  [downstream] Done. {n_cells} cells, {n_cl} clusters "
          f"({results['cluster_method']}).")
    return results
