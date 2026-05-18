"""
Assessment metrics for evaluating ambient RNA correction quality.

Five metrics corresponding to the original SoupX paper (Young & Behjati 2020):
  1. cross_species_reduction  — ≥2x fold reduction in cross-species UMIs
  2. marker_fold_change       — marker genes become more cell-type-specific
  3. cluster_membership_delta — artificial clusters disappear after correction
  4. batch_entropy            — cross-batch mixing improves after correction
  5. hbb_expression_analysis  — HBB/HBA removed from non-erythroid cells

All functions accept (toc_raw, toc_corrected, ...) scipy sparse matrices and
return a dict, making results easy to tabulate for baseline vs upgraded comparison.
"""

import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import entropy as scipy_entropy


# ── 1. Cross-species contamination ─────────────────────────────────────────────

def cross_species_reduction(toc_raw, toc_corrected, gene_names, cell_species):
    """
    Measure cross-species contamination fold reduction.

    In a barnyard experiment (human + mouse mixed), contamination is directly
    observable as the fraction of UMIs from the wrong species.  After
    correction this should drop by ≥2x.

    Gene names must carry species prefixes:
      human: 'hg19_*', 'GRCh38_*', 'hg_*'
      mouse: 'mm10_*', 'mm9_*', 'mm_*'
    Fallback: uppercase first letter = human, lowercase = mouse.

    Parameters
    ----------
    toc_raw, toc_corrected : sparse (n_genes, n_cells)
    gene_names   : array-like str, length n_genes
    cell_species : array-like ('human'/'mouse'), length n_cells

    Returns
    -------
    dict
        human_before, human_after    cross-species fraction in human cells
        mouse_before, mouse_after    cross-species fraction in mouse cells
        contamination_before/after   mean of both
        fold_reduction               before / after
        meets_2fold_threshold        bool
    """
    gene_names   = np.asarray(gene_names)
    cell_species = np.asarray(cell_species)

    human_mask  = _species_gene_mask(gene_names, 'human')
    mouse_mask  = _species_gene_mask(gene_names, 'mouse')
    human_cells = cell_species == 'human'
    mouse_cells = cell_species == 'mouse'

    if not (human_cells.any() and mouse_cells.any()):
        raise ValueError("cell_species must contain both 'human' and 'mouse'.")
    if not (human_mask.any() and mouse_mask.any()):
        raise ValueError(
            "Cannot detect species from gene_names. "
            "Prefix with 'hg_'/'mm_' or 'GRCh38_'/'mm10_'."
        )

    def _frac(mat, wrong_gene_mask, right_cell_mask):
        m     = sparse.csc_matrix(mat)
        wrong = float(m[wrong_gene_mask, :][:, right_cell_mask].sum())
        total = float(m[:, right_cell_mask].sum())
        return wrong / (total + 1e-10)

    h_bef = _frac(toc_raw,       mouse_mask, human_cells)
    h_aft = _frac(toc_corrected, mouse_mask, human_cells)
    m_bef = _frac(toc_raw,       human_mask, mouse_cells)
    m_aft = _frac(toc_corrected, human_mask, mouse_cells)

    bef  = (h_bef + m_bef) / 2.0
    aft  = (h_aft + m_aft) / 2.0
    fold = bef / (aft + 1e-10)

    return {
        'human_before':          h_bef,
        'human_after':           h_aft,
        'mouse_before':          m_bef,
        'mouse_after':           m_aft,
        'contamination_before':  bef,
        'contamination_after':   aft,
        'fold_reduction':        fold,
        'meets_2fold_threshold': fold >= 2.0,
    }


def _species_gene_mask(gene_names, species):
    """
    Return boolean mask of genes belonging to the given species.

    :param gene_names: Array of gene names (may carry species prefixes).
    :type gene_names: np.ndarray
    :param species: ``'human'`` or ``'mouse'``.
    :type species: str
    :return: Boolean array, True for genes of the given species.
    :rtype: np.ndarray
    """
    prefixes = (
        ('hg19_', 'GRCh38_', 'hg_', 'Human_') if species == 'human'
        else ('mm10_', 'mm9_', 'mm_', 'Mouse_')
    )
    sample = gene_names[:200]
    for p in prefixes:
        if any(g.startswith(p) for g in sample):
            return np.array([g.startswith(p) for g in gene_names])
    if species == 'human':
        return np.array([bool(g) and g[0].isupper() for g in gene_names])
    return np.array([bool(g) and g[0].islower() for g in gene_names])


# ── 2. Marker gene fold change ──────────────────────────────────────────────────

def marker_fold_change(toc_raw, toc_corrected, clusters, marker_genes, gene_names):
    """
    Measure how marker gene specificity changes after correction.

    Fold change = mean_CPM_in_target_cluster / mean_CPM_in_all_other_clusters.
    Soup contamination inflates off-target expression, reducing fold change.
    After correction, fold changes should increase (markers become more specific).

    Parameters
    ----------
    toc_raw, toc_corrected : sparse (n_genes, n_cells)
    clusters     : array-like cluster labels, length n_cells
    marker_genes : dict {cluster_label: [gene_name, ...]}
                   OR list of gene names (auto-assigned to highest-expressing cluster)
    gene_names   : array-like str, length n_genes

    Returns
    -------
    dict
        per_gene       DataFrame: gene, cluster, fc_before, fc_after, fc_ratio
        mean_fc_before, mean_fc_after   mean fold change across all markers
        fc_ratio       after / before  (> 1 = correction helped)
        improved       bool, majority of markers have higher FC after correction
    """
    gene_names = np.asarray(gene_names)
    clusters   = np.asarray(clusters)

    raw_cpm  = _cpm(toc_raw)
    corr_cpm = _cpm(toc_corrected)

    marker_dict = _to_marker_dict(marker_genes, clusters, gene_names, raw_cpm)
    rows = []
    for clust, genes in marker_dict.items():
        tgt  = clusters == clust
        rest = ~tgt
        if not tgt.any() or not rest.any():
            continue
        for gene in genes:
            idx = np.where(gene_names == gene)[0]
            if not len(idx):
                warnings.warn(f"marker gene '{gene}' not found in gene_names",
                              stacklevel=2)
                continue
            g      = idx[0]
            fc_bef = _row_mean(raw_cpm,  g, tgt) / (_row_mean(raw_cpm,  g, rest) + 1e-4)
            fc_aft = _row_mean(corr_cpm, g, tgt) / (_row_mean(corr_cpm, g, rest) + 1e-4)
            rows.append({'gene': gene, 'cluster': clust,
                         'fc_before': fc_bef, 'fc_after': fc_aft,
                         'fc_ratio':  fc_aft / (fc_bef + 1e-4)})

    if not rows:
        raise ValueError("No marker genes matched gene_names.")

    df = pd.DataFrame(rows)
    return {
        'per_gene':       df,
        'mean_fc_before': float(df['fc_before'].mean()),
        'mean_fc_after':  float(df['fc_after'].mean()),
        'fc_ratio':       float(df['fc_after'].mean()) / (float(df['fc_before'].mean()) + 1e-4),
        'improved':       bool((df['fc_after'] > df['fc_before']).mean() > 0.5),
    }


def _cpm(mat):
    """
    Normalize a count matrix to counts-per-million (CPM).

    :param mat: Sparse count matrix (genes × cells).
    :type mat: scipy.sparse matrix
    :return: CPM-normalized sparse matrix (genes × cells).
    :rtype: scipy.sparse.csc_matrix
    """
    m = sparse.csc_matrix(mat).astype(float)
    s = np.array(m.sum(axis=0)).flatten()
    s[s == 0] = 1.0
    return sparse.csc_matrix(m.multiply(1e6 / s))


def _row_mean(cpm_mat, gene_idx, cell_mask):
    """
    Mean CPM for one gene across a subset of cells.

    :param cpm_mat: CPM matrix (genes × cells).
    :type cpm_mat: scipy.sparse.csc_matrix
    :param gene_idx: Row index of the gene.
    :type gene_idx: int
    :param cell_mask: Boolean mask selecting cells.
    :type cell_mask: np.ndarray
    :return: Mean CPM value.
    :rtype: float
    """
    row = cpm_mat[gene_idx, :]
    row = row.toarray().flatten() if sparse.issparse(row) else np.asarray(row).flatten()
    return float(row[cell_mask].mean())


def _to_marker_dict(marker_genes, clusters, gene_names, raw_cpm):
    """
    Convert a flat gene list to a cluster-keyed marker dict by highest mean CPM.

    :param marker_genes: Either a dict {cluster: [genes]} or a list of gene names.
    :type marker_genes: dict or list
    :param clusters: Cluster label per cell.
    :type clusters: np.ndarray
    :param gene_names: Gene names array.
    :type gene_names: np.ndarray
    :param raw_cpm: CPM matrix for highest-expressing cluster assignment.
    :type raw_cpm: scipy.sparse.csc_matrix
    :return: Dict mapping cluster label to list of marker gene names.
    :rtype: dict
    """
    if isinstance(marker_genes, dict):
        return marker_genes
    unique = np.unique(clusters)
    d = {c: [] for c in unique}
    for gene in marker_genes:
        idx = np.where(gene_names == gene)[0]
        if not len(idx):
            continue
        g    = idx[0]
        best = max(unique, key=lambda c: _row_mean(raw_cpm, g, clusters == c))
        d[best].append(gene)
    return {c: genes for c, genes in d.items() if genes}


# ── 3. Cluster membership changes ──────────────────────────────────────────────

def cluster_membership_delta(toc_raw, toc_corrected, n_clusters=None,
                              n_pcs=10, seed=42):
    """
    Detect structural cluster changes after correction.

    Artificial clusters driven by contamination should collapse or merge when
    contamination is removed.  k-means is applied to both raw and corrected
    matrices; the number of occupied clusters and cell movement are reported.

    Parameters
    ----------
    toc_raw, toc_corrected : sparse (n_genes, n_cells)
    n_clusters : int, k for k-means.  Default: sqrt(n_cells/2), min 2.
    n_pcs      : int, PCA components before clustering.
    seed       : int

    Returns
    -------
    dict
        n_clusters_k          k used
        n_occupied_before/after  clusters with > 0.5% of cells
        n_clusters_lost       max(0, before − after)
        n_cells_changed       cells that moved cluster
        pct_cells_changed     percentage
        ari                   adjusted Rand index between before/after (1=identical)
        labels_before, labels_after  np.ndarray of integer labels
    """
    n_cells = toc_raw.shape[1]
    k = (n_clusters if n_clusters is not None
         else max(2, int(round(np.sqrt(n_cells / 2)))))
    k = min(k, n_cells)

    lb = _kmeans_on(toc_raw,       k, n_pcs, seed)
    la = _kmeans_on(toc_corrected, k, n_pcs, seed)

    min_sz = max(1, int(n_cells * 0.005))
    occ_b  = sum(1 for c in np.unique(lb) if (lb == c).sum() >= min_sz)
    occ_a  = sum(1 for c in np.unique(la) if (la == c).sum() >= min_sz)
    changed = int((lb != la).sum())

    return {
        'n_clusters_k':      k,
        'n_occupied_before': occ_b,
        'n_occupied_after':  occ_a,
        'n_clusters_lost':   max(0, occ_b - occ_a),
        'n_cells_changed':   changed,
        'pct_cells_changed': changed / n_cells * 100,
        'ari':               _ari(lb, la),
        'labels_before':     lb,
        'labels_after':      la,
    }


def _kmeans_on(mat, k, n_pcs, seed):
    """
    Run k-means on a PCA embedding of the count matrix.

    :param mat: Count matrix (genes × cells).
    :type mat: scipy.sparse matrix
    :param k: Number of clusters.
    :type k: int
    :param n_pcs: Number of PCA components to compute before clustering.
    :type n_pcs: int
    :param seed: Random seed.
    :type seed: int
    :return: Integer cluster label array, shape (n_cells,).
    :rtype: np.ndarray
    """
    from scipy.cluster.vq import kmeans2
    from scipy.sparse.linalg import svds

    m  = sparse.csc_matrix(mat).astype(float)
    s  = np.array(m.sum(axis=0)).flatten()
    s[s == 0] = 1.0
    mn = m.multiply(1e4 / s)

    n_genes, n_cells = mn.shape
    k_svd = min(n_pcs, min(n_genes, n_cells) - 1)

    try:
        rng = np.random.default_rng(seed)
        v0  = rng.standard_normal(min(mn.shape))
        U, _, _ = svds(mn.T.tocsr(), k=max(1, k_svd), v0=v0)
        X = U   # shape (n_cells, k) — left singular vectors
    except Exception:
        X = mn.T.toarray()[:, :n_pcs]

    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xw = X / std

    _, labels = kmeans2(Xw, k=min(k, Xw.shape[0]),
                        iter=100, minit='points', seed=seed)
    return labels


def _ari(a, b):
    """
    Adjusted Rand Index between two label arrays (pure-Python, no sklearn).

    :param a: First label array.
    :type a: array-like
    :param b: Second label array.
    :type b: array-like
    :return: ARI in [-1, 1]. 1 = perfect agreement.
    :rtype: float
    """
    a, b = np.asarray(a), np.asarray(b)
    n    = len(a)
    ca, cb = np.unique(a), np.unique(b)
    ai = {c: i for i, c in enumerate(ca)}
    bi = {c: i for i, c in enumerate(cb)}
    C  = np.zeros((len(ca), len(cb)), dtype=np.int64)
    for ai_, bi_ in zip(a, b):
        C[ai[ai_], bi[bi_]] += 1
    sc  = sum(_c2(v) for row in C for v in row)
    sa  = sum(_c2(v) for v in C.sum(axis=1))
    sb  = sum(_c2(v) for v in C.sum(axis=0))
    exp = sa * sb / (_c2(n) + 1e-10)
    mx  = (sa + sb) / 2.0
    den = mx - exp
    if den < 1e-10:
        return 1.0 if sc == mx else 0.0
    return (sc - exp) / den


def _c2(n):
    """
    Combinatorial C(n, 2) = n*(n-1)/2. Returns 0 for n < 2.

    :param n: Non-negative integer.
    :type n: int
    :return: Number of pairs.
    :rtype: int
    """
    return n * (n - 1) // 2 if n >= 2 else 0


# ── 4. Batch entropy ────────────────────────────────────────────────────────────

def batch_entropy(toc_raw, toc_corrected, batch_labels, n_neighbors=15, n_pcs=10):
    """
    Measure cross-batch mixing via k-NN Shannon entropy.

    For each cell, compute Shannon entropy of batch labels among its k nearest
    neighbors in PCA space.  Ambient RNA creates batch-specific expression
    artifacts; after correction, biologically similar cells from different
    batches should cluster together (higher entropy = better mixing).

    Parameters
    ----------
    toc_raw, toc_corrected : sparse (n_genes, n_cells)
    batch_labels           : array-like str, length n_cells
    n_neighbors            : int, k for kNN (default 15)
    n_pcs                  : int, PCA components (default 10)

    Returns
    -------
    dict
        mean_entropy_before/after   mean Shannon entropy across cells
        entropy_increase            after − before  (positive = improved mixing)
        max_entropy                 log(n_batches), theoretical uniform max
        normalized_before/after     entropy / max_entropy
        improved                    bool
    """
    batch_labels = np.asarray(batch_labels)
    uniq         = np.unique(batch_labels)
    if len(uniq) < 2:
        raise ValueError("batch_labels must have ≥2 distinct batches.")

    max_ent = float(np.log(len(uniq)))
    emb_b   = _pca(toc_raw,       n_pcs)
    emb_a   = _pca(toc_corrected, n_pcs)
    ent_b   = _knn_entropy(emb_b, batch_labels, n_neighbors)
    ent_a   = _knn_entropy(emb_a, batch_labels, n_neighbors)

    return {
        'mean_entropy_before': float(ent_b.mean()),
        'mean_entropy_after':  float(ent_a.mean()),
        'entropy_increase':    float(ent_a.mean() - ent_b.mean()),
        'max_entropy':         max_ent,
        'normalized_before':   float(ent_b.mean()) / (max_ent + 1e-10),
        'normalized_after':    float(ent_a.mean()) / (max_ent + 1e-10),
        'improved':            bool(ent_a.mean() > ent_b.mean()),
    }


def _pca(mat, n_pcs):
    """
    Compute a PCA embedding of the count matrix via sparse SVD.

    :param mat: Count matrix (genes × cells).
    :type mat: scipy.sparse matrix
    :param n_pcs: Number of principal components.
    :type n_pcs: int
    :return: Embedding array (n_cells × n_pcs).
    :rtype: np.ndarray
    """
    from scipy.sparse.linalg import svds
    m = sparse.csc_matrix(mat).astype(float)
    s = np.array(m.sum(axis=0)).flatten()
    s[s == 0] = 1.0
    mn = m.multiply(1e4 / s)
    n_genes, n_cells = mn.shape
    k = min(n_pcs, min(n_genes, n_cells) - 1)
    try:
        U, _, _ = svds(mn.T.tocsr(), k=max(1, k))
        return U  # shape (n_cells, k) — left singular vectors
    except Exception:
        return mn.T.toarray()[:, :n_pcs]


def _knn_entropy(emb, batch_labels, k):
    """
    Compute per-cell Shannon entropy of batch labels among k nearest neighbours.

    :param emb: Embedding array (n_cells × n_dims).
    :type emb: np.ndarray
    :param batch_labels: Batch label per cell, length n_cells.
    :type batch_labels: array-like
    :param k: Number of nearest neighbours (excluding self).
    :type k: int
    :return: Entropy array, shape (n_cells,).
    :rtype: np.ndarray
    """
    from scipy.spatial import KDTree
    tree     = KDTree(emb)
    _, idx   = tree.query(emb, k=k + 1)
    idx      = idx[:, 1:]
    uniq     = np.unique(batch_labels)
    b2i      = {b: i for i, b in enumerate(uniq)}
    bi       = np.array([b2i[b] for b in batch_labels])
    ents     = np.zeros(len(batch_labels))
    for i, nbrs in enumerate(idx):
        counts = np.bincount(bi[nbrs], minlength=len(uniq)).astype(float)
        counts /= counts.sum() + 1e-10
        ents[i] = scipy_entropy(counts + 1e-10)
    return ents


# ── 5. HBB expression analysis ──────────────────────────────────────────────────

def hbb_expression_analysis(toc_raw, toc_corrected, cell_types, gene_names,
                             hbb_genes=None, erythroid_labels=None):
    """
    Measure removal of haemoglobin gene signal from non-erythroid cells.

    HBB and HBA genes are massively expressed in erythroid cells and leak into
    all cell types as ambient RNA.  After correction, these genes should be
    present in far fewer non-erythroid cells while remaining high in erythroid.

    Parameters
    ----------
    toc_raw, toc_corrected : sparse (n_genes, n_cells)
    cell_types      : array-like str, length n_cells
    gene_names      : array-like str, length n_genes
    hbb_genes       : list of gene names.  Default: ['HBB','HBA2','HBA1','HBD']
    erythroid_labels: set of cell type labels considered erythroid.
                      Default: any label containing 'erythr', 'rbc', 'red blood'

    Returns
    -------
    dict
        per_gene                 DataFrame: pct_noneryth before/after + mean expr
        mean_pct_noneryth_before/after  % non-erythroid cells with HBB > 0
        mean_pct_reduction       before − after (positive = signal removed)
        mean_erythroid_before/after   HBB in erythroid cells (should stay high)
        hbb_signal_reduced       bool
    """
    if hbb_genes is None:
        hbb_genes = ['HBB', 'HBA2', 'HBA1', 'HBD', 'HBG1', 'HBG2']

    gene_names = np.asarray(gene_names)
    cell_types = np.asarray(cell_types)

    if erythroid_labels is None:
        ery = np.array([
            any(k in ct.lower() for k in ('erythr', 'rbc', 'red blood'))
            for ct in cell_types
        ])
    else:
        ery = np.isin(cell_types, list(erythroid_labels))

    non_ery = ~ery
    if not non_ery.any():
        raise ValueError("No non-erythroid cells found. Check cell_types/erythroid_labels.")

    present = [g for g in hbb_genes if g in gene_names]
    if not present:
        raise ValueError(f"None of hbb_genes {hbb_genes} found in gene_names.")

    raw_csc = sparse.csc_matrix(toc_raw)
    cor_csc = sparse.csc_matrix(toc_corrected)

    rows = []
    for gene in present:
        gi  = int(np.where(gene_names == gene)[0][0])
        rn  = np.asarray(raw_csc[gi, non_ery].todense()).flatten()
        cn  = np.asarray(cor_csc[gi, non_ery].todense()).flatten()
        pct_b = float((rn > 0).mean()) * 100
        pct_a = float((cn > 0).mean()) * 100
        re = np.asarray(raw_csc[gi, ery].todense()).flatten() if ery.any() else np.array([np.nan])
        ce = np.asarray(cor_csc[gi, ery].todense()).flatten() if ery.any() else np.array([np.nan])
        rows.append({
            'gene':                  gene,
            'pct_noneryth_before':   pct_b,
            'pct_noneryth_after':    pct_a,
            'pct_reduction':         pct_b - pct_a,
            'mean_noneryth_before':  float(rn.mean()),
            'mean_noneryth_after':   float(cn.mean()),
            'mean_erythroid_before': float(np.nanmean(re)),
            'mean_erythroid_after':  float(np.nanmean(ce)),
        })

    df = pd.DataFrame(rows)
    return {
        'per_gene':                 df,
        'mean_pct_noneryth_before': float(df['pct_noneryth_before'].mean()),
        'mean_pct_noneryth_after':  float(df['pct_noneryth_after'].mean()),
        'mean_pct_reduction':       float(df['pct_reduction'].mean()),
        'mean_erythroid_before':    float(df['mean_erythroid_before'].mean()),
        'mean_erythroid_after':     float(df['mean_erythroid_after'].mean()),
        'hbb_signal_reduced':       bool(df['pct_reduction'].mean() > 0),
    }


# ── 6. Cluster silhouette score ─────────────────────────────────────────────────

def cluster_silhouette(toc_raw, toc_corrected, clusters, n_pcs=15,
                        max_cells=2000, seed=42):
    """
    Measure cluster separation quality via silhouette score before/after correction.

    Ambient RNA blurs cell type boundaries.  After removal, biologically
    distinct clusters should be tighter and more separated in PCA space,
    producing a higher silhouette score.

    Subsamples to max_cells for computational tractability on large datasets.

    Parameters
    ----------
    toc_raw, toc_corrected : sparse (n_genes, n_cells)
    clusters   : array-like cluster labels, length n_cells
    n_pcs      : int, PCA components (default 15)
    max_cells  : int, subsample cap (default 2000)
    seed       : int

    Returns
    -------
    dict
        sil_before, sil_after   silhouette scores in [-1, 1]
        sil_delta               after - before  (positive = improved)
        improved                bool
    """
    try:
        from sklearn.metrics import silhouette_score as _sil_fn
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    clusters = np.asarray(clusters)
    n_cells  = len(clusters)
    unique_cls = np.unique(clusters)

    if len(unique_cls) < 2:
        raise ValueError("Need >= 2 clusters for silhouette score.")
    if len(unique_cls) >= n_cells:
        raise ValueError("More clusters than cells; silhouette undefined.")

    rng = np.random.default_rng(seed)
    if n_cells > max_cells:
        idx = rng.choice(n_cells, size=max_cells, replace=False)
        clusters      = clusters[idx]
        toc_raw       = sparse.csc_matrix(toc_raw)[:, idx]
        toc_corrected = sparse.csc_matrix(toc_corrected)[:, idx]

    emb_b = _pca(toc_raw,       n_pcs)
    emb_a = _pca(toc_corrected, n_pcs)

    sil_b = float(_sil_fn(emb_b, clusters, metric='euclidean'))
    sil_a = float(_sil_fn(emb_a, clusters, metric='euclidean'))

    return {
        'sil_before': sil_b,
        'sil_after':  sil_a,
        'sil_delta':  sil_a - sil_b,
        'improved':   bool(sil_a > sil_b),
    }


# ── 7. Spurious DE gene reduction ───────────────────────────────────────────────

def spurious_de_reduction(toc_raw, toc_corrected, clusters, gene_names,
                           log2fc_threshold=1.0, min_expr_frac=0.1):
    """
    Count cluster-vs-rest DE genes that disappear after correction.

    Ambient RNA inflates off-target gene expression, creating false
    differential expression between clusters.  After correction, genes
    driven purely by contamination should drop below the log2FC threshold.

    A gene is called DE if |log2FC(cluster vs rest)| > log2fc_threshold
    AND expressed in > min_expr_frac of target cluster cells.

    n_spurious = max(0, n_de_raw - n_de_corrected): genes the correction
    silenced.  Higher n_spurious = more contamination artifacts removed.

    Parameters
    ----------
    toc_raw, toc_corrected : sparse (n_genes, n_cells)
    clusters           : array-like cluster labels, length n_cells
    gene_names         : array-like str, length n_genes
    log2fc_threshold   : float (default 1.0 = 2-fold)
    min_expr_frac      : float (default 0.1 = expressed in >= 10% of cells)

    Returns
    -------
    dict
        n_de_raw, n_de_corrected   DE gene counts (summed over all clusters)
        n_spurious                 max(0, raw - corrected)
        pct_spurious               n_spurious / n_de_raw * 100
        improved                   bool (fewer DE genes after)
    """
    clusters   = np.asarray(clusters)
    gene_names = np.asarray(gene_names)
    unique_cls = np.unique(clusters)

    if len(unique_cls) < 2:
        raise ValueError("Need >= 2 clusters for spurious DE reduction.")

    raw_cpm = _cpm(toc_raw)
    cor_cpm = _cpm(toc_corrected)

    n_de_raw = 0
    n_de_cor = 0

    for cl in unique_cls:
        cl_idx   = np.where(clusters == cl)[0]
        rest_idx = np.where(clusters != cl)[0]
        if len(cl_idx) < 3 or len(rest_idx) < 3:
            continue

        for mat, is_raw in [(raw_cpm, True), (cor_cpm, False)]:
            mean_cl   = np.asarray(mat[:, cl_idx].mean(axis=1)).flatten()
            mean_rest = np.asarray(mat[:, rest_idx].mean(axis=1)).flatten()

            sub_cl    = sparse.csc_matrix(mat[:, cl_idx])
            expr_frac = np.asarray((sub_cl != 0).mean(axis=1)).flatten()

            log2fc  = np.log2(mean_cl + 1.0) - np.log2(mean_rest + 1.0)
            de_mask = (np.abs(log2fc) > log2fc_threshold) & (expr_frac > min_expr_frac)

            if is_raw:
                n_de_raw += int(de_mask.sum())
            else:
                n_de_cor += int(de_mask.sum())

    n_spurious = max(0, n_de_raw - n_de_cor)
    return {
        'n_de_raw':       n_de_raw,
        'n_de_corrected': n_de_cor,
        'n_spurious':     n_spurious,
        'pct_spurious':   float(n_spurious) / (n_de_raw + 1e-10) * 100.0,
        'improved':       bool(n_de_cor < n_de_raw),
    }


# ── 8. Marker enrichment score ──────────────────────────────────────────────────

def marker_enrichment_score(toc_raw, toc_corrected, clusters, gene_names,
                             marker_genes):
    """
    Measure how well known marker genes rank among top DE genes per cluster.

    For each cluster, genes are ranked by log2FC (cluster vs rest).  The
    percentile rank of each known marker is recorded (1.0 = top-ranked,
    0.0 = bottom).  After correction, contamination-inflated non-target
    expression drops, so markers should rank higher in their target cluster.

    Parameters
    ----------
    toc_raw, toc_corrected : sparse (n_genes, n_cells)
    clusters     : array-like cluster labels, length n_cells
    gene_names   : array-like str, length n_genes
    marker_genes : dict {cluster_label: [gene_name, ...]}
                   OR list of gene names (auto-assigned to highest cluster)

    Returns
    -------
    dict
        mean_rank_before, mean_rank_after   mean percentile rank in [0, 1]
        rank_delta                          after - before (positive = improved)
        improved                            bool
    """
    gene_names = np.asarray(gene_names)
    clusters   = np.asarray(clusters)
    gene_idx   = {g: i for i, g in enumerate(gene_names)}
    n_genes    = len(gene_names)

    raw_cpm = _cpm(toc_raw)
    cor_cpm = _cpm(toc_corrected)

    marker_dict = _to_marker_dict(marker_genes, clusters, gene_names, raw_cpm)

    ranks_b, ranks_a = [], []

    for cl, genes in marker_dict.items():
        cl_idx   = np.where(clusters == cl)[0]
        rest_idx = np.where(clusters != cl)[0]
        if len(cl_idx) < 3 or len(rest_idx) < 3 or not genes:
            continue

        for mat, ranks_list in [(raw_cpm, ranks_b), (cor_cpm, ranks_a)]:
            mean_cl   = np.asarray(mat[:, cl_idx].mean(axis=1)).flatten()
            mean_rest = np.asarray(mat[:, rest_idx].mean(axis=1)).flatten()
            fc        = np.log2(mean_cl + 1.0) - np.log2(mean_rest + 1.0)

            order    = np.argsort(fc)[::-1]
            rank_of  = {int(g_idx): pos for pos, g_idx in enumerate(order)}

            for gene in genes:
                gi = gene_idx.get(gene)
                if gi is None:
                    warnings.warn(f"marker gene '{gene}' not found", stacklevel=2)
                    continue
                pct_rank = 1.0 - rank_of[gi] / n_genes
                ranks_list.append(pct_rank)

    if not ranks_b or not ranks_a:
        raise ValueError("No marker genes matched gene_names.")

    mb = float(np.mean(ranks_b))
    ma = float(np.mean(ranks_a))
    return {
        'mean_rank_before': mb,
        'mean_rank_after':  ma,
        'rank_delta':       ma - mb,
        'improved':         bool(ma > mb),
    }
