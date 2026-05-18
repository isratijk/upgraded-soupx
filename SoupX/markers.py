import numpy as np
import pandas as pd
import scipy.sparse
from scipy.stats import hypergeom


def quick_markers(toc, clusters, genes=None, n=10, fdr=0.01, express_cut=0.9):
    """
    Find top marker genes for each cluster using TF-IDF ranking.

    Port of R SoupX quickMarkers(). Binarises expression (count > express_cut),
    scores each gene per cluster with TF-IDF, and returns the top-N markers
    per cluster that pass a hypergeometric FDR cut-off.

    Parameters
    ----------
    toc : scipy.sparse matrix
        (n_genes x n_cells) count matrix.
    clusters : array-like
        Cluster label for each cell (length n_cells).
    genes : array-like, optional
        Gene names (length n_genes). If None, integer indices are used.
    n : int
        Maximum markers per cluster to return.
    fdr : float
        FDR threshold for hypergeometric test.
    express_cut : float
        Counts strictly above this value are treated as "expressed".

    Returns
    -------
    pd.DataFrame
        Columns: gene, cluster, geneFrequency, geneFrequencyOutsideCluster,
        geneFrequencySecondBest, geneFrequencyGlobal, secondBestClusterName,
        tfidf, idf, qval.
    """
    from statsmodels.stats.multitest import multipletests

    clusters = np.asarray(clusters, dtype=str)
    unique_clusters = sorted(np.unique(clusters))
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}

    coo = toc.tocoo()
    # Binarise: entries above express_cut are "expressed"
    expr_mask = coo.data > express_cut
    expr_rows = coo.row[expr_mask]   # gene indices
    expr_cols = coo.col[expr_mask]   # cell indices

    n_genes = toc.shape[0]
    n_cells = toc.shape[1]

    # nObs[gene, cluster] = number of cells in cluster expressing gene
    nObs = np.zeros((n_genes, n_clusters), dtype=int)
    for g, c_idx in zip(expr_rows, clusters[expr_cols]):
        nObs[g, cluster_to_idx[c_idx]] += 1

    # Cluster sizes
    cl_cnts = np.array([np.sum(clusters == c) for c in unique_clusters], dtype=int)

    # Gene-level totals and frequencies
    nTot = nObs.sum(axis=1)                         # total cells expressing each gene

    # TF-IDF
    with np.errstate(divide='ignore', invalid='ignore'):
        tf = nObs / np.maximum(cl_cnts[np.newaxis, :], 1)
        ntf = (nTot[:, np.newaxis] - nObs) / np.maximum(
            (n_cells - cl_cnts)[np.newaxis, :], 1
        )
        idf = np.where(nTot > 0, np.log(n_cells / nTot), 0.0)
    score = tf * idf[:, np.newaxis]

    # Hypergeometric q-values per cluster
    qvals = np.ones((n_genes, n_clusters))
    for e, cl in enumerate(unique_clusters):
        if cl_cnts[e] == 0:
            continue  # empty cluster: no cells, all qvals stay 1
        # phyper(nObs[:,e]-1, nTot, n_cells-nTot, cl_cnts[e], lower.tail=FALSE)
        # = P(X >= nObs[:,e]) for X ~ Hypergeom(N=n_cells, K=nTot, n=cl_cnts[e])
        pvals = hypergeom.sf(nObs[:, e] - 1, n_cells, nTot, cl_cnts[e])
        pvals = np.nan_to_num(pvals, nan=1.0)
        # Only run multipletests when there are variable p-values
        if pvals.min() < 1.0:
            qvals[:, e] = multipletests(pvals, method='fdr_bh')[1]

    # Second-best cluster frequency — vectorized: sort once, read top-2 per gene.
    # Avoids O(n_clusters²×n_genes) np.delete copies.
    snd_best = np.zeros_like(tf)
    snd_best_name = np.full((n_genes, n_clusters), '', dtype=object)
    if n_clusters > 1:
        # top2_idx[g, 0/1] = cluster indices with 1st/2nd highest tf for gene g
        top2_idx = np.argsort(-tf, axis=1)[:, :2]          # (n_genes, 2)
        top2_val = tf[np.arange(n_genes)[:, None], top2_idx]  # (n_genes, 2)
        cl_names = np.asarray(unique_clusters)
        for e in range(n_clusters):
            # second-best = top2[0] if top2[0] != e, else top2[1]
            use_idx = np.where(top2_idx[:, 0] != e, 0, 1)   # (n_genes,)
            best_other = top2_idx[np.arange(n_genes), use_idx]
            snd_best[:, e] = top2_val[np.arange(n_genes), use_idx]
            snd_best_name[:, e] = cl_names[best_other]

    gene_labels = np.asarray(genes) if genes is not None else np.arange(n_genes)

    rows = []
    for e, cl in enumerate(unique_clusters):
        if cl_cnts[e] == 0:
            continue  # empty cluster produces only NaN TF-IDF; skip
        # Sort by descending TF-IDF score
        o = np.argsort(-score[:, e])
        n_sig = int(np.sum(qvals[:, e] < fdr))
        if n_sig >= n:
            keep = o[:n]
        else:
            keep = o[qvals[o, e] < fdr]

        for g in keep:
            rows.append({
                'gene': gene_labels[g],
                'cluster': cl,
                'geneFrequency': tf[g, e],
                'geneFrequencyOutsideCluster': ntf[g, e],
                'geneFrequencySecondBest': snd_best[g, e],
                'geneFrequencyGlobal': nTot[g] / n_cells,
                'secondBestClusterName': snd_best_name[g, e],
                'tfidf': score[g, e],
                'idf': idf[g],
                'qval': qvals[g, e],
            })

    return pd.DataFrame(rows)
