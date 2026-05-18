---
sidebar_position: 7
---

# Downstream Analysis

The `downstream` module provides a complete post-correction analysis pipeline: normalization → PCA → UMAP/tSNE → Leiden/k-means clustering → differential expression.

## Requirements

```bash
pip install -e ".[downstream]"
# installs: scikit-learn, umap-learn, leidenalg, python-igraph
```

## Full pipeline in one call

```python
from SoupX import load_10x, set_clusters, auto_est_cont, adjust_counts
from SoupX.downstream import run_downstream, plot_embedding

sc        = auto_est_cont(load_10x('path/to/cellranger/outs/'))
corrected = adjust_counts(sc)

result = run_downstream(
    corrected,
    gene_names = sc.genes.tolist(),
    n_pcs      = 50,
    n_hvg      = 2000,
    n_topics   = 15,       # Leiden resolution
    method     = 'leiden', # or 'kmeans'
    run_umap   = True,
    run_tsne   = False,
)

# result keys: 'embedding', 'cluster_labels', 'de_results', 'pca'
plot_embedding(result['embedding'], result['cluster_labels'], title='UMAP - corrected')
```

## Step-by-step

### Normalization

```python
from SoupX.downstream import normalize_log1p

# Input: genes × cells sparse matrix
# Output: cells × genes dense float64 array
normed = normalize_log1p(corrected, target_sum=1e4)
```

### PCA

```python
from SoupX.downstream import run_pca

pca = run_pca(
    corrected,
    gene_names   = sc.genes.tolist(),
    n_components = 50,
    n_top_genes  = 2000,   # HVG selection by variance
)
# pca['embedding']       (n_cells × n_components)
# pca['variance_ratio']  (n_components,)
```

### UMAP

```python
from SoupX.downstream import run_umap

umap_coords = run_umap(pca, n_neighbors=15, min_dist=0.1)
# (n_cells × 2)
```

### Clustering

```python
from SoupX.downstream import cluster_leiden, cluster_kmeans

# Leiden (requires leidenalg + python-igraph)
labels = cluster_leiden(pca, resolution=0.5, n_neighbors=15)

# k-means (requires scikit-learn only)
labels = cluster_kmeans(pca, n_clusters=10)
```

### Differential expression

```python
from SoupX.downstream import differential_expression

de = differential_expression(
    corrected,
    gene_names     = sc.genes.tolist(),
    cluster_labels = labels,
    min_cells      = 5,
    top_n          = 20,
    log2fc_thresh  = 0.25,
)
# de is a pd.DataFrame: cluster, gene, statistic, pvalue, log2fc, rank
```

### Cell-type scoring

```python
from SoupX.downstream import score_cell_types

marker_dict = {
    'T_cell':  ['CD3D', 'CD3E', 'CD3G'],
    'B_cell':  ['CD19', 'MS4A1', 'CD79A'],
    'NK_cell': ['GNLY', 'NKG7', 'KLRD1'],
}

scores = score_cell_types(corrected, sc.genes.tolist(), marker_dict)
# scores is pd.DataFrame (cells × cell_types)
```
