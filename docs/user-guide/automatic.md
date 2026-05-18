# Automatic Workflow

The automatic workflow is recommended for most datasets. It uses TF-IDF marker gene detection combined with Bayesian posterior aggregation to estimate the contamination fraction ρ without requiring prior biological knowledge.

## Overview

```
load_10x() → set_clusters() → auto_est_cont() → adjust_counts()
```

## Minimal example

```python
from SoupX import load_10x, set_clusters, auto_est_cont, adjust_counts

sc        = load_10x('path/to/cellranger/outs/')
sc        = set_clusters(sc, cluster_labels)
sc        = auto_est_cont(sc)
corrected = adjust_counts(sc)
```

## Step-by-step

### 1. Load data

```python
from SoupX import load_10x

sc = load_10x(
    data_dir    = 'path/to/cellranger/outs/',
    verbose     = True,    # print loading progress
)
print(sc)
# SoupChannel with 33538 genes and 10209 cells
```

`load_10x` automatically detects cellranger v2 and v3 layouts and loads cluster/tSNE/UMAP projections from the `analysis/` directory when present.

### 2. Add cluster labels

```python
from SoupX import set_clusters

# From a pandas Series indexed by cell barcode:
sc = set_clusters(sc, cluster_series)

# From an array (order must match sc.cells):
sc = set_clusters(sc, cluster_array)
```

Clusters from Seurat, Scanpy, or any other tool work. The contamination estimate is more stable with more clusters.

### 3. Estimate contamination

```python
from SoupX import auto_est_cont

sc = auto_est_cont(
    sc,
    # --- marker selection ---
    tfidf_min         = 1.0,   # minimum TF-IDF score
    soup_quantile     = 0.90,  # minimum soup expression quantile
    max_markers       = 100,   # max marker genes to use
    # --- Bayesian prior ---
    prior_rho         = 0.05,  # prior mode contamination
    prior_rho_std_dev = 0.10,  # prior standard deviation
    # --- search bounds ---
    contamination_range = (0.01, 0.80),
    # --- per-cell refinement ---
    cell_rho_method   = None,  # None | 'empirical_bayes' | 'glm' | 'decontx'
    verbose           = True,
    do_plot           = True,  # show posterior density plot
)

print(f"rho = {sc.meta_data['rho'].mean():.3f}")
```

The posterior density plot shows the prior (dashed) and posterior (solid) distributions over ρ, with the MAP estimate (red vertical line).

### 4. Remove contamination

```python
from SoupX import adjust_counts

corrected = adjust_counts(
    sc,
    method       = 'subtraction',  # 'subtraction' | 'multinomial' | 'soupOnly'
    round_to_int = False,           # stochastically round to integers
    verbose      = 1,
)
# Returns scipy.sparse.csc_matrix, same shape as sc.toc
```

## Per-cell rho refinement

After estimating a global rho, you can refine to per-cell estimates:

=== "Empirical Bayes"

    Gamma-Poisson conjugate model. Fast, requires soup marker genes.

    ```python
    sc = auto_est_cont(sc, cell_rho_method='empirical_bayes')
    ```

=== "GLM"

    Poisson GLM with log(nUMI) covariate. Captures the nUMI → rho relationship.

    ```python
    sc = auto_est_cont(sc, cell_rho_method='glm')
    ```

=== "DecontX EM"

    Full Dirichlet-Multinomial EM (no topics). More accurate but slower.

    ```python
    sc = auto_est_cont(sc, cell_rho_method='decontx')
    ```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "No plausible marker genes found" | TF-IDF or soup filter too strict | Reduce `tfidf_min` or `soup_quantile` |
| MAP rho at boundary | Contamination outside search range | Adjust `contamination_range` |
| `ValueError: Clustering must be set` | Missing clusters | Call `set_clusters()` first |
| rho > 0.5 warning | Very high contamination or estimation error | Check soup profile; use `force_accept=True` if expected |
