---
sidebar_position: 2
---

# Manual Workflow

Use the manual workflow when you have prior biological knowledge about which genes should not be expressed in certain cell populations. The canonical example is haemoglobin genes (HBB, HBA1, HBA2) in non-erythroid cells.

## When to use

- You know a gene set is biologically absent in a specific cell population
- The automatic workflow produces unreliable estimates (too few markers, extreme rho)
- You want to use a specific set of "contamination marker" genes for calibration

## Example: haemoglobin genes in PBMC

```python
from SoupX import (
    load_10x, set_clusters,
    estimate_non_expressing_cells,
    calculate_contamination_fraction,
    adjust_counts,
)

sc = load_10x('path/to/cellranger/outs/')
sc = set_clusters(sc, cluster_labels)

# Define gene sets that are absent in some clusters
gene_list = {
    'HB':   ['HBB', 'HBA2', 'HBA1', 'HBD', 'HBG1', 'HBG2'],
    'IGKC': ['IGKC'],   # immunoglobulin light chain (T-cell negative)
}

# Identify clusters safe to use for calibration (Poisson FDR test)
use_to_est = estimate_non_expressing_cells(
    sc,
    non_expressed_gene_list = gene_list,
    maximum_contamination   = 0.2,
    fdr                     = 0.05,
)
# Returns DataFrame (cells × gene_sets), True = safe to use

# Fit Poisson GLM and set rho
sc = calculate_contamination_fraction(
    sc,
    non_expressed_gene_list = gene_list,
    use_to_est              = use_to_est,
    verbose                 = True,
)

corrected = adjust_counts(sc)
```

## What `estimate_non_expressing_cells` does

For each gene set and each cluster, the function tests whether any cell in the cluster shows counts significantly higher than expected from soup alone (Poisson test). Clusters where no cell passes the test are marked as **non-expressing** and can be used for calibration.

The test is conservative by design: if any cell in a cluster shows significant expression of a gene set, the entire cluster is excluded for that gene set.

## Combining manual and per-cell refinement

```python
sc = calculate_contamination_fraction(
    sc, gene_list, use_to_est,
    cell_rho_method = 'empirical_bayes',  # or 'glm', 'decontx'
)
```

## Tips

- Use gene sets specific to one cell type (not ubiquitously expressed)
- More gene sets → more calibration data → more stable estimate
- If `estimate_non_expressing_cells` returns no passing clusters, try:
  - Setting `clusters=False` (treat each cell independently)
  - Increasing `maximum_contamination`
  - Increasing `fdr`
