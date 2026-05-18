---
sidebar_position: 6
---

# Iterative Refinement

The iterative workflow runs `auto_est_cont` → `adjust_counts` → soup profile update in a loop until ρ converges. After the first correction round, genes that remain highly expressed in corrected cells are likely genuinely cellular. Down-weighting these genes in the soup profile and re-running estimation converges to a more accurate ρ.

## When to use

- Datasets where ambient RNA overlaps substantially with cellular expression (e.g. PBMC, fetal liver, tumour microenvironment)
- The initial automatic estimate seems too low (marker genes still appear cross-cluster after correction)

:::tip Benchmark result
`upg-iterative` achieves the **highest cluster ARI** across all benchmark datasets, outperforming all other pipelines on cluster preservation.
:::

## Usage

```python
from SoupX import load_10x, set_clusters, iterative_auto_est_cont, adjust_counts

sc = load_10x('path/to/cellranger/outs/')
sc = set_clusters(sc, cluster_labels)

sc = iterative_auto_est_cont(
    sc,
    n_iter        = 3,      # refinement iterations (1–3 is typical)
    shrink_factor = 5.0,    # aggressiveness of soup profile update
    tol           = 1e-3,   # convergence threshold on mean |Δrho|
    do_plot       = False,
)

corrected = adjust_counts(sc)
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_iter` | 2 | Number of refinement iterations |
| `shrink_factor` | 5.0 | Higher = more aggressive suppression of cellular genes |
| `tol` | 1e-3 | Mean absolute change in rho to declare convergence |
| `**aec_kwargs` | - | Forwarded to `auto_est_cont` |

## How the soup profile update works

Each iteration:

1. Run `auto_est_cont` on current soup profile → ρ
2. `adjust_counts` → corrected matrix
3. Compute the ratio `cell_share_g / soup_share_g` for each gene in the corrected matrix
4. Apply weight: `weight_g = max(1 / (1 + shrink_factor × ratio_g), 0.3)` - no gene drops below 30% of its original weight
5. Renormalize soup profile
6. Repeat from step 1 until `mean |Δrho| < tol`
