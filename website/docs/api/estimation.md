---
sidebar_position: 3
---

# Estimation Functions

## `auto_est_cont`

Automatically estimate the contamination fraction using TF-IDF markers and Bayesian aggregation.

```python
from SoupX import auto_est_cont

sc = auto_est_cont(
    sc,
    tfidf_min           = 1.0,
    soup_quantile       = 0.90,
    max_markers         = 100,
    contamination_range = (0.01, 0.80),
    prior_rho           = 0.05,
    prior_rho_std_dev   = 0.10,
    cell_rho_method     = None,   # None | 'empirical_bayes' | 'glm' | 'decontx'
    do_plot             = True,
    verbose             = True,
)
```

**Parameters:**

- `sc` - `SoupChannel` with clusters set
- `tfidf_min` - Minimum TF-IDF score for marker gene selection
- `soup_quantile` - Minimum soup expression quantile (0–1) for marker genes
- `max_markers` - Maximum number of marker genes to use
- `contamination_range` - `(min_rho, max_rho)` search bounds
- `prior_rho` - Mode of the Gamma prior on rho
- `prior_rho_std_dev` - Standard deviation of the Gamma prior
- `cell_rho_method` - Per-cell refinement method after global MAP estimation
- `do_plot` - Show posterior density plot

**Returns:** `SoupChannel` with `rho` set in `meta_data` and `sc.fit` populated

---

## `calculate_contamination_fraction`

Estimate contamination fraction using known non-expressing gene sets (Poisson GLM).

```python
from SoupX import calculate_contamination_fraction

sc = calculate_contamination_fraction(
    sc,
    non_expressed_gene_list = {'HB': ['HBB', 'HBA2']},
    use_to_est              = use_to_est,
    cell_rho_method         = None,
    verbose                 = True,
)
```

**Returns:** `SoupChannel` with `rho`, `rhoLow`, `rhoHigh` in `meta_data`

---

## `estimate_non_expressing_cells`

Identify cells/clusters that genuinely do not express each gene set (Poisson FDR test).

**Parameters:**

- `sc` - `SoupChannel`
- `non_expressed_gene_list` - Dict mapping set names to gene lists
- `clusters` - Cell-to-cluster mapping. `None` = use `sc.meta_data['clusters']`; `False` = per-cell
- `maximum_contamination` - Upper bound on expected contamination fraction
- `fdr` - FDR threshold for the Poisson test

**Returns:** Boolean `pd.DataFrame` (cells × gene_sets). `True` = safe for estimation

---

## `estimate_cell_rho`

Refine contamination to per-cell estimates via Gamma-Poisson empirical Bayes shrinkage.

**Parameters:**

- `sc` - `SoupChannel` with `rho` already set
- `soup_quantile` - Percentile cutoff for soup marker gene selection
- `prior_rho` - Prior mean contamination. If None, uses mean of current rho
- `prior_std` - Prior standard deviation (smaller = stronger shrinkage)

**Returns:** `SoupChannel` with per-cell `rho` in `meta_data`

---

## `estimate_decontx_rho`

Per-cell contamination via DecontX-style Dirichlet-Multinomial EM (no LDA topics).

**Parameters:**

- `sc` - `SoupChannel` with `soup_profile` set
- `prior_rho` - Initial contamination guess for all cells
- `n_iter` - Maximum EM iterations
- `tol` - Parameter-delta convergence threshold

**Returns:** `SoupChannel` with per-cell `rho` in `meta_data`

---

## `iterative_auto_est_cont`

Iteratively refine the soup profile and re-estimate contamination until convergence.

**Parameters:**

- `sc` - `SoupChannel` with `soup_profile` and clusters set
- `n_iter` - Number of refinement iterations
- `shrink_factor` - Controls aggressiveness of soup profile update
- `tol` - Mean absolute change in rho for convergence

**Returns:** `SoupChannel` with refined `rho` and `soup_profile`
