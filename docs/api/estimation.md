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

:param sc: SoupChannel with clusters set.
:type sc: SoupChannel

:param tfidf_min: Minimum TF-IDF score for marker gene selection.
:type tfidf_min: float

:param soup_quantile: Minimum soup expression quantile (0–1) for marker genes.
:type soup_quantile: float

:param max_markers: Maximum number of marker genes to use.
:type max_markers: int

:param contamination_range: (min_rho, max_rho) search bounds.
:type contamination_range: tuple

:param prior_rho: Mode of the Gamma prior on rho.
:type prior_rho: float

:param prior_rho_std_dev: Standard deviation of the Gamma prior.
:type prior_rho_std_dev: float

:param cell_rho_method: Per-cell refinement method after global MAP estimation.
:type cell_rho_method: str or None

:param do_plot: Show posterior density plot.
:type do_plot: bool

:param verbose: Print progress.
:type verbose: bool

:return: SoupChannel with rho set in meta_data and sc.fit populated.
:rtype: SoupChannel

---

## `calculate_contamination_fraction`

Estimate contamination fraction using known non-expressing gene sets (Poisson GLM).

```python
from SoupX import calculate_contamination_fraction

sc = calculate_contamination_fraction(
    sc,
    non_expressed_gene_list = {'HB': ['HBB', 'HBA2']},
    use_to_est              = use_to_est,   # from estimate_non_expressing_cells
    cell_rho_method         = None,
    verbose                 = True,
)
```

:param sc: SoupChannel with soup_profile set.
:type sc: SoupChannel

:param non_expressed_gene_list: Dict mapping set names to gene lists.
:type non_expressed_gene_list: dict

:param use_to_est: Boolean DataFrame (cells × gene_sets) from estimate_non_expressing_cells.
:type use_to_est: pd.DataFrame

:param cell_rho_method: Per-cell refinement method. See auto_est_cont for options.
:type cell_rho_method: str or None

:return: SoupChannel with rho, rhoLow, rhoHigh in meta_data.
:rtype: SoupChannel

---

## `estimate_non_expressing_cells`

Identify cells/clusters that genuinely do not express each gene set (Poisson FDR test).

:param sc: SoupChannel.
:type sc: SoupChannel

:param non_expressed_gene_list: Dict mapping set names to gene lists.
:type non_expressed_gene_list: dict

:param clusters: Cell-to-cluster mapping. None = use sc.meta_data['clusters']; False = per-cell.
:type clusters: pd.Series or None or False

:param maximum_contamination: Upper bound on expected contamination fraction.
:type maximum_contamination: float

:param fdr: FDR threshold for the Poisson test.
:type fdr: float

:return: Boolean DataFrame (cells × gene_sets). True = safe for estimation.
:rtype: pd.DataFrame

---

## `estimate_cell_rho`

Refine contamination to per-cell estimates via Gamma-Poisson empirical Bayes shrinkage.

:param sc: SoupChannel with rho already set.
:type sc: SoupChannel

:param soup_quantile: Percentile cutoff for soup marker gene selection.
:type soup_quantile: float

:param prior_rho: Prior mean contamination. If None, uses mean of current rho.
:type prior_rho: float, optional

:param prior_std: Prior standard deviation (smaller = stronger shrinkage).
:type prior_std: float

:return: SoupChannel with per-cell rho in meta_data.
:rtype: SoupChannel

---

## `estimate_decontx_rho`

Per-cell contamination via DecontX-style Dirichlet-Multinomial EM (no LDA topics).

:param sc: SoupChannel with soup_profile set.
:type sc: SoupChannel

:param prior_rho: Initial contamination guess for all cells.
:type prior_rho: float, optional

:param n_iter: Maximum EM iterations.
:type n_iter: int

:param tol: Parameter-delta convergence threshold.
:type tol: float

:param verbose: Print convergence progress.
:type verbose: bool

:return: SoupChannel with per-cell rho in meta_data.
:rtype: SoupChannel

---

## `iterative_auto_est_cont`

Iteratively refine the soup profile and re-estimate contamination until convergence.

:param sc: SoupChannel with soup_profile and clusters set.
:type sc: SoupChannel

:param n_iter: Number of refinement iterations.
:type n_iter: int

:param shrink_factor: Controls aggressiveness of soup profile update.
:type shrink_factor: float

:param tol: Mean absolute change in rho for convergence.
:type tol: float

:return: SoupChannel with refined rho and soup_profile.
:rtype: SoupChannel
