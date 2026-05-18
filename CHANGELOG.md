# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- MkDocs-based static documentation site
- `.env.example` for environment configuration
- `requirements.txt` for reproducible installs
- GitHub issue templates and pull request template
- Contributing guidelines

---

## [1.6.0.python] — 2025-01-01

### Added
- **Full Python port** of the R SoupX package (Young & Behjati, 2020) — no R dependency
- `SoupChannel` container class with AnnData/pickle interoperability
- HDF5 input support via `load_10x_h5` / `read_10x_h5` (5–10× faster than MEX format)
- **DecontX** per-cell decontamination: two-component Dirichlet-Multinomial EM with LDA topics (`run_decontx`, `select_n_topics`)
- Per-cell rho refinement via empirical Bayes (`estimate_cell_rho`) and DecontX EM (`estimate_decontx_rho`)
- **Doublet-aware estimation**: Scrublet-style doublet scoring integrated into contamination estimation (`estimate_doublet_scores`, `auto_est_cont_doublet_aware`)
- **Gene-heterogeneity correction**: amplify truly ambient genes before EM (`compute_gene_enrichment`, `reweight_soup_profile`, `run_decontx_genehet`)
- **Iterative refinement loop** (`iterative_auto_est_cont`): auto_est_cont → adjust_counts → soup profile update until convergence
- **Downstream analysis pipeline** (`run_downstream`): normalization → PCA → UMAP/tSNE → Leiden/k-means clustering → one-vs-rest Wilcoxon DE
- **Eight quantitative benchmark metrics**:
  1. `cross_species_reduction` — barnyard experiment contamination fold-change
  2. `marker_fold_change` — cell-type marker specificity
  3. `cluster_membership_delta` — artificial cluster dissolution
  4. `batch_entropy` — local neighbourhood batch-mixing
  5. `hbb_expression_analysis` — HBB removal in non-erythroid cells
  6. `cluster_silhouette` — post-correction cluster coherence
  7. `spurious_de_reduction` — spurious DE gene reduction
  8. `marker_enrichment_score` — known marker enrichment post-correction
- Three soup profile estimation methods: `fixed`, `statistical`, `emptydrops`
- Three `adjust_counts` methods: `subtraction` (default), `multinomial`, `soupOnly`
- Three per-cell rho methods: `empirical_bayes`, `glm`, `decontx`
- Visualization: `plot_soup_correlation`, `plot_marker_distribution`, `plot_marker_map`, `plot_change_map`
- Full test suite: 16 test modules, regression golden baseline

### Changed
- `auto_est_cont`: Bayesian posterior uses proper joint log-posterior (product of Poisson likelihoods × Gamma prior) instead of the original mixture-density approach
- Cluster-level `adjust_counts` uses weighted-mean rho aggregation
- `_subtraction`: warns when rho × nUMI exceeds allocatable counts

### Fixed
- Zero-UMI cell detection and removal in `SoupChannel.__init__`
- `adjust_counts`: normalize `rho` array to 1-D before indexing (prevents scalar-as-0d bug)
- `estimate_decontx_rho`: convergence on both parameter delta and relative log-likelihood

---

## [1.0.0] — Reference R Package

> Original R SoupX package by Matthew D. Young & Sam Behjati.
> This Python implementation begins at v1.6.0 to indicate major feature additions beyond the R baseline.

- Core `SoupChannel` workflow: `load10X` → `autoEstCont` → `adjustCounts`
- tf-idf marker detection
- Bayesian rho estimation
- Subtraction and multinomial count correction
