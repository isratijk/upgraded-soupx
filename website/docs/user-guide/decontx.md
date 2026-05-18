---
sidebar_position: 3
---

# DecontX

DecontX is a Bayesian Dirichlet-multinomial decontamination model that estimates a per-cell contamination fraction θ using LDA (Latent Dirichlet Allocation) topics to model native expression.

:::info Reference
Yang S et al. (2020). Decontamination of ambient RNA in single-cell RNA-seq with DecontX. *Genome Biology*, 21, 289.
:::

## Model

Each cell's count vector is modelled as a two-component mixture:

```
x_i ~ Multinomial(n_i, θ_i · π + (1 − θ_i) · φ_i)
```

where:

- **θ_i** - per-cell contamination fraction (what we estimate)
- **π** - the soup (ambient) expression profile (fixed, from empty droplets)
- **φ_i** - the cell's native expression profile (modelled via K shared LDA topics)

Using shared LDA topics means rare cell types borrow expression patterns from similar cells instead of relying only on their own sparse counts.

## Usage

```python
from SoupX import load_10x, set_clusters, run_decontx, adjust_counts

sc = load_10x('path/to/cellranger/outs/')
sc = set_clusters(sc, cluster_labels)

sc_decontx = run_decontx(
    sc,
    n_topics   = 20,      # LDA topics (more → better, slower)
    n_iter     = 500,     # EM iterations
    n_hvg      = 3000,    # highly-variable genes for PCA init
    prior_rho  = 0.05,    # initial contamination guess
    exclude_mt = True,    # zero MT genes from soup (recommended)
    verbose    = True,
)

# Per-cell rho stored in:
print(sc_decontx.meta_data['rho'].describe())

corrected = adjust_counts(sc_decontx)
```

## Choosing the number of topics

Use `select_n_topics` to find the elbow in the held-out log-likelihood curve:

```python
from SoupX import select_n_topics

results = select_n_topics(sc, topic_range=range(2, 30, 2), n_iter=200)
# Returns dict: {'n_topics': [...], 'log_likelihood': [...]}
# Plot and find the elbow point.
```

Rule of thumb: K ≈ number of distinct cell types in the dataset. Values between 10 and 30 work well for most experiments.

## MT gene exclusion

Mitochondrial genes leak from damaged cells into every droplet and are also genuinely expressed by real cells. When the soup profile is MT-dominated, the model mistakes real MT expression for contamination. Set `exclude_mt=True` to zero MT genes from the soup profile before EM:

```python
sc_out = run_decontx(sc, exclude_mt=True)
```

## Gene-heterogeneity variant

When soup expression substantially overlaps with cellular expression, the standard DecontX soup profile may be ambiguous. `run_decontx_genehet` reweights the soup profile to amplify truly ambient genes before the EM:

```python
from SoupX import run_decontx_genehet

sc_out = run_decontx_genehet(
    sc,
    log_smooth = True,
    min_weight = 0.5,
    max_weight = 1.5,
)
```

## Parameters reference

| Parameter | Default | Description |
|---|---|---|
| `n_topics` | 20 | LDA topics |
| `n_iter` | 500 | Maximum EM iterations |
| `n_hvg` | None | HVGs for PCA initialisation (`None` = all genes) |
| `prior_rho` | None | Initial contamination guess (None = auto from auto_est_cont) |
| `exclude_mt` | False | Zero MT genes from soup before EM |
| `pca_init` | True | Use PCA to initialise topic proportions |
| `seed` | 42 | Random seed |
| `verbose` | True | Print convergence progress |
