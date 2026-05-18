---
sidebar_position: 5
---

# Gene Heterogeneity Correction

Standard DecontX uses the raw normalized empty-droplet counts as the fixed soup profile. In practice, some genes are uniquely ambient (e.g. haemoglobin in a non-erythroid experiment) while others are equally abundant in soup and cells, making them uninformative for separating contamination from native expression.

The gene-heterogeneity module **reweights the soup profile** to amplify truly ambient genes and suppress ambiguous ones before the DecontX EM.

## Enrichment weight

For each gene g:

```
enrichment_g = log1p(soup_share_g / cell_share_g)
```

clipped to `[min_weight, max_weight]`. Genes with high soup fraction and low cellular expression get the largest boost.

## Usage

### Compute enrichment weights

```python
from SoupX import compute_gene_enrichment

weights = compute_gene_enrichment(
    sc,
    log_smooth = True,
    min_weight = 0.5,
    max_weight = 2.0,
)
# Returns ndarray (n_genes,)
```

### Reweight the soup profile

```python
from SoupX import reweight_soup_profile

sc_weighted = reweight_soup_profile(
    sc,
    log_smooth = True,
    min_weight = 0.5,
    max_weight = 2.0,
)
# sc_weighted.soup_profile['est'] is now reweighted
```

### Full DecontX with gene-het reweighting

```python
from SoupX import run_decontx_genehet

sc_out = run_decontx_genehet(
    sc,
    n_topics   = None,   # None = n_unique_clusters
    n_iter     = 300,
    log_smooth = True,
    min_weight = 0.5,
    max_weight = 1.5,
)
```

## When to use

Gene-het correction is most beneficial when:

- The soup expression profile closely resembles the cellular expression profile
- Standard `run_decontx` produces near-zero rho for most cells (model cannot distinguish contamination from native expression)
- The dataset has tissue-specific contamination (e.g. blood contamination in a non-blood tissue)

:::tip Benchmark result
On the HGMM barnyard dataset, `upg-genehet` reduced spurious DE genes from **347 (baseline)** down to just **8** - a 98% reduction.
:::
