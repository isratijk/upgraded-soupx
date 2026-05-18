# Visualization

## Soup correlation plot

Scatter plot comparing the aggregate cell expression profile vs the soup profile (log10 scale). Genes that fall far above the diagonal in cells are potential contamination markers.

```python
from SoupX import plot_soup_correlation

fig = plot_soup_correlation(sc, save_path=None)
```

## Marker distribution

Violin plot of observed/expected expression ratio for marker gene sets across clusters.

```python
from SoupX import plot_marker_distribution

gene_list = {'HB': ['HBB', 'HBA2', 'HBA1']}
fig = plot_marker_distribution(sc, non_expressed_gene_list=gene_list)
```

## Marker map

2D scatter plot of the dimension reduction coloured by expression of a marker gene.

```python
from SoupX import plot_marker_map

# Requires sc.DR to be set (loaded automatically by load_10x when tSNE/UMAP present,
# or set manually via set_dr())
fig = plot_marker_map(sc, gene='HBB', dr='umap')
```

## Change map

Before/after comparison of a gene's expression in the 2D embedding.

```python
from SoupX import plot_change_map

corrected = adjust_counts(sc)
fig = plot_change_map(sc, corrected, gene='HBB', dr='umap')
```

## Downstream embedding

After running `run_downstream`, plot the UMAP coloured by cluster labels:

```python
from SoupX.downstream import plot_embedding, plot_top_de_genes

plot_embedding(
    embedding = result['embedding'],   # (n_cells × 2)
    labels    = result['cluster_labels'],
    title     = 'UMAP — corrected',
    save_path = None,
)

plot_top_de_genes(result['de_results'], top_n=5)
```

## Saving figures

All plot functions accept an optional `save_path` argument:

```python
fig = plot_soup_correlation(sc, save_path='figures/soup_correlation.png')
```

When `save_path` is set, the figure is saved at 150 DPI and closed automatically. When `None`, the figure is displayed interactively.
