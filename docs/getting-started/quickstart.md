# Quick Start

## The standard workflow

```python
from SoupX import load_10x, set_clusters, auto_est_cont, adjust_counts

# 1. Load CellRanger output directory (v2 or v3, auto-detected)
sc = load_10x('path/to/cellranger/outs/')

# 2. Add cluster labels from your favourite clustering tool
#    (Seurat, Scanpy, CellRanger graph-based, etc.)
sc = set_clusters(sc, cluster_labels)

# 3. Estimate the contamination fraction automatically
sc = auto_est_cont(sc)
print(f"Estimated contamination: {sc.meta_data['rho'].mean():.1%}")

# 4. Produce the corrected count matrix
corrected = adjust_counts(sc)  # scipy.sparse.csc_matrix, same shape as sc.toc
```

## Loading from HDF5

CellRanger v3+ produces `raw_feature_bc_matrix.h5` and `filtered_feature_bc_matrix.h5`. Loading these is 5–10× faster than the MEX text format:

```python
from SoupX import load_10x_h5

sc = load_10x_h5(
    raw_h5      = 'outs/raw_feature_bc_matrix.h5',
    filtered_h5 = 'outs/filtered_feature_bc_matrix.h5',
)
```

## Using the toyData dataset (no download needed)

The bundled toy PBMC dataset is perfect for testing:

```python
import os
from SoupX import load_10x, set_clusters, auto_est_cont, adjust_counts
import pandas as pd

data_dir = 'dataset/upgraded_soupX_datasets/toyData'

sc = load_10x(data_dir)

meta = pd.read_csv(os.path.join(data_dir, 'metaData.tsv'), sep='\t', index_col=0)
sc   = set_clusters(sc, meta['clusters'])

sc        = auto_est_cont(sc, do_plot=False)
corrected = adjust_counts(sc)

print(sc)
# SoupChannel with N genes and N cells, rho=0.XXX
```

## Saving the result

```python
import scipy.io

# Save corrected matrix as Market Exchange (MEX) format
scipy.io.mmwrite('corrected_matrix.mtx', corrected)

# Or pickle the full SoupChannel
sc.save('my_channel.pkl')
sc2 = SoupChannel.load('my_channel.pkl')
```

## AnnData interoperability

```python
adata = sc.to_anndata(corrected=corrected)
# adata.X          — corrected matrix
# adata.layers['counts']    — raw counts
# adata.layers['corrected'] — corrected counts
# adata.obs        — per-cell metadata including rho
# adata.var        — gene metadata including soup_profile
```

## Next steps

- [Automatic workflow](../user-guide/automatic.md) — detailed parameter guide for `auto_est_cont`
- [DecontX](../user-guide/decontx.md) — per-cell probabilistic decontamination
- [Downstream analysis](../user-guide/downstream.md) — PCA, UMAP, clustering, DE on corrected data
