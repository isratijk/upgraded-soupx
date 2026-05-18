---
sidebar_position: 1
---

# SoupChannel

The central data container for a single droplet sequencing channel.

```python
from SoupX import SoupChannel
```

## Attributes

| Attribute | Type | Description |
|---|---|---|
| `tod` | `csc_matrix` or `None` | Table of all droplets (genes × droplets). Set to `None` after `estimate_soup` unless `keep_droplets=True`. |
| `toc` | `csc_matrix` | Table of counts (genes × cells). Filtered barcodes only. |
| `genes` | `pd.Index` | Gene names (row labels). |
| `cells` | `pd.Index` | Cell barcodes (column labels of `toc`). |
| `meta_data` | `pd.DataFrame` | Per-cell metadata. Always contains `nUMIs`. Populated with `clusters`, `rho`, DR columns as pipeline progresses. |
| `n_drop_umis` | `pd.Series` | Per-droplet UMI counts. |
| `soup_profile` | `pd.DataFrame` or `None` | Per-gene soup expression profile. Columns: `est` (fraction), `counts`. |
| `DR` | `list` or `None` | Column names in `meta_data` holding 2D dimension reduction coordinates. |
| `fit` | `object` or `None` | Stored fit result from contamination estimation. |

## Constructor

```python
SoupChannel(
    tod,               # sparse (genes × droplets) - raw/unfiltered matrix
    toc,               # sparse (genes × cells)    - filtered matrix
    genes      = None, # gene names (array-like)
    cells      = None, # cell barcodes (array-like)
    meta_data  = None, # pd.DataFrame, index = cell barcodes
    calc_soup_profile = True,  # call estimate_soup automatically
    **kwargs           # extra attributes (channel_name, data_dir, …)
)
```

**Parameters:**

- `tod` - Table of droplets (genes × all droplets), `scipy.sparse` matrix
- `toc` - Table of counts (genes × cells, filtered barcodes only), `scipy.sparse` matrix
- `genes` - Gene names. If None, uses integer indices
- `cells` - Cell barcodes. If None, uses integer indices
- `meta_data` - Additional per-cell metadata. Index must match cells
- `calc_soup_profile` - Whether to call `estimate_soup` automatically on construction

**Raises:** `ValueError` if `tod` and `toc` have different numbers of genes.

## Methods

### `to_anndata(corrected=None)`

Convert SoupChannel to AnnData (cells × genes convention).

- `corrected` - Corrected matrix from `adjust_counts()`. Stored as `adata.layers['corrected']`
- **Returns:** `anndata.AnnData` with `toc` as `X`, `soup_profile` in `var`, `meta_data` in `obs`

### `from_anndata(adata, tod, layer='counts', soup_profile=None, **kwargs)` *(classmethod)*

Build a SoupChannel from an AnnData object.

- `adata` - AnnData in cells × genes convention
- `tod` - genes × droplets table for soup estimation
- `layer` - Layer to use as cell count matrix
- **Returns:** Constructed `SoupChannel`

### `save(path)`

Save SoupChannel to disk using pickle.

- `path` - Output file path (e.g. `'channel.pkl'`)

### `load(path)` *(classmethod)*

Load a SoupChannel saved with `save()`.

- `path` - Path to `.pkl` file
- **Returns:** Loaded `SoupChannel`

### `copy()`

Return a deep copy of the SoupChannel.

## Example

```python
import scipy.sparse as sp
import numpy as np
from SoupX import SoupChannel

tod = sp.random(200, 5000, density=0.01, format='csc')  # genes × droplets
toc = sp.random(200, 300, density=0.05, format='csc')   # genes × cells

sc = SoupChannel(tod, toc)
print(sc)
# SoupChannel with 200 genes and 300 cells
```
