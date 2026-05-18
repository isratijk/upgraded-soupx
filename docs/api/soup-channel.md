# SoupChannel

The central data container for a single droplet sequencing channel.

## Class: `SoupChannel`

```python
from SoupX import SoupChannel
```

### Attributes

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

### Constructor

```python
SoupChannel(
    tod,               # sparse (genes × droplets) — raw/unfiltered matrix
    toc,               # sparse (genes × cells)    — filtered matrix
    genes      = None, # gene names (array-like)
    cells      = None, # cell barcodes (array-like)
    meta_data  = None, # pd.DataFrame, index = cell barcodes
    calc_soup_profile = True,  # call estimate_soup automatically
    **kwargs           # extra attributes (channel_name, data_dir, …)
)
```

:param tod: Table of droplets (genes × all droplets).
:type tod: scipy.sparse matrix

:param toc: Table of counts (genes × cells, filtered barcodes only).
:type toc: scipy.sparse matrix

:param genes: Gene names. If None, uses integer indices.
:type genes: array-like, optional

:param cells: Cell barcodes. If None, uses integer indices.
:type cells: array-like, optional

:param meta_data: Additional per-cell metadata. Index must match cells.
:type meta_data: pd.DataFrame, optional

:param calc_soup_profile: Whether to call estimate_soup automatically on construction.
:type calc_soup_profile: bool

:raises ValueError: If tod and toc have different numbers of genes.

### Methods

#### `to_anndata(corrected=None)`

Convert SoupChannel to AnnData (cells × genes convention).

:param corrected: Corrected matrix from adjust_counts(). Stored as adata.layers['corrected'].
:type corrected: scipy.sparse matrix, optional
:return: AnnData object with toc as X, soup_profile in var, meta_data in obs.
:rtype: anndata.AnnData

#### `from_anndata(adata, tod, layer='counts', soup_profile=None, **kwargs)` *(classmethod)*

Build a SoupChannel from an AnnData object.

:param adata: AnnData in cells × genes convention.
:type adata: anndata.AnnData
:param tod: genes × droplets table for soup estimation.
:type tod: scipy.sparse matrix
:param layer: Layer to use as cell count matrix.
:type layer: str
:return: Constructed SoupChannel.
:rtype: SoupChannel

#### `save(path)`

Save SoupChannel to disk using pickle.

:param path: Output file path (e.g. 'channel.pkl').
:type path: str

#### `load(path)` *(classmethod)*

Load a SoupChannel saved with save().

:param path: Path to .pkl file.
:type path: str
:return: Loaded SoupChannel.
:rtype: SoupChannel

#### `copy()`

Return a deep copy of the SoupChannel.

:return: Deep copy.
:rtype: SoupChannel

### Example

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
