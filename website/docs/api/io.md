---
sidebar_position: 2
---

# I/O Functions

## `load_10x`

Load a 10X CellRanger output directory and build a SoupChannel.

```python
from SoupX import load_10x

sc = load_10x(
    data_dir         = 'path/to/cellranger/outs/',
    cell_ids         = None,
    channel_name     = None,
    include_features = ('Gene Expression',),
    verbose          = True,
)
```

**Parameters:**

- `data_dir` - Top-level CellRanger output directory. Must contain `raw_feature_bc_matrix/` (v3) or `raw_gene_bc_matrices/` (v2)
- `cell_ids` - Barcodes of droplets that are cells. If None, uses the filtered matrix
- `channel_name` - Name for this channel. Defaults to `data_dir`
- `include_features` - Feature types to retain (v3 multi-modal data)
- `verbose` - Print loading progress

**Returns:** `SoupChannel` with soup profile estimated from raw droplets

---

## `load_10x_h5`

Build a SoupChannel from 10X HDF5 files. 5–10× faster than MEX format for large datasets.

```python
from SoupX import load_10x_h5

sc = load_10x_h5(
    raw_h5      = 'outs/raw_feature_bc_matrix.h5',
    filtered_h5 = 'outs/filtered_feature_bc_matrix.h5',
)
```

**Parameters:**

- `raw_h5` - Path to `raw_feature_bc_matrix.h5` (all droplets)
- `filtered_h5` - Path to `filtered_feature_bc_matrix.h5`. Required when `cell_ids` is None
- `cell_ids` - Cell barcodes. Overrides `filtered_h5` when supplied

**Returns:** Constructed `SoupChannel`

---

## `read_10x`

Read a 10X MEX-format directory, returning the raw matrix without building a SoupChannel.

```python
from SoupX import read_10x

mat, gene_names, barcodes = read_10x('path/to/matrix_dir/')[:3]
```

**Parameters:**

- `data_dir` - Directory containing `matrix.mtx(.gz)`, `barcodes.tsv(.gz)`, and `genes/features.tsv(.gz)`

**Returns:** Tuple of `(sparse matrix, gene names, barcodes, feature_types)`

---

## `read_10x_h5`

Read a CellRanger HDF5 file, returning the raw matrix.

```python
from SoupX import read_10x_h5

mat, gene_names, barcodes, feature_types = read_10x_h5('matrix.h5')
```

**Parameters:**

- `h5_path` - Path to `.h5` file
- `genome` - Genome group name for v2 multi-genome files

**Returns:** Tuple of `(sparse matrix, gene names, barcodes, feature_types)`
