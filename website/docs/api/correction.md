---
sidebar_position: 4
---

# Correction Functions

## `adjust_counts`

Remove ambient RNA contamination from the count matrix.

```python
from SoupX import adjust_counts

corrected = adjust_counts(
    sc,
    clusters     = None,
    method       = 'subtraction',  # 'subtraction' | 'multinomial' | 'soupOnly'
    round_to_int = False,
    verbose      = 1,
    tol          = 1e-3,
)
```

**Parameters:**

- `sc` - `SoupChannel` with `rho` set in `meta_data`
- `clusters` - Cluster labels for cluster-level adjustment. `None` = use `sc.meta_data['clusters']`; `False` = cell-level
- `method` - Correction method (see table below)
- `round_to_int` - Stochastically round result to integers
- `verbose` - Verbosity level (0 = silent, 1 = basic, 2 = chatty)
- `tol` - Convergence tolerance (subtraction method)

**Returns:** Corrected count matrix `scipy.sparse.csc_matrix`, same shape as `sc.toc`

## Choosing a method

| Method | Speed | Use case |
|---|---|---|
| `subtraction` | Fast | Default - proportional subtraction weighted by soup profile |
| `multinomial` | Slow | When you need the most statistically principled result |
| `soupOnly` | Fast | Conservative - only removes counts consistent with pure ambient origin |

## Example

```python
from SoupX import load_10x, set_clusters, auto_est_cont, adjust_counts

sc        = load_10x('path/to/cellranger/outs/')
sc        = set_clusters(sc, cluster_labels)
sc        = auto_est_cont(sc)
corrected = adjust_counts(sc)

# corrected is a genes × cells sparse matrix
print(corrected.shape)  # (n_genes, n_cells)

# Compare total counts before and after
import numpy as np
print("Before:", sc.toc.sum())
print("After: ", corrected.sum())
```
