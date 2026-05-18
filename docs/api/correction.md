# Correction Functions

## `adjust_counts`

Remove ambient RNA contamination from the count matrix.

```python
from SoupX import adjust_counts

corrected = adjust_counts(
    sc,
    clusters     = None,          # None | False | array
    method       = 'subtraction', # 'subtraction' | 'multinomial' | 'soupOnly'
    round_to_int = False,
    verbose      = 1,
    tol          = 1e-3,
)
```

:param sc: SoupChannel with rho set in meta_data.
:type sc: SoupChannel

:param clusters: Cluster labels for cluster-level adjustment. None = use sc.meta_data['clusters']; False = cell-level.
:type clusters: None, False, or array-like

:param method: Correction method.
  - ``'subtraction'`` (default): Iterative weighted subtraction of expected soup counts.
  - ``'multinomial'``: Greedy swap optimisation to maximise multinomial likelihood.
  - ``'soupOnly'``: Conservative removal only where counts are consistent with pure soup.
:type method: str

:param round_to_int: Stochastically round result to integers.
:type round_to_int: bool

:param verbose: Verbosity level (0 = silent, 1 = basic, 2 = chatty).
:type verbose: int

:param tol: Convergence tolerance (subtraction method).
:type tol: float

:return: Corrected count matrix, same shape as sc.toc.
:rtype: scipy.sparse.csc_matrix

## Choosing a method

| Method | Speed | Use case |
|---|---|---|
| `subtraction` | Fast | Default — proportional subtraction weighted by soup profile |
| `multinomial` | Slow | When you need the most statistically principled result |
| `soupOnly` | Fast | Conservative — only removes counts consistent with pure ambient origin |

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
