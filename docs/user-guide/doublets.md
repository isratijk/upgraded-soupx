# Doublet-Aware Estimation

Doublets — droplets that capture two cells — corrupt TF-IDF marker gene selection because their mixed expression profiles look like co-expression of multiple cell types. Excluding them from the contamination estimation leads to cleaner per-cluster rho estimates.

> **Reference:** Wolock SL et al. (2019). Scrublet: Computational Identification of Cell Doublets in Single-Cell Transcriptomic Data. *Cell Systems*, 8, 281–291.

## How it works

1. Simulated doublets are created by summing raw count vectors of random cell pairs.
2. Real cells and simulated doublets are embedded together in PCA space (sqrt-normalized).
3. For each real cell, the doublet score = fraction of k nearest neighbours that are simulated doublets.
4. Cells with high doublet scores are masked during contamination estimation.

## Usage

### Standalone scoring

```python
from SoupX import estimate_doublet_scores

# toc: (genes × cells) sparse matrix
scores = estimate_doublet_scores(
    sc.toc,
    n_sim  = None,   # simulated doublets (default = n_cells)
    n_pcs  = 30,
    k      = 20,
    seed   = 42,
    n_hvg  = 2000,   # HVG pre-filter for large gene sets
)
# Returns ndarray (n_cells,) with doublet score in [0, 1]
```

### Doublet-aware auto_est_cont

```python
from SoupX import auto_est_cont_doublet_aware

sc = auto_est_cont_doublet_aware(
    sc,
    doublet_threshold = 0.25,  # cells above this are excluded from estimation
    n_sim             = None,
    n_pcs             = 30,
)
```

## Interpreting doublet scores

| Score | Interpretation |
|---|---|
| < 0.1 | Likely singlet |
| 0.1 – 0.3 | Uncertain |
| > 0.3 | Likely doublet |

The threshold is dataset-dependent. A typical expected doublet rate for 10X Chromium is ~0.8% per 1000 cells loaded.

## Notes

- The doublet scores are computed on the **raw** count matrix before correction. Do not use on corrected matrices.
- For very large datasets (>50k cells), reduce `n_sim` to keep memory manageable.
- PCA is computed on the `n_hvg` most variable genes to keep SVD tractable.
