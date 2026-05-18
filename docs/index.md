# SoupX — Python

**Ambient RNA contamination removal for droplet-based single-cell RNA-seq**

<p align="center">
  <!-- Replace with actual project banner when available -->
  <em>Project image coming soon</em>
</p>

---

SoupX is a full Python port and extension of the original [SoupX R package](https://github.com/constantAmateur/SoupX) (Young & Behjati, 2020). It removes ambient RNA contamination from droplet-based scRNA-seq count matrices — no R dependency required.

## Why SoupX?

Every 10X Chromium droplet contains a small amount of **soup** — free RNA from lysed cells that accumulated in the cell suspension before capture. This ambient contamination systematically inflates counts for highly expressed extracellular genes (e.g. haemoglobin in non-erythroid cells) and, if uncorrected, creates spurious cell-type differences.

SoupX estimates and removes this contamination by:

1. Modelling the soup expression profile from empty droplets.
2. Estimating the per-cell or per-cluster contamination fraction ρ (rho).
3. Subtracting expected soup counts from each cell.

## What's New in This Python Version

| Feature | R Baseline | Python Version |
|---|---|---|
| Core SoupX workflow | ✓ | ✓ (full parity) |
| DecontX per-cell decontamination | — | ✓ `run_decontx` |
| Per-cell ρ refinement | — | ✓ `estimate_cell_rho` |
| Doublet-aware estimation | — | ✓ `estimate_doublet_scores` |
| Gene-heterogeneity correction | — | ✓ `compute_gene_enrichment` |
| Iterative contamination refinement | — | ✓ `iterative_auto_est_cont` |
| Downstream analysis | — | ✓ PCA / UMAP / Leiden / DE |
| 8 quantitative benchmark metrics | — | ✓ |
| HDF5 input | — | ✓ `load_10x_h5` |
| Python ecosystem (scipy.sparse) | — | ✓ native |

## Installation

```bash
pip install -e .
# or with downstream extras (UMAP, Leiden clustering):
pip install -e ".[downstream]"
```

See [Installation](getting-started/installation.md) for full instructions.

## 30-Second Example

```python
from SoupX import load_10x, set_clusters, auto_est_cont, adjust_counts

sc        = load_10x('path/to/cellranger/outs/')
sc        = set_clusters(sc, cluster_labels)   # Seurat/Scanpy clusters
sc        = auto_est_cont(sc)                  # automatic ρ estimation
corrected = adjust_counts(sc)                  # corrected count matrix

print(f"Mean contamination: {sc.meta_data['rho'].mean():.1%}")
```

## Citation

If you use this package in your research, cite the original SoupX paper:

> Young, M.D. & Behjati, S. (2020). SoupX removes ambient RNA contamination from droplet-based single-cell RNA sequencing data. *GigaScience*, 9(12), giaa151.

If you use the DecontX module:

> Yang, S. et al. (2020). Decontamination of ambient RNA in single-cell RNA-seq with DecontX. *Genome Biology*, 21, 57.
