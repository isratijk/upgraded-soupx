---
id: intro
sidebar_position: 1
slug: /
---

# Upgraded-SoupX

**Ambient RNA contamination removal for droplet-based single-cell RNA-seq**

**Developed by [Israt Jahan Khan](https://www.isratjahankhan.com)** &mdash; [LinkedIn](https://www.linkedin.com/in/isratijk/) · [Google Scholar](https://scholar.google.com/citations?user=n4mCE9QAAAAJ&hl=en) · [isratjahankhanijk@gmail.com](mailto:isratjahankhanijk@gmail.com)

Upgraded-SoupX is a full Python port and extension of the original [SoupX R package](https://github.com/constantAmateur/SoupX) (Young & Behjati, 2020). It removes ambient RNA contamination from droplet-based scRNA-seq count matrices - no R dependency required.

## Why ambient RNA matters

Every 10X Chromium droplet contains a small amount of **soup** - free RNA from lysed cells that accumulated in the cell suspension before capture. This contamination:

- Inflates counts for highly expressed extracellular genes (e.g. haemoglobin in non-erythroid cells)
- Creates spurious cell-type differences across clusters
- Reduces differential expression signal-to-noise

Upgraded-SoupX estimates and removes this contamination by:

1. Modelling the soup expression profile from empty droplets
2. Estimating the per-cell or per-cluster contamination fraction rho
3. Subtracting expected soup counts from each cell

## What this Python version adds

| Feature | R Baseline | Python Version |
|---|---|---|
| Core SoupX workflow | Yes | Yes (full parity) |
| DecontX per-cell decontamination | - | Yes `run_decontx` |
| Per-cell rho refinement | - | Yes `estimate_cell_rho` |
| Doublet-aware estimation | - | Yes `estimate_doublet_scores` |
| Gene-heterogeneity correction | - | Yes `compute_gene_enrichment` |
| Iterative contamination refinement | - | Yes `iterative_auto_est_cont` |
| Downstream analysis | - | Yes PCA / UMAP / Leiden / DE |
| 8 quantitative benchmark metrics | - | Yes |
| HDF5 input | - | Yes `load_10x_h5` |
| Python ecosystem (scipy.sparse) | - | Yes native |

## Quick navigation

- **[Installation](getting-started/installation)** - requirements and install commands
- **[Quick Start](getting-started/quickstart)** - four-line example
- **[Automatic Workflow](user-guide/automatic)** - the recommended starting point
- **[DecontX](user-guide/decontx)** - per-cell decontamination with LDA topics
- **[Benchmark Results](results)** - plots and findings across 5 datasets
- **[API Reference](api/soup-channel)** - full function documentation

## Citation

If you use Upgraded-SoupX, please cite this software:

> Khan, I.J. (2026). *Upgraded-SoupX: A Python port and extension of SoupX for ambient RNA decontamination in single-cell RNA-seq.* GitHub. https://github.com/IsratIJK/Upgraded-soupX

Also cite the original algorithms this work builds on:

> Young, M.D. & Behjati, S. (2020). SoupX removes ambient RNA contamination from droplet-based single-cell RNA sequencing data. *GigaScience*, 9(12), giaa151.

If you use the DecontX module:

> Yang, S. et al. (2020). Decontamination of ambient RNA in single-cell RNA-seq with DecontX. *Genome Biology*, 21, 57.
