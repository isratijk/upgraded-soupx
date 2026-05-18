# SoupX — Python

**Ambient RNA contamination removal for droplet-based single-cell RNA-seq**

A full Python port and extension of the original [SoupX R package](https://github.com/constantAmateur/SoupX) (Young & Behjati, 2020). Drops the R dependency entirely, adds probabilistic per-cell decontamination (DecontX), doublet-aware estimation, gene-heterogeneity correction, a complete downstream analysis pipeline, and eight quantitative benchmark metrics — all on top of the same core `SoupChannel` abstraction.

---

## Table of Contents

- [Background](#background)
- [What's New vs the R Baseline](#whats-new-vs-the-r-baseline)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflows](#workflows)
  - [Automatic (recommended)](#automatic-recommended)
  - [Manual (known non-expressing genes)](#manual-known-non-expressing-genes)
  - [Per-cell probabilistic (DecontX)](#per-cell-probabilistic-decontx)
  - [Iterative refinement](#iterative-refinement)
  - [Visualization](#visualization)
- [API Reference](#api-reference)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
  - [Dataset overview](#dataset-overview)
  - [Downloading datasets from S3](#downloading-datasets-from-s3)
  - [Running benchmarks](#running-benchmarks)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Background

Droplet-based scRNA-seq protocols (10X Chromium and similar) capture cells inside oil droplets. Before cells are captured, free RNA released by lysed cells accumulates in the suspension — this ambient pool is called the **soup**. Every droplet carries a small amount of soup in addition to the cell's own transcriptome, introducing systematic contamination that inflates expression counts for highly expressed ambient genes across all cell types.

SoupX models this contamination using the empty-droplet pool to infer the soup expression profile, then estimates the per-cell or per-cluster contamination fraction (ρ) and subtracts it from the count matrix.

---

## What's New vs the R Baseline

| Feature | R Baseline | This Python Version |
|---|---|---|
| Core SoupX workflow | Yes | Yes (full parity) |
| DecontX per-cell decontamination | No | Yes (`run_decontx`) |
| Per-cell ρ refinement | No | Yes (`estimate_cell_rho`, `estimate_decontx_rho`) |
| Doublet-aware estimation | No | Yes (`estimate_doublet_scores`, `auto_est_cont_doublet_aware`) |
| Gene-heterogeneity correction | No | Yes (`compute_gene_enrichment`, `run_decontx_genehet`) |
| Iterative contamination refinement | No | Yes (`iterative_auto_est_cont`) |
| Downstream analysis (PCA/UMAP/clustering/DE) | No | Yes (`run_downstream`) |
| Quantitative benchmark metrics | No | Yes (8 metrics) |
| HDF5 input (`*.h5`) | No | Yes (`load_10x_h5`) |
| Python ecosystem integration | No | Native `scipy.sparse`, `pandas`, `numpy` |

---

## Installation

**Requirements:** Python ≥ 3.9

### From source (development)

```bash
git clone https://github.com/IsratIJK/Upgraded-soupX.git
cd Upgraded-soupX
pip install -e .
```

### With downstream analysis extras

```bash
pip install -e ".[downstream]"
```

This installs `scikit-learn`, `umap-learn`, `leidenalg`, and `python-igraph` for PCA, UMAP, tSNE, and Leiden clustering.

### Core dependencies

```
numpy>=1.21    scipy>=1.7    pandas>=1.3
statsmodels>=0.13    matplotlib>=3.4    tqdm>=4.60
```

See `requirements.txt` for the full dependency list.

---

## Quick Start

```python
from SoupX import load_10x, set_clusters, auto_est_cont, adjust_counts

# Load CellRanger output (v2 or v3 format, plain or gzipped)
sc = load_10x('path/to/cellranger/outs/')

# Attach cluster labels (from Seurat, Scanpy, or any clustering)
sc = set_clusters(sc, cluster_labels)

# Automatically estimate contamination fraction (ρ)
sc = auto_est_cont(sc)

# Remove contamination; returns corrected sparse count matrix
corrected = adjust_counts(sc)
```

---

## Workflows

### Automatic (recommended)

Uses tf-idf marker detection + Bayesian aggregation to estimate ρ without prior knowledge of which genes are contaminated.

```python
from SoupX import load_10x, set_clusters, auto_est_cont, adjust_counts

sc = load_10x('path/to/cellranger/outs/')
sc = set_clusters(sc, cluster_labels)
sc = auto_est_cont(sc)
corrected = adjust_counts(sc)

print(f"Mean contamination: {sc.meta_data['rho'].mean():.1%}")
```

### Manual (known non-expressing genes)

Use when you know which genes are biologically absent in certain cell populations (e.g., haemoglobin genes in non-erythroid cells).

```python
from SoupX import (load_10x, set_clusters,
                   estimate_non_expressing_cells,
                   calculate_contamination_fraction, adjust_counts)

sc = load_10x('path/to/cellranger/outs/')
sc = set_clusters(sc, cluster_labels)

gene_list = {'HB': ['HBB', 'HBA2', 'HBA1']}
use_to_est = estimate_non_expressing_cells(sc, gene_list)
sc = calculate_contamination_fraction(sc, gene_list, use_to_est)
corrected = adjust_counts(sc)
```

### Per-cell probabilistic (DecontX)

Two-component latent-variable model (LDA-based) that estimates ρ independently for every cell. More sensitive to contamination heterogeneity across the tissue.

```python
from SoupX import load_10x, set_clusters, run_decontx

sc = load_10x('path/to/cellranger/outs/')
sc = set_clusters(sc, cluster_labels)

sc_out = run_decontx(
    sc,
    n_topics=20,
    n_iter=500,
    n_hvg=3000,
    prior_rho=0.05,
    exclude_mt=True,
)

# Per-cell contamination estimates in sc_out.meta_data['rho']
```

### Iterative refinement

Runs `auto_est_cont` → `adjust_counts` → re-estimates markers in a loop until ρ converges.

```python
from SoupX import load_10x, set_clusters, iterative_auto_est_cont

sc = load_10x('path/to/cellranger/outs/')
sc = set_clusters(sc, cluster_labels)
sc = iterative_auto_est_cont(sc, max_iter=5, tol=1e-3)
```

### Visualization

```python
from SoupX import (plot_soup_correlation, plot_marker_distribution,
                   plot_marker_map, plot_change_map)

plot_soup_correlation(sc)
plot_marker_distribution(sc, gene_list)
plot_marker_map(sc, dr='umap')
plot_change_map(sc, corrected, dr='umap')
```

---

## API Reference

### Core Object

| Symbol | Module | Description |
|---|---|---|
| `SoupChannel` | `soup_channel` | Central data container: `tod`, `toc`, `soup_profile`, `meta_data` |

### Data I/O

| Function | Description |
|---|---|
| `load_10x(path)` | Load CellRanger output directory (v2 or v3, auto-detected) |
| `read_10x(path)` | Read count matrix only |
| `load_10x_h5(path)` | Load from CellRanger HDF5 file |
| `read_10x_h5(path)` | Read count matrix from HDF5 |

### Soup Estimation

| Function | Description |
|---|---|
| `estimate_soup(sc)` | Estimate soup profile from empty droplets |
| `set_soup_profile(sc, profile)` | Manually supply a soup profile |
| `quick_markers(sc)` | tf-idf marker detection for cluster annotation |

### Contamination Estimation

| Function | Description |
|---|---|
| `auto_est_cont(sc)` | Fully automatic ρ estimation (tf-idf + Bayesian) |
| `estimate_non_expressing_cells(sc, gene_list)` | Identify cells/clusters safe to use for manual calibration |
| `calculate_contamination_fraction(sc, gene_list, use_to_est)` | Manual Poisson GLM-based ρ |
| `estimate_cell_rho(sc)` | Per-cell refinement via empirical Bayes shrinkage |
| `estimate_decontx_rho(sc)` | Per-cell refinement via DecontX EM |
| `iterative_auto_est_cont(sc)` | Iterative refinement loop |

### Contamination Correction

| Function | Description |
|---|---|
| `adjust_counts(sc, method='subtraction')` | Remove contamination. Methods: `subtraction` (default), `soupOnly`, `multinomial` |

### Probabilistic Decontamination

| Function | Description |
|---|---|
| `run_decontx(sc, ...)` | Full DecontX two-component EM model |
| `select_n_topics(sc, ...)` | Cross-validation helper for topic count selection |

### Doublet Detection

| Function | Description |
|---|---|
| `estimate_doublet_scores(sc)` | Score cells by doublet likelihood |
| `auto_est_cont_doublet_aware(sc)` | Contamination estimation excluding probable doublets |

### Gene Heterogeneity

| Function | Description |
|---|---|
| `compute_gene_enrichment(sc)` | Per-gene enrichment over soup background |
| `reweight_soup_profile(sc)` | Adjust soup weights by gene-level heterogeneity |
| `run_decontx_genehet(sc)` | DecontX with gene-heterogeneity-aware soup model |

### Downstream Analysis

| Function | Description |
|---|---|
| `normalize_log1p(mat)` | Library-size normalization + log1p |
| `run_pca(sc)` | PCA via sklearn TruncatedSVD |
| `run_umap(sc)` | UMAP embedding |
| `run_tsne(sc)` | tSNE embedding |
| `cluster_leiden(sc)` | Leiden community detection |
| `cluster_kmeans(sc)` | K-means clustering |
| `differential_expression(sc)` | Wilcoxon-based DE per cluster |
| `score_cell_types(sc, marker_dict)` | Score cells against a marker dictionary |
| `plot_embedding(sc)` | Plot UMAP/tSNE coloured by any metadata column |
| `plot_top_de_genes(sc)` | Dot/violin plot of top DE genes |
| `run_downstream(sc)` | Full pipeline: norm → PCA → UMAP → Leiden → DE |

### Assessment Metrics

| Metric | Description |
|---|---|
| `cross_species_reduction` | Species-mixing reduction (barnyard datasets, exact ground truth) |
| `marker_fold_change` | Before/after fold change for known contamination markers |
| `cluster_membership_delta` | Shift in cluster composition after correction |
| `batch_entropy` | Local neighbourhood batch-mixing entropy |
| `hbb_expression_analysis` | HBB contamination analysis in non-erythroid cells |
| `cluster_silhouette` | Silhouette score of corrected clusters |
| `spurious_de_reduction` | Reduction in spurious DE genes between clusters |
| `marker_enrichment_score` | Enrichment of known cell-type markers post-correction |

---

## Benchmarks and Datasets

### Dataset overview

| Dataset | Cells | Format | Key Use |
|---|---|---|---|
| `toyData` (in-repo) | ~500 | 10X v2 | Regression golden baseline; always available |
| `pbmc_10k_v3` | ~10K | 10X v3 | Clean blood; near-zero ρ baseline |
| `hgmm_1k` | 1K | 10X v2 barnyard | Human+mouse mix; exact per-cell ground truth via species math |
| `E-MTAB-7407` (fetal liver) | ~200K | Custom archive | HBB-dominated soup; interpretable ground truth |
| `rep1_Zenodo` | — | HDF5 + RDS | Ground-truth CAST allele contamination |

### Downloading datasets from S3

All benchmark datasets (except `toyData`) are bundled in a single archive on AWS S3:

```
s3://<SOUPX_S3_BUCKET>/<SOUPX_S3_PREFIX>upgraded_soupX_datasets.zip
```

After downloading and extracting, the contents go under `dataset/`:

```
dataset/
└── upgraded_soupX_datasets/
    ├── toyData/                  ← in-repo, always present
    ├── hgmm_1k/
    ├── pbmc_10k_v3/
    ├── E-MTAB-7407_fetal_liver/
    └── rep1_Zenodo/
```

#### Option A — AWS CLI (recommended)

```bash
# Configure credentials (or use IAM role / instance profile)
aws configure

# Download
aws s3 cp s3://<BUCKET>/<PREFIX>upgraded_soupX_datasets.zip ./dataset/

# Extract
cd dataset && unzip upgraded_soupX_datasets.zip && cd ..
```

Replace `<BUCKET>` and `<PREFIX>` with the values from your `.env` file.

#### Option B — Python (boto3)

```python
import os, zipfile, boto3

bucket = os.environ["SOUPX_S3_BUCKET"]
prefix = os.environ.get("SOUPX_S3_PREFIX", "datasets/")
dest   = "dataset/upgraded_soupX_datasets.zip"

boto3.client("s3").download_file(bucket, f"{prefix}upgraded_soupX_datasets.zip", dest)

with zipfile.ZipFile(dest, "r") as zf:
    zf.extractall("dataset/")
```

#### Option C — Pre-signed URL

```bash
curl -L "https://<presigned-url>" -o dataset/upgraded_soupX_datasets.zip
cd dataset && unzip upgraded_soupX_datasets.zip && cd ..
```

#### Expected directory layout after extraction

```
dataset/upgraded_soupX_datasets/
├── toyData/
│   ├── filtered_gene_bc_matrices/
│   ├── raw_gene_bc_matrices/
│   └── metaData.tsv
├── hgmm_1k/
│   ├── hgmm_1k_filtered_gene_bc_matrices.tar.gz
│   └── hgmm_1k_raw_gene_bc_matrices.tar.gz
├── pbmc_10k_v3/
│   ├── analysis.tar.gz
│   ├── filtered.tar.gz
│   └── raw.tar.gz
├── E-MTAB-7407_fetal_liver/
│   └── FCAImmP7352195.tar.gz
└── rep1_Zenodo/
    ├── filtered_feature_bc_matrix.h5
    ├── raw_feature_bc_matrix.h5
    ├── rep1_cast_gt.csv
    ├── seurat.RDS
    └── seurat_CAST.RDS
```

### Running benchmarks

```bash
# Quick smoke test (toyData only, no download needed)
python benchmarks/benchmark.py --quick

# List dataset availability
python benchmarks/benchmark.py --list

# Run specific datasets
python benchmarks/benchmark.py --datasets hgmm fetal_liver

# Run all available
python benchmarks/benchmark.py
```

### Standalone validation scripts

```bash
python benchmarks/validate_hgmm.py          # barnyard — exact ground truth
python benchmarks/validate_fetal_liver.py   # fetal liver — HBB soup profile
```

---

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run full test suite
pytest

# With coverage report
pytest --cov=SoupX --cov-report=term-missing

# Run a specific module
pytest tests/test_decontx.py -v
```

The test suite covers 16 modules:

| Module | Focus |
|---|---|
| `test_core.py` | `SoupChannel`, `load_10x`, basic workflows |
| `test_io.py` | I/O functions, v2/v3 format handling |
| `test_estimate_soup.py` | Soup profile estimation |
| `test_markers.py` | tf-idf marker detection |
| `test_estimation.py` | `auto_est_cont`, non-expressing cell detection |
| `test_correction.py` | `adjust_counts` (all three methods) |
| `test_decontx.py` | `run_decontx`, topic selection |
| `test_downstream.py` | PCA, UMAP, clustering, DE |
| `test_assessment_metrics.py` | All 8 assessment metrics |
| `test_plot.py` | `plot_*` functions |
| `test_edge_cases.py` | Boundary and corner cases |
| `test_utils.py` | Utility helpers |
| `test_regression.py` | Golden regression against `regression_golden.json` |
| `test_validate_hgmm.py` | Barnyard dataset integration test |

---

## Project Structure

```
Upgraded-soupX/
├── SoupX/                          # Python package (v1.6.0)
│   ├── __init__.py                 # Public API, version
│   ├── soup_channel.py             # SoupChannel class
│   ├── io.py                       # load_10x, load_10x_h5
│   ├── estimate_soup.py            # Soup profile estimation
│   ├── markers.py                  # quick_markers (tf-idf)
│   ├── estimation.py               # auto_est_cont, calculate_contamination_fraction
│   ├── correction.py               # adjust_counts
│   ├── decontx.py                  # run_decontx (LDA two-component EM)
│   ├── doublet.py                  # estimate_doublet_scores
│   ├── iterative.py                # iterative_auto_est_cont
│   ├── gene_het.py                 # compute_gene_enrichment, run_decontx_genehet
│   ├── set_properties.py           # set_clusters, set_contamination_fraction
│   ├── downstream.py               # PCA, UMAP, clustering, DE
│   ├── metrics.py                  # 8 assessment metrics
│   ├── plot.py                     # Visualization functions
│   └── utils.py                    # Internal helpers
├── benchmarks/                     # Benchmark runner + per-dataset validation scripts
├── dataset/                        # Dataset root (contents downloaded from S3)
│   └── upgraded_soupX_datasets/   # Extracted from upgraded_soupX_datasets.zip
├── tests/                          # Pytest test suite (16 modules)
├── plots/                          # Benchmark visualization outputs
├── docs/                           # Static documentation site (MkDocs)
├── .github/
│   ├── ISSUE_TEMPLATE/             # Bug report + feature request templates
│   ├── PULL_REQUEST_TEMPLATE.md    # PR checklist
│   └── workflows/tests.yml         # CI (GitHub Actions)
├── pyproject.toml                  # Package metadata and dependencies
├── requirements.txt                # Pinned/optional dependencies
├── .env.example                    # Environment variable template
├── CHANGELOG.md                    # Version history
├── CONTRIBUTING.md                 # Contribution guidelines
└── LICENSE                         # MIT License
```

---

## Citation

If you use this package in your research, please cite the original SoupX paper:

> Young, M.D. & Behjati, S. (2020). SoupX removes ambient RNA contamination from droplet-based single-cell RNA sequencing data. *GigaScience*, 9(12), giaa151. https://doi.org/10.1093/gigascience/giaa151

If you use the DecontX-based decontamination:

> Yang, S. et al. (2020). Decontamination of ambient RNA in single-cell RNA-seq with DecontX. *Genome Biology*, 21, 57. https://doi.org/10.1186/s13059-020-1950-6

Dataset citations:

- **E-MTAB-7407 (Fetal Liver)**: Popescu, D.-M. et al. (2019). Decoding human fetal liver haematopoiesis. *Nature*, 574, 365–371.
- **scKidneyTumors**: Young, M.D. et al. Single cell transcriptomes from human kidneys reveal the cellular identity of renal tumours. *Science*.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Original SoupX R package: MIT License, Copyright (c) Matthew Young.
