---
sidebar_position: 1
---

# Installation

## Requirements

- Python ≥ 3.9
- pip

## Install from source

```bash
git clone https://github.com/IsratIJK/Upgraded-soupX.git
cd Upgraded-soupX
pip install -e .
```

## Optional extras

### Downstream analysis

Enables PCA, UMAP, tSNE, Leiden clustering, and differential expression:

```bash
pip install -e ".[downstream]"
```

Installs: `scikit-learn>=1.0`, `umap-learn>=0.5`, `leidenalg>=0.9`, `python-igraph>=0.10`

### HDF5 support

For loading CellRanger HDF5 files (`*.h5`):

```bash
pip install h5py
```

### AnnData interoperability

For `SoupChannel.to_anndata()` / `SoupChannel.from_anndata()`:

```bash
pip install anndata
```

### Development

```bash
pip install -e ".[dev]"
```

Installs `pytest` and `pytest-cov`.

## Verify the installation

```python
import SoupX
print(SoupX.__version__)
# 1.7.0
```

## Environment variables

Copy `.env.example` to `.env` and fill in values needed for dataset downloads from S3:

```bash
cp .env.example .env
```

See the [Datasets](../datasets) page for details.
