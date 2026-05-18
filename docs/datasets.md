# Datasets

## Overview

| Dataset | Cells | Format | Key Use |
|---|---|---|---|
| `toyData` | ~500 | 10X v2 MEX | In-repo; regression tests; always available |
| `pbmc_10k_v3` | ~10 K | 10X v3 MEX | Near-zero ρ baseline (healthy PBMC) |
| `hgmm_1k` | 1 K | 10X v2 MEX | Human+mouse barnyard; exact per-cell ground truth |
| `E-MTAB-7407_fetal_liver` | ~200 K | Custom archive | HBB-dominated soup; cell-type-level ground truth |
| `rep1_Zenodo` | — | HDF5 + RDS | Ground-truth CAST allele contamination |

`toyData` is committed to the repository under `dataset/upgraded_soupX_datasets/toyData/`. All other datasets are stored in AWS S3 and must be downloaded separately.

---

## Downloading datasets from S3

All benchmark datasets are distributed as a single archive:

```
upgraded_soupX_datasets.zip
```

After extraction the archive produces the same structure as `dataset/upgraded_soupX_datasets/`.

### Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| AWS CLI v2 | Option A | [docs.aws.amazon.com/cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) |
| boto3 | Option B | `pip install boto3` |
| curl / wget | Option C | System package manager |

### Configure AWS credentials

If accessing from a personal machine:

```bash
aws configure
# AWS Access Key ID:     <your key>
# AWS Secret Access Key: <your secret>
# Default region name:   us-east-1
# Default output format: json
```

If running on an EC2 instance with an IAM role attached, credentials are automatically resolved — no `aws configure` needed.

### Option A — AWS CLI

```bash
# Download
aws s3 cp \
  s3://${SOUPX_S3_BUCKET}/${SOUPX_S3_PREFIX}upgraded_soupX_datasets.zip \
  ./dataset/upgraded_soupX_datasets.zip

# Extract
cd dataset && unzip upgraded_soupX_datasets.zip && cd ..
```

Set `SOUPX_S3_BUCKET` and `SOUPX_S3_PREFIX` in your `.env` file (see `.env.example`).

### Option B — Python (boto3)

```python
import os
import zipfile
import boto3

bucket = os.environ["SOUPX_S3_BUCKET"]
prefix = os.environ.get("SOUPX_S3_PREFIX", "datasets/")
key    = f"{prefix}upgraded_soupX_datasets.zip"
dest   = "dataset/upgraded_soupX_datasets.zip"

print(f"Downloading s3://{bucket}/{key}  →  {dest}")
boto3.client("s3").download_file(bucket, key, dest)

print("Extracting …")
with zipfile.ZipFile(dest, "r") as zf:
    zf.extractall("dataset/")

print("Done. Contents under dataset/upgraded_soupX_datasets/")
```

### Option C — Pre-signed URL

If you have been given a pre-signed HTTPS URL:

```bash
# curl
curl -L "https://<presigned-url>" -o dataset/upgraded_soupX_datasets.zip
cd dataset && unzip upgraded_soupX_datasets.zip && cd ..

# wget
wget -O dataset/upgraded_soupX_datasets.zip "https://<presigned-url>"
cd dataset && unzip upgraded_soupX_datasets.zip && cd ..
```

---

## Expected directory layout after extraction

```
dataset/upgraded_soupX_datasets/
├── toyData/
│   ├── filtered_gene_bc_matrices/
│   │   └── hg19/
│   │       ├── barcodes.tsv
│   │       ├── genes.tsv
│   │       └── matrix.mtx
│   ├── raw_gene_bc_matrices/
│   │   └── hg19/
│   │       ├── barcodes.tsv
│   │       ├── genes.tsv
│   │       └── matrix.mtx
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
    ├── perCell_noMito_CAST_binom.RDS
    ├── seurat.RDS
    └── seurat_CAST.RDS
```

---

## Running benchmarks

```bash
# Quick smoke test using toyData (no download required)
python benchmarks/benchmark.py --quick

# List which datasets are detected
python benchmarks/benchmark.py --list

# Run a specific dataset
python benchmarks/benchmark.py --datasets hgmm

# Run all available datasets
python benchmarks/benchmark.py
```

Standalone per-dataset validation scripts:

```bash
python benchmarks/validate_hgmm.py         # barnyard — exact ground truth
python benchmarks/validate_fetal_liver.py  # fetal liver — HBB soup profile
```

---

## Dataset citations

- **toyData / PBMC**: 10X Genomics public datasets.
- **hgmm_1k**: 10X Genomics 1k 1:1 mixture of human (HEK293T) and mouse (NIH3T3) cells.
- **pbmc_10k_v3**: 10X Genomics 10k PBMCs from a healthy donor, v3 chemistry.
- **E-MTAB-7407 (Fetal Liver)**: Popescu, D.-M. et al. (2019). Decoding human fetal liver haematopoiesis. *Nature*, 574, 365–371.
- **rep1_Zenodo**: Young, M.D. et al. (2018). Single-cell transcriptomes from human kidneys reveal the cellular identity of renal tumours. *Science*, 361, 594–599.
