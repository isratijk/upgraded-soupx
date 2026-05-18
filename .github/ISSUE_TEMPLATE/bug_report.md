---
name: Bug report
about: Report incorrect behaviour or an unexpected error
title: "[BUG] "
labels: bug
assignees: ''
---

## Describe the bug

A clear and concise description of what the bug is.

## Steps to reproduce

Minimal code snippet that reproduces the issue:

```python
from SoupX import load_10x, set_clusters, auto_est_cont, adjust_counts

sc = load_10x('path/to/cellranger/outs/')
# ...
```

## Expected behaviour

What you expected to happen.

## Actual behaviour

What actually happened. Include the full error traceback:

```
Traceback (most recent call last):
  ...
```

## Environment

- **SoupX version** (`python -c "import SoupX; print(SoupX.__version__)"`)  :
- **Python version** (`python --version`)  :
- **OS**  :
- **Key package versions** (`pip show numpy scipy pandas`)  :

## Dataset

- [ ] `toyData` (bundled)
- [ ] `hgmm_1k`
- [ ] `pbmc_10k_v3`
- [ ] `E-MTAB-7407` fetal liver
- [ ] `rep1_Zenodo`
- [ ] Custom dataset

If custom, please describe it briefly (number of cells/genes, cellranger version, organism).

## Additional context

Any other context, screenshots, or links to related issues.
