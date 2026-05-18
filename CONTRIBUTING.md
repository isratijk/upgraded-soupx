# Contributing to Upgraded-SoupX

Thank you for your interest in contributing! This document describes how to set up your development environment, submit changes, and follow project conventions.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Docstring Format](#docstring-format)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Dataset Access](#dataset-access)

---

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:

   ```bash
   git clone https://github.com/<your-username>/Upgraded-soupX.git
   cd Upgraded-soupX
   ```

3. Add the upstream remote:

   ```bash
   git remote add upstream https://github.com/IsratIJK/Upgraded-soupX.git
   ```

---

## Development Setup

### Requirements

- Python ≥ 3.9
- `pip` (or `uv`)

### Install in editable mode with dev dependencies

```bash
pip install -e ".[dev]"
```

For downstream analysis dependencies (PCA, UMAP, Leiden clustering):

```bash
pip install -e ".[dev,downstream]"
```

### Environment variables

Copy `.env.example` to `.env` and fill in any values you need:

```bash
cp .env.example .env
```

The `.env` file is ignored by git — never commit secrets.

---

## Code Style

- **Python version**: 3.9+ syntax (no walrus operator unless guarded by version check)
- **Formatter**: no mandatory formatter, but consistent 4-space indentation
- **Line length**: 100 characters max
- **Imports**: stdlib → third-party → local, separated by blank lines
- **Type hints**: encouraged for public API; not required for private helpers
- **Logging**: use `warnings.warn()` for user-visible issues, `print()` only behind `verbose` flags

### Naming conventions

| Kind | Convention | Example |
|---|---|---|
| Public functions | `snake_case` | `auto_est_cont` |
| Private helpers | `_snake_case` | `_decontx_em` |
| Classes | `PascalCase` | `SoupChannel` |
| Constants | `UPPER_SNAKE` | `REPO_ROOT` |
| Module-level docstrings | Required | `"""Description.\n\n..."""` |

---

## Docstring Format

All public functions must have docstrings following Sphinx RST style:

```python
def my_function(sc, threshold=0.05, verbose=True):
    """
    One-line summary of what the function does.

    Optional longer description explaining the algorithm, assumptions,
    or important behaviour.

    :param sc: The SoupChannel object to process.
    :type sc: SoupChannel
    :param threshold: FDR threshold for the statistical test.
    :type threshold: float
    :param verbose: Print progress messages when True.
    :type verbose: bool
    :return: Updated SoupChannel with results in meta_data.
    :rtype: SoupChannel

    :raises ValueError: If required attributes are missing from sc.
    """
```

Private helper functions (prefixed with `_`) require at minimum a one-line docstring describing their purpose.

---

## Testing

### Run the full test suite

```bash
pytest
```

### Run with coverage

```bash
pytest --cov=SoupX --cov-report=term-missing
```

### Run a specific module

```bash
pytest tests/test_decontx.py -v
```

### Regression tests

The regression golden baseline is stored in `tests/regression_golden.json`. If your change intentionally modifies numerical outputs, regenerate it:

```bash
python -c "
from tests.conftest import *
regenerate_golden()
"
```

Then commit the updated `regression_golden.json` with a clear explanation.

### Test data

Unit tests use the `toyData` dataset bundled in `dataset/upgraded_soupX_datasets/toyData/`. This dataset is always present and does **not** require downloading from S3.

Integration tests that require the full datasets (hgmm, fetal liver, pbmc) are gated by the presence of the dataset directory. See [Dataset Access](#dataset-access).

---

## Submitting Changes

### Branch naming

```
feature/<short-description>
fix/<short-description>
docs/<short-description>
refactor/<short-description>
```

### Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add emptydrops soup estimation method
fix: handle zero-UMI cells in SoupChannel constructor
docs: add S3 dataset download instructions to README
test: add regression test for estimate_cell_rho
refactor: extract _decontx_em into estimation module
```

### Pull Request checklist

Before opening a PR, confirm:

- [ ] All existing tests pass (`pytest`)
- [ ] New/changed functions have complete docstrings (`:param`/`:return` style)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Public API additions are exported in `SoupX/__init__.py` and `__all__`
- [ ] No secrets, credentials, or large binary files committed

Use the PR template — it will be pre-filled when you open a PR.

---

## Reporting Issues

Use the GitHub issue tracker. Fill in the appropriate template:

- **Bug report**: unexpected error or incorrect result
- **Feature request**: new functionality or enhancement

For security issues, contact the maintainer directly rather than filing a public issue.

---

## Dataset Access

The benchmark datasets are stored in an AWS S3 bucket and are **not** included in the repository. See [docs/datasets.md](docs/datasets.md) for full download instructions.

The `toyData` dataset (small, in-repo) is sufficient for running the unit tests and quick benchmarks without any download.
