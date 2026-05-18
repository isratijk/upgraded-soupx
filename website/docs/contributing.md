---
sidebar_position: 12
---

# Contributing

Thank you for your interest in contributing! This document describes how to set up your development environment, submit changes, and follow project conventions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/<your-username>/Upgraded-soupX.git
   cd Upgraded-soupX
   ```

3. Add the upstream remote:

   ```bash
   git remote add upstream https://github.com/IsratIJK/Upgraded-soupX.git
   ```

## Development Setup

### Requirements

- Python ≥ 3.9
- `pip` (or `uv`)

### Install in editable mode with dev dependencies

```bash
pip install -e ".[dev]"
```

For downstream analysis dependencies:

```bash
pip install -e ".[dev,downstream]"
```

### Environment variables

```bash
cp .env.example .env
```

The `.env` file is ignored by git - never commit secrets.

## Code Style

- **Python version**: 3.9+ syntax
- **Formatter**: consistent 4-space indentation, 100 character max line length
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

## Docstring Format

All public functions must have docstrings following Sphinx RST style:

```python
def my_function(sc, threshold=0.05, verbose=True):
    """
    One-line summary of what the function does.

    :param sc: The SoupChannel object to process.
    :type sc: SoupChannel
    :param threshold: FDR threshold for the statistical test.
    :type threshold: float
    :return: Updated SoupChannel with results in meta_data.
    :rtype: SoupChannel
    :raises ValueError: If required attributes are missing from sc.
    """
```

## Testing

```bash
# Full test suite
pytest

# With coverage
pytest --cov=SoupX --cov-report=term-missing

# Specific module
pytest tests/test_decontx.py -v
```

### Regression tests

The regression golden baseline is stored in `tests/regression_golden.json`. If your change intentionally modifies numerical outputs, regenerate it:

```bash
python -c "from tests.conftest import *; regenerate_golden()"
```

Then commit the updated `regression_golden.json` with a clear explanation.

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
docs: add S3 dataset download instructions
test: add regression test for estimate_cell_rho
```

### Pull Request checklist

Before opening a PR, confirm:

- [ ] All existing tests pass (`pytest`)
- [ ] New/changed functions have complete docstrings
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Public API additions are exported in `SoupX/__init__.py` and `__all__`
- [ ] No secrets, credentials, or large binary files committed

## Reporting Issues

Use the GitHub issue tracker. Fill in the appropriate template:

- **Bug report**: unexpected error or incorrect result
- **Feature request**: new functionality or enhancement

For security issues, contact the maintainer directly rather than filing a public issue.

## Dataset Access

The benchmark datasets are stored in an AWS S3 bucket and are **not** included in the repository. See [Datasets](datasets) for full download instructions.

The `toyData` dataset (small, in-repo) is sufficient for running the unit tests and quick benchmarks without any download.
