"""
Direct tests for estimate_soup.py — covers all three methods:
  'fixed', 'statistical', 'emptydrops'

The conftest fixtures use method='fixed' implicitly; these tests exercise
the statistical and emptydrops code paths which are otherwise untouched.
"""

import numpy as np
import pandas as pd
import scipy.sparse
import pytest

from SoupX import SoupChannel
from SoupX.estimate_soup import estimate_soup


# ─── fixture helpers ─────────────────────────────────────────────────────────

def _make_sc_with_empties(n_genes=30, n_cells=40, n_drops=300, seed=1):
    """
    SoupChannel where empty droplets (UMI 1-10) are cleanly distinct from
    cells (UMI 50-150), giving all three soup methods a clear ambient signal.
    """
    rng = np.random.default_rng(seed)

    # Cells: high UMI, mixed gene expression
    toc_dense = rng.poisson(3.0, size=(n_genes, n_cells)).astype(float)
    toc = scipy.sparse.csc_matrix(toc_dense)

    # Droplets: first 200 are empty (low UMI), last 100 are "contaminant" cells
    empty_counts = rng.poisson(0.1, size=(n_genes, 200)).astype(float)
    cell_like    = rng.poisson(2.0, size=(n_genes, 100)).astype(float)
    tod_dense    = np.hstack([empty_counts, cell_like])
    tod = scipy.sparse.csc_matrix(tod_dense)

    drop_barcodes = [f"DROP{i:05d}" for i in range(n_drops)]

    return SoupChannel(
        tod=tod,
        toc=toc,
        genes=pd.Index([f"Gene{i}" for i in range(n_genes)]),
        cells=pd.Index([f"CELL{i:04d}" for i in range(n_cells)]),
        drop_barcodes=drop_barcodes,
        calc_soup_profile=False,   # manual so tod stays intact
    )


@pytest.fixture
def raw_sc():
    return _make_sc_with_empties()


# ─── method='fixed' ──────────────────────────────────────────────────────────

class TestFixedMethod:
    def test_returns_soup_profile(self, raw_sc):
        sc = estimate_soup(raw_sc, method='fixed', soup_range=(0, 100))
        assert sc.soup_profile is not None

    def test_soup_profile_sums_to_one(self, raw_sc):
        sc = estimate_soup(raw_sc, method='fixed', soup_range=(0, 100))
        assert np.isclose(sc.soup_profile['est'].sum(), 1.0, atol=1e-9)

    def test_no_negative_fractions(self, raw_sc):
        sc = estimate_soup(raw_sc, method='fixed', soup_range=(0, 100))
        assert (sc.soup_profile['est'] >= 0).all()

    def test_tod_cleared_by_default(self, raw_sc):
        sc = estimate_soup(raw_sc, method='fixed', soup_range=(0, 100))
        assert sc.tod is None

    def test_keep_droplets_preserves_tod(self, raw_sc):
        sc = estimate_soup(raw_sc, method='fixed', soup_range=(0, 100), keep_droplets=True)
        assert sc.tod is not None

    def test_empty_range_raises(self, raw_sc):
        with pytest.raises(ValueError, match="No droplets found"):
            estimate_soup(raw_sc, method='fixed', soup_range=(1000, 2000))

    def test_inplace_modifies_original(self, raw_sc):
        estimate_soup(raw_sc, method='fixed', soup_range=(0, 100), inplace=True)
        assert raw_sc.soup_profile is not None

    def test_returns_copy_by_default(self, raw_sc):
        sc2 = estimate_soup(raw_sc, method='fixed', soup_range=(0, 100))
        assert raw_sc.soup_profile is None   # original unchanged
        assert sc2.soup_profile is not None


# ─── method='statistical' ────────────────────────────────────────────────────

class TestStatisticalMethod:
    def test_returns_soup_profile(self, raw_sc):
        sc = estimate_soup(raw_sc, method='statistical', soup_range=(0, 15))
        assert sc.soup_profile is not None

    def test_soup_profile_sums_to_one(self, raw_sc):
        sc = estimate_soup(raw_sc, method='statistical', soup_range=(0, 15))
        assert np.isclose(sc.soup_profile['est'].sum(), 1.0, atol=1e-9)

    def test_no_negative_fractions(self, raw_sc):
        sc = estimate_soup(raw_sc, method='statistical', soup_range=(0, 15))
        assert (sc.soup_profile['est'] >= 0).all()

    def test_tod_cleared_by_default(self, raw_sc):
        sc = estimate_soup(raw_sc, method='statistical', soup_range=(0, 15))
        assert sc.tod is None

    def test_same_genes_as_fixed(self, raw_sc):
        sc_fix  = estimate_soup(raw_sc.copy(), method='fixed',       soup_range=(0, 15))
        sc_stat = estimate_soup(raw_sc.copy(), method='statistical', soup_range=(0, 15))
        assert list(sc_fix.soup_profile.index) == list(sc_stat.soup_profile.index)

    def test_profile_shape(self, raw_sc):
        sc = estimate_soup(raw_sc, method='statistical', soup_range=(0, 15))
        assert sc.soup_profile.shape[0] == len(raw_sc.genes)

    def test_counts_column_present(self, raw_sc):
        sc = estimate_soup(raw_sc, method='statistical', soup_range=(0, 15))
        assert 'counts' in sc.soup_profile.columns


# ─── method='emptydrops' ────────────────────────────────────────────────────

class TestEmptydropsMethod:
    def test_returns_soup_profile(self, raw_sc):
        sc = estimate_soup(raw_sc, method='emptydrops', soup_range=(0, 15))
        assert sc.soup_profile is not None

    def test_soup_profile_sums_to_one(self, raw_sc):
        sc = estimate_soup(raw_sc, method='emptydrops', soup_range=(0, 15))
        assert np.isclose(sc.soup_profile['est'].sum(), 1.0, atol=1e-9)

    def test_no_negative_fractions(self, raw_sc):
        sc = estimate_soup(raw_sc, method='emptydrops', soup_range=(0, 15))
        assert (sc.soup_profile['est'] >= 0).all()

    def test_tod_cleared_by_default(self, raw_sc):
        sc = estimate_soup(raw_sc, method='emptydrops', soup_range=(0, 15))
        assert sc.tod is None

    def test_profile_shape(self, raw_sc):
        sc = estimate_soup(raw_sc, method='emptydrops', soup_range=(0, 15))
        assert sc.soup_profile.shape[0] == len(raw_sc.genes)

    def test_counts_column_present(self, raw_sc):
        sc = estimate_soup(raw_sc, method='emptydrops', soup_range=(0, 15))
        assert 'counts' in sc.soup_profile.columns


# ─── method comparison: all three should agree on gene order for clean data ──

class TestMethodConsistency:
    """All three methods operating on the same clean data should rank the top
    soup gene the same way — they differ in which droplets they include, not
    in the ranking logic itself."""

    def test_top_gene_consistent(self, raw_sc):
        sc_fix  = estimate_soup(raw_sc.copy(), method='fixed',       soup_range=(0, 15))
        sc_stat = estimate_soup(raw_sc.copy(), method='statistical', soup_range=(0, 15))
        sc_ed   = estimate_soup(raw_sc.copy(), method='emptydrops',  soup_range=(0, 15))

        top_fix  = sc_fix.soup_profile['est'].idxmax()
        top_stat = sc_stat.soup_profile['est'].idxmax()
        top_ed   = sc_ed.soup_profile['est'].idxmax()

        assert top_fix == top_stat == top_ed, (
            f"Top soup gene differs across methods: "
            f"fixed={top_fix}, statistical={top_stat}, emptydrops={top_ed}"
        )

    def test_unknown_method_raises(self, raw_sc):
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_soup(raw_sc, method='magic', soup_range=(0, 100))
