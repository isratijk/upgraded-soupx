"""
Edge-case tests: 0 cells, single cell, boundary rho values, empty gene lists,
and the all-useEst=False posterior bug (estimation.py regression).
"""

import numpy as np
import pandas as pd
import scipy.sparse
import pytest
import warnings

from SoupX import (
    SoupChannel,
    set_clusters,
    set_contamination_fraction,
    estimate_non_expressing_cells,
    auto_est_cont,
)


def _make_sc(n_genes=10, n_cells=2, n_drops=30, seed=7):
    rng = np.random.default_rng(seed)
    toc = scipy.sparse.csc_matrix(rng.poisson(1.0, size=(n_genes, n_cells)))
    tod = scipy.sparse.csc_matrix(rng.poisson(0.05, size=(n_genes, n_drops)))
    return SoupChannel(tod=tod, toc=toc, calc_soup_profile=True)


# ── 0 cells ───────────────────────────────────────────────────────────────────

class TestZeroCells:
    def test_construction(self):
        tod = scipy.sparse.csc_matrix(np.ones((5, 20)))
        toc = scipy.sparse.csc_matrix((5, 0))
        sc = SoupChannel(tod=tod, toc=toc, calc_soup_profile=True)
        assert sc.toc.shape == (5, 0)
        assert len(sc.cells) == 0
        assert len(sc.meta_data) == 0

    def test_soup_profile_present(self):
        tod = scipy.sparse.csc_matrix(np.ones((5, 20)))
        toc = scipy.sparse.csc_matrix((5, 0))
        sc = SoupChannel(tod=tod, toc=toc, calc_soup_profile=True)
        assert sc.soup_profile is not None
        assert np.isclose(sc.soup_profile['est'].sum(), 1.0)

    def test_set_contamination(self):
        tod = scipy.sparse.csc_matrix(np.ones((5, 20)))
        toc = scipy.sparse.csc_matrix((5, 0))
        sc = SoupChannel(tod=tod, toc=toc, calc_soup_profile=True)
        sc2 = set_contamination_fraction(sc, 0.05)
        assert 'rho' in sc2.meta_data.columns
        assert len(sc2.meta_data) == 0


# ── single cell ───────────────────────────────────────────────────────────────

class TestSingleCell:
    def test_construction(self):
        sc = _make_sc(n_cells=1)
        assert sc.toc.shape[1] == 1
        assert len(sc.cells) == 1
        assert len(sc.meta_data) == 1

    def test_soup_profile_sums_to_one(self):
        sc = _make_sc(n_cells=1)
        assert np.isclose(sc.soup_profile['est'].sum(), 1.0)

    def test_set_contamination(self):
        sc = _make_sc(n_cells=1)
        sc2 = set_contamination_fraction(sc, 0.05)
        assert sc2.meta_data['rho'].iloc[0] == pytest.approx(0.05)

    def test_n_umis_positive(self):
        rng = np.random.default_rng(0)
        toc = scipy.sparse.csc_matrix(rng.poisson(5.0, size=(10, 1)) + 1)
        tod = scipy.sparse.csc_matrix(rng.poisson(0.1, size=(10, 30)))
        sc = SoupChannel(tod=tod, toc=toc, calc_soup_profile=True)
        assert sc.meta_data['nUMIs'].iloc[0] > 0


# ── boundary rho values ───────────────────────────────────────────────────────

class TestBoundaryRho:
    def test_rho_zero_accepted(self, tiny_sc):
        sc = set_contamination_fraction(tiny_sc, 0.0)
        assert (sc.meta_data['rho'] == 0.0).all()

    def test_rho_one_raises_without_force(self, tiny_sc):
        with pytest.raises(ValueError, match="Extremely high"):
            set_contamination_fraction(tiny_sc, 1.0)

    def test_rho_one_with_force_accept(self, tiny_sc):
        sc = set_contamination_fraction(tiny_sc, 1.0, force_accept=True)
        assert (sc.meta_data['rho'] == 1.0).all()

    def test_rho_gt_one_always_raises(self, tiny_sc):
        with pytest.raises(ValueError, match="impossible"):
            set_contamination_fraction(tiny_sc, 1.01, force_accept=True)

    def test_rho_zero_adjust_counts_no_removal(self, tiny_sc):
        from SoupX import adjust_counts
        sc = set_contamination_fraction(tiny_sc, 0.0)
        out = adjust_counts(sc, clusters=False)
        # rho=0 → no counts removed
        np.testing.assert_array_equal(out.toarray(), tiny_sc.toc.toarray())


# ── empty / invalid gene lists ────────────────────────────────────────────────

class TestEmptyGeneLists:
    def test_empty_dict_returns_zero_columns(self, tiny_sc_with_clusters):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimate_non_expressing_cells(tiny_sc_with_clusters, {})
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (len(tiny_sc_with_clusters.cells), 0)

    def test_unknown_genes_vacuously_all_true(self, tiny_sc_with_clusters):
        """Gene not in sc.genes: zero obs, zero expected → p=1 → all cells pass."""
        result = estimate_non_expressing_cells(
            tiny_sc_with_clusters, {"phantom": ["NOSUCHGENE_XYZ_000"]}
        )
        assert result.shape == (len(tiny_sc_with_clusters.cells), 1)
        assert result["phantom"].all()

    def test_non_dict_raises(self, tiny_sc_with_clusters):
        with pytest.raises(TypeError):
            estimate_non_expressing_cells(tiny_sc_with_clusters, ["GeneA"])


# ── bug regression: all useEst=False silently wrong rho ──────────────────────

class TestAllUseEstFalseBug:
    """
    Regression for estimation.py — previously when all gene×cluster estimates
    were marked useEst=False, np.mean([]) returned NaN and np.argmax(NaN array)
    silently returned index 0 instead of raising, yielding wrong rho with no
    diagnostic. Fixed: raise ValueError before computing the posterior.
    """

    def test_raises_not_silent(self, tiny_sc_with_clusters, monkeypatch):
        import SoupX.estimation as _est

        def _all_false(sc, gene_list, **kwargs):
            return pd.DataFrame(False, index=sc.cells, columns=list(gene_list.keys()))

        monkeypatch.setattr(_est, 'estimate_non_expressing_cells', _all_false)

        with pytest.raises(ValueError, match="useEst=False"):
            auto_est_cont(
                tiny_sc_with_clusters,
                tfidf_min=0.0,
                soup_quantile=0.0,
                do_plot=False,
                verbose=False,
            )

    def test_error_message_actionable(self, tiny_sc_with_clusters, monkeypatch):
        import SoupX.estimation as _est

        def _all_false(sc, gene_list, **kwargs):
            return pd.DataFrame(False, index=sc.cells, columns=list(gene_list.keys()))

        monkeypatch.setattr(_est, 'estimate_non_expressing_cells', _all_false)

        with pytest.raises(ValueError) as exc_info:
            auto_est_cont(
                tiny_sc_with_clusters,
                tfidf_min=0.0,
                soup_quantile=0.0,
                do_plot=False,
                verbose=False,
            )
        msg = str(exc_info.value)
        # Message must suggest remediation
        assert any(
            kw in msg for kw in ("tfidf_min", "rho_max_fdr", "soup_quantile")
        ), f"Error message lacks remediation hint: {msg!r}"
