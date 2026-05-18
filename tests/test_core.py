import numpy as np
import pandas as pd
import scipy.sparse
import pytest

from SoupX import SoupChannel, set_clusters, set_contamination_fraction, set_dr, set_soup_profile
from SoupX.estimate_soup import estimate_soup


class TestSoupChannel:
    def test_construction(self, tiny_sc):
        sc = tiny_sc
        assert sc.toc.shape[0] == len(sc.genes)
        assert sc.toc.shape[1] == len(sc.cells)
        assert "nUMIs" in sc.meta_data.columns
        assert sc.meta_data["nUMIs"].min() > 0

    def test_soup_profile_set_on_construction(self, tiny_sc):
        assert tiny_sc.soup_profile is not None
        assert "est" in tiny_sc.soup_profile.columns
        assert "counts" in tiny_sc.soup_profile.columns
        assert np.isclose(tiny_sc.soup_profile["est"].sum(), 1.0)

    def test_repr(self, tiny_sc):
        r = repr(tiny_sc)
        assert "SoupChannel" in r
        assert str(len(tiny_sc.genes)) in r

    def test_shape_mismatch_raises(self):
        tod = scipy.sparse.csc_matrix(np.ones((5, 10)))
        toc = scipy.sparse.csc_matrix(np.ones((6, 4)))
        with pytest.raises(ValueError, match="genes"):
            SoupChannel(tod=tod, toc=toc, calc_soup_profile=False)

    def test_tod_released_after_estimate(self, tiny_sc):
        assert tiny_sc.tod is None

    def test_copy(self, tiny_sc):
        sc2 = tiny_sc.copy()
        assert sc2 is not tiny_sc
        assert sc2.toc.shape == tiny_sc.toc.shape


class TestEstimateSoup:
    def test_estimate_soup_profile_sums_to_one(self):
        rng = np.random.default_rng(1)
        tod = scipy.sparse.csc_matrix(rng.poisson(0.1, size=(10, 100)))
        toc = scipy.sparse.csc_matrix(rng.poisson(2.0, size=(10, 20)))
        sc = SoupChannel(tod=tod, toc=toc, calc_soup_profile=True)
        assert np.isclose(sc.soup_profile["est"].sum(), 1.0)

    def test_estimate_soup_no_droplets_raises(self, tiny_sc):
        with pytest.raises(ValueError, match="tod is None"):
            estimate_soup(tiny_sc)

    def test_estimate_soup_empty_range_raises(self):
        rng = np.random.default_rng(2)
        tod = scipy.sparse.csc_matrix(rng.poisson(50.0, size=(10, 50)))
        toc = scipy.sparse.csc_matrix(rng.poisson(2.0, size=(10, 10)))
        sc = SoupChannel(tod=tod, toc=toc, calc_soup_profile=False)
        with pytest.raises(ValueError, match="No droplets"):
            estimate_soup(sc, soup_range=(0, 5))


class TestSetProperties:
    def test_set_clusters_array(self, tiny_sc):
        labels = np.array(["A"] * 40 + ["B"] * 40)
        sc = set_clusters(tiny_sc, labels)
        assert "clusters" in sc.meta_data.columns
        assert set(sc.meta_data["clusters"]) == {"A", "B"}

    def test_set_clusters_series(self, tiny_sc):
        s = pd.Series(
            ["X"] * 40 + ["Y"] * 40,
            index=tiny_sc.cells,
        )
        sc = set_clusters(tiny_sc, s)
        assert set(sc.meta_data["clusters"]) == {"X", "Y"}

    def test_set_clusters_length_mismatch_raises(self, tiny_sc):
        with pytest.raises(ValueError):
            set_clusters(tiny_sc, np.array(["A", "B"]))

    def test_set_contamination_fraction_scalar(self, tiny_sc):
        sc = set_contamination_fraction(tiny_sc, 0.05)
        assert "rho" in sc.meta_data.columns
        assert (sc.meta_data["rho"] == 0.05).all()

    def test_set_contamination_fraction_array(self, tiny_sc):
        rho = np.linspace(0.01, 0.1, len(tiny_sc.cells))
        sc = set_contamination_fraction(tiny_sc, rho)
        np.testing.assert_allclose(sc.meta_data["rho"].values, rho)

    def test_set_contamination_fraction_series(self, tiny_sc):
        rho = pd.Series(
            np.linspace(0.01, 0.1, len(tiny_sc.cells)),
            index=tiny_sc.cells,
        )
        sc = set_contamination_fraction(tiny_sc, rho)
        np.testing.assert_allclose(sc.meta_data["rho"].values, rho.values)

    def test_set_contamination_fraction_partial_series_raises(self, tiny_sc):
        rho = pd.Series([0.05] * (len(tiny_sc.cells) - 1), index=tiny_sc.cells[:-1])
        with pytest.raises(ValueError, match="must cover exactly the cells"):
            set_contamination_fraction(tiny_sc, rho)

    def test_set_contamination_fraction_single_cell_series_does_not_broadcast(self, tiny_sc):
        rho = pd.Series([0.05], index=[tiny_sc.cells[0]])
        with pytest.raises(ValueError, match="must cover exactly the cells"):
            set_contamination_fraction(tiny_sc, rho)

    def test_set_contamination_fraction_gt_one_raises(self, tiny_sc):
        with pytest.raises(ValueError, match="impossible"):
            set_contamination_fraction(tiny_sc, 1.5)

    def test_set_contamination_fraction_high_raises(self, tiny_sc):
        with pytest.raises(ValueError, match="Extremely high"):
            set_contamination_fraction(tiny_sc, 0.6)

    def test_set_contamination_fraction_force_accept(self, tiny_sc):
        sc = set_contamination_fraction(tiny_sc, 0.6, force_accept=True)
        assert (sc.meta_data["rho"] == 0.6).all()

    def test_set_dr(self, tiny_sc):
        coords = pd.DataFrame(
            np.random.randn(len(tiny_sc.cells), 2),
            index=tiny_sc.cells,
            columns=["UMAP1", "UMAP2"],
        )
        sc = set_dr(tiny_sc, coords, reduct_name="UMAP")
        assert "UMAP_1" in sc.meta_data.columns
        assert "UMAP_2" in sc.meta_data.columns
        assert sc.DR == ["UMAP_1", "UMAP_2"]

    def test_inplace_flag(self, tiny_sc):
        sc2 = set_contamination_fraction(tiny_sc, 0.05, inplace=False)
        assert sc2 is not tiny_sc
        sc3 = set_contamination_fraction(tiny_sc, 0.05, inplace=True)
        assert sc3 is tiny_sc

    def test_set_soup_profile_requires_all_genes(self, tiny_sc):
        profile = tiny_sc.soup_profile.iloc[:-1].copy()
        with pytest.raises(ValueError, match="must contain every gene"):
            set_soup_profile(tiny_sc, profile)

    def test_set_soup_profile_duplicate_gene_raises(self, tiny_sc):
        profile = tiny_sc.soup_profile.copy()
        profile = pd.concat([profile, profile.iloc[[0]]])
        with pytest.raises(ValueError, match="duplicate gene names"):
            set_soup_profile(tiny_sc, profile)
