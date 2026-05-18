import numpy as np
import pytest
import scipy.sparse
import pandas as pd

from SoupX.utils import alloc, expand_clusters


class TestAlloc:
    def test_no_caps_hit(self):
        result = alloc(2.0, [5.0, 5.0])
        assert np.allclose(result, [1.0, 1.0])
        assert np.isclose(result.sum(), 2.0)

    def test_uniform_weights_sum(self):
        result = alloc(3.0, [5.0, 5.0, 5.0])
        assert np.isclose(result.sum(), 3.0)
        assert np.allclose(result, 1.0)

    def test_one_cap_hit(self):
        result = alloc(3.0, [0.5, 5.0, 5.0])
        assert np.isclose(result[0], 0.5)
        assert np.isclose(result.sum(), 3.0)
        assert np.isclose(result[1], result[2])

    def test_explicit_weights(self):
        result = alloc(1.0, [5.0, 5.0], ws=np.array([0.25, 0.75]))
        assert np.allclose(result, [0.25, 0.75])

    def test_zero_target(self):
        result = alloc(0.0, [2.0, 2.0, 2.0])
        assert np.all(result == 0.0)

    def test_zero_weights_returns_zeros(self):
        result = alloc(3.0, [2.0, 2.0], ws=np.array([0.0, 0.0]))
        assert np.all(result == 0.0)

    def test_all_caps_hit(self):
        result = alloc(10.0, [1.0, 1.0, 1.0])
        assert np.allclose(result, [1.0, 1.0, 1.0])

    def test_tgt_exceeds_total_capacity(self):
        result = alloc(100.0, [1.0, 2.0, 3.0])
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_output_respects_caps(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            n = int(rng.integers(2, 8))
            lims = rng.uniform(0.1, 3.0, size=n)
            tgt = float(rng.uniform(0, lims.sum()))
            ws = rng.uniform(0.1, 1.0, size=n)
            result = alloc(tgt, lims, ws)
            assert np.all(result >= -1e-10), "negative values"
            assert np.all(result <= lims + 1e-10), "cap exceeded"
            assert np.isclose(result.sum(), tgt, atol=1e-8), "wrong total"


class TestExpandClusters:
    def test_basic_expansion(self):
        n_genes, n_cells = 5, 6
        clusters = np.array(["A", "A", "A", "B", "B", "B"])
        cells = pd.Index([f"C{i}" for i in range(n_cells)])
        genes = pd.Index([f"G{i}" for i in range(n_genes)])

        cell_obs = scipy.sparse.csc_matrix(
            np.ones((n_genes, n_cells), dtype=float)
        )
        clust_soup = pd.DataFrame(
            {"A": [1.0] * n_genes, "B": [2.0] * n_genes},
            index=genes,
        )
        weights = np.ones(n_cells)

        result = expand_clusters(clust_soup, cell_obs, clusters, weights, verbose=0)
        assert result.shape == (n_genes, n_cells)
        assert result.sum() > 0

    def test_expansion_sums_to_cluster_total(self):
        n_genes, n_cells = 4, 4
        clusters = np.array(["X", "X", "Y", "Y"])
        genes = pd.Index([f"G{i}" for i in range(n_genes)])

        toc_dense = np.array([
            [2.0, 3.0, 1.0, 4.0],
            [1.0, 1.0, 2.0, 2.0],
            [0.0, 2.0, 0.0, 1.0],
            [3.0, 1.0, 2.0, 0.0],
        ])
        cell_obs = scipy.sparse.csc_matrix(toc_dense)
        clust_soup = pd.DataFrame(
            {
                "X": [1.5, 0.5, 0.8, 1.0],
                "Y": [0.8, 1.0, 0.0, 0.6],
            },
            index=genes,
        )
        weights = np.array([1.0, 1.0, 1.0, 1.0])

        result = expand_clusters(clust_soup, cell_obs, clusters, weights, verbose=0)
        result_dense = result.toarray()

        # Sum over cells in cluster X should equal clust_soup["X"] (where <= row_sum)
        x_sum = result_dense[:, :2].sum(axis=1)
        y_sum = result_dense[:, 2:].sum(axis=1)
        x_cap = toc_dense[:, :2].sum(axis=1)
        y_cap = toc_dense[:, 2:].sum(axis=1)
        np.testing.assert_allclose(
            x_sum, np.minimum(clust_soup["X"].values, x_cap), atol=1e-8
        )
        np.testing.assert_allclose(
            y_sum, np.minimum(clust_soup["Y"].values, y_cap), atol=1e-8
        )
