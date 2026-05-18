import numpy as np
import scipy.sparse
import pytest

from SoupX import adjust_counts


class TestAdjustCountsSubtraction:
    def test_returns_sparse_csc(self, tiny_sc_rho):
        out = adjust_counts(tiny_sc_rho, clusters=False)
        assert scipy.sparse.issparse(out)
        assert out.format == "csc"

    def test_preserves_shape(self, tiny_sc_rho):
        out = adjust_counts(tiny_sc_rho, clusters=False)
        assert out.shape == tiny_sc_rho.toc.shape

    def test_all_values_nonneg(self, tiny_sc_rho):
        out = adjust_counts(tiny_sc_rho, clusters=False)
        if out.nnz > 0:
            assert out.data.min() >= -1e-10

    def test_removes_counts(self, tiny_sc_rho):
        out = adjust_counts(tiny_sc_rho, clusters=False)
        assert out.sum() < tiny_sc_rho.toc.sum()

    def test_no_new_nonzero_positions(self, tiny_sc_rho):
        """Corrected matrix cannot gain new non-zero entries vs original."""
        out = adjust_counts(tiny_sc_rho, clusters=False)
        original_dense = tiny_sc_rho.toc.toarray()
        out_dense = out.toarray()
        assert np.all((out_dense > 0) <= (original_dense > 0))

    def test_no_rho_raises(self, tiny_sc):
        with pytest.raises(ValueError, match="rho"):
            adjust_counts(tiny_sc, clusters=False)

    def test_bad_method_raises(self, tiny_sc_rho):
        with pytest.raises(ValueError, match="Unknown method"):
            adjust_counts(tiny_sc_rho, clusters=False, method="magic")

    def test_souponly_works(self, tiny_sc_rho):
        out = adjust_counts(tiny_sc_rho, clusters=False, method="soupOnly")
        assert out.shape == tiny_sc_rho.toc.shape
        assert out.toarray().min() >= -1e-9
        assert out.sum() <= tiny_sc_rho.toc.sum() + 1e-6

    def test_souponly_unknown_method_still_raises(self, tiny_sc_rho):
        with pytest.raises(ValueError, match="Unknown method"):
            adjust_counts(tiny_sc_rho, clusters=False, method="bogus")


class TestAdjustCountsWithClusters:
    def test_cluster_aware_shape(self, tiny_sc_rho_clustered):
        out = adjust_counts(tiny_sc_rho_clustered)
        assert out.shape == tiny_sc_rho_clustered.toc.shape

    def test_cluster_aware_removes_counts(self, tiny_sc_rho_clustered):
        out = adjust_counts(tiny_sc_rho_clustered)
        assert out.sum() <= tiny_sc_rho_clustered.toc.sum() + 1e-6

    def test_cluster_aware_nonneg(self, tiny_sc_rho_clustered):
        out = adjust_counts(tiny_sc_rho_clustered)
        if out.nnz > 0:
            assert out.data.min() >= -1e-10

    def test_explicit_clusters_param(self, tiny_sc_rho):
        import numpy as np
        clusters = np.array(["A"] * 40 + ["B"] * 40)
        out = adjust_counts(tiny_sc_rho, clusters=clusters)
        assert out.shape == tiny_sc_rho.toc.shape

    def test_explicit_false_clusters(self, tiny_sc_rho):
        out = adjust_counts(tiny_sc_rho, clusters=False)
        assert out.shape == tiny_sc_rho.toc.shape



class TestRoundToInt:
    def test_output_integers(self, tiny_sc_rho):
        np.random.seed(0)
        out = adjust_counts(tiny_sc_rho, clusters=False, round_to_int=True)
        if out.nnz > 0:
            np.testing.assert_allclose(out.data, np.round(out.data), atol=1e-10)

    def test_shape_preserved(self, tiny_sc_rho):
        np.random.seed(0)
        out = adjust_counts(tiny_sc_rho, clusters=False, round_to_int=True)
        assert out.shape == tiny_sc_rho.toc.shape
