import os
import pytest
import numpy as np

from SoupX import load_10x, read_10x

TOY_DATA = os.path.join(os.path.dirname(__file__), "..", "datasets", "toyData")


@pytest.mark.skipif(not os.path.isdir(TOY_DATA), reason="toy data not present")
class TestRead10X:
    def test_read_10x_raw(self):
        raw_dir = os.path.join(TOY_DATA, "raw_gene_bc_matrices", "GRCh38")
        mat, genes, barcodes, feat_types = read_10x(raw_dir)
        assert mat.shape[0] == len(genes)
        assert mat.shape[1] == len(barcodes)
        assert mat.nnz > 0

    def test_read_10x_filtered(self):
        filt_dir = os.path.join(TOY_DATA, "filtered_gene_bc_matrices", "GRCh38")
        mat, genes, barcodes, feat_types = read_10x(filt_dir)
        assert mat.shape[0] == len(genes)
        assert mat.shape[1] == len(barcodes)

    def test_filtered_fewer_barcodes_than_raw(self):
        raw_dir = os.path.join(TOY_DATA, "raw_gene_bc_matrices", "GRCh38")
        filt_dir = os.path.join(TOY_DATA, "filtered_gene_bc_matrices", "GRCh38")
        _, _, raw_bc, _ = read_10x(raw_dir)
        _, _, filt_bc, _ = read_10x(filt_dir)
        assert len(filt_bc) <= len(raw_bc)


@pytest.mark.skipif(not os.path.isdir(TOY_DATA), reason="toy data not present")
class TestLoad10X:
    def test_returns_soup_channel(self):
        from SoupX import SoupChannel
        sc = load_10x(TOY_DATA)
        assert isinstance(sc, SoupChannel)

    def test_soup_profile_set(self):
        sc = load_10x(TOY_DATA)
        assert sc.soup_profile is not None
        assert np.isclose(sc.soup_profile["est"].sum(), 1.0)

    def test_genes_cells_populated(self):
        sc = load_10x(TOY_DATA)
        assert len(sc.genes) > 0
        assert len(sc.cells) > 0

    def test_meta_data_has_nUMIs(self):
        sc = load_10x(TOY_DATA)
        assert "nUMIs" in sc.meta_data.columns
        assert (sc.meta_data["nUMIs"] > 0).all()
