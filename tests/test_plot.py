import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import scipy.sparse
import pytest

from SoupX import (
    SoupChannel,
    set_clusters,
    set_contamination_fraction,
    set_dr,
    adjust_counts,
    plot_marker_map,
    plot_change_map,
)
from SoupX.plot import plot_marker_map, plot_change_map


@pytest.fixture
def sc_with_dr():
    rng = np.random.default_rng(42)
    n_genes, n_cells, n_drops = 20, 40, 100

    genes = pd.Index([f"Gene{i}" for i in range(n_genes)])
    cells = pd.Index([f"CELL{i:04d}" for i in range(n_cells)])

    toc_dense = rng.poisson(0.5, size=(n_genes, n_cells)).astype(float)
    toc_dense[:5, :20] += rng.poisson(3.0, size=(5, 20))
    toc_dense[5:10, 20:] += rng.poisson(3.0, size=(5, 20))

    tod_dense = rng.poisson(0.05, size=(n_genes, n_drops)).astype(float)

    sc = SoupChannel(
        tod=scipy.sparse.csc_matrix(tod_dense),
        toc=scipy.sparse.csc_matrix(toc_dense),
        genes=genes,
        cells=cells,
        drop_barcodes=[f"DROP{i:05d}" for i in range(n_drops)],
        calc_soup_profile=True,
    )
    sc = set_contamination_fraction(sc, 0.05)

    dr_coords = pd.DataFrame(
        rng.standard_normal((n_cells, 2)),
        index=cells,
        columns=['UMAP1', 'UMAP2'],
    )
    sc = set_dr(sc, dr_coords, reduct_name='UMAP')
    return sc


class TestPlotMarkerMap:
    def test_returns_figure(self, sc_with_dr):
        import matplotlib.figure
        fig = plot_marker_map(sc_with_dr, gene_set='Gene0')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_list_genes(self, sc_with_dr):
        import matplotlib.figure
        fig = plot_marker_map(sc_with_dr, gene_set=['Gene0', 'Gene1'])
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_explicit_dr(self, sc_with_dr):
        import matplotlib.figure
        dr = sc_with_dr.meta_data[['UMAP_1', 'UMAP_2']]
        fig = plot_marker_map(sc_with_dr, gene_set='Gene0', dr=dr)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_no_dr_raises(self, sc_with_dr):
        sc_with_dr.DR = None
        sc_with_dr.meta_data.drop(columns=['UMAP1', 'UMAP2'], errors='ignore', inplace=True)
        with pytest.raises(ValueError, match="dimension reduction"):
            plot_marker_map(sc_with_dr, gene_set='Gene0')

    def test_unknown_gene_raises(self, sc_with_dr):
        with pytest.raises(ValueError, match="found"):
            plot_marker_map(sc_with_dr, gene_set='NONEXISTENT_GENE')

    def test_use_to_est_override(self, sc_with_dr):
        import matplotlib.figure
        sig = np.zeros(len(sc_with_dr.cells), dtype=bool)
        sig[:5] = True
        fig = plot_marker_map(sc_with_dr, gene_set='Gene0', use_to_est=sig)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestPlotChangeMap:
    def test_soupfrac_returns_figure(self, sc_with_dr):
        import matplotlib.figure
        cleaned = adjust_counts(sc_with_dr, clusters=False)
        fig = plot_change_map(sc_with_dr, cleaned, gene_set='Gene0', data_type='soupFrac')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_binary_returns_figure(self, sc_with_dr):
        import matplotlib.figure
        cleaned = adjust_counts(sc_with_dr, clusters=False)
        fig = plot_change_map(sc_with_dr, cleaned, gene_set='Gene0', data_type='binary')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_counts_returns_figure(self, sc_with_dr):
        import matplotlib.figure
        cleaned = adjust_counts(sc_with_dr, clusters=False)
        fig = plot_change_map(sc_with_dr, cleaned, gene_set='Gene0', data_type='counts')
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_explicit_dr(self, sc_with_dr):
        import matplotlib.figure
        cleaned = adjust_counts(sc_with_dr, clusters=False)
        dr = sc_with_dr.meta_data[['UMAP_1', 'UMAP_2']]
        fig = plot_change_map(sc_with_dr, cleaned, gene_set='Gene0', dr=dr)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_no_dr_raises(self, sc_with_dr):
        sc_with_dr.DR = None
        sc_with_dr.meta_data.drop(columns=['UMAP1', 'UMAP2'], errors='ignore', inplace=True)
        cleaned = adjust_counts(sc_with_dr, clusters=False)
        with pytest.raises(ValueError, match="dimension reduction"):
            plot_change_map(sc_with_dr, cleaned, gene_set='Gene0')

    def test_unknown_gene_raises(self, sc_with_dr):
        cleaned = adjust_counts(sc_with_dr, clusters=False)
        with pytest.raises(ValueError, match="found"):
            plot_change_map(sc_with_dr, cleaned, gene_set='NONEXISTENT_GENE')

    def test_log_data_soupfrac(self, sc_with_dr):
        import matplotlib.figure
        cleaned = adjust_counts(sc_with_dr, clusters=False)
        fig = plot_change_map(sc_with_dr, cleaned, gene_set='Gene0',
                              data_type='soupFrac', log_data=True)
        assert isinstance(fig, matplotlib.figure.Figure)
