import numpy as np
import pytest
import warnings

from SoupX import (
    set_contamination_fraction,
    estimate_non_expressing_cells,
    calculate_contamination_fraction,
)


class TestEstimateNonExpressingCells:
    def test_returns_bool_dataframe(self, tiny_sc_with_clusters):
        sc = tiny_sc_with_clusters
        gene_list = {"GroupA": list(sc.genes[:3])}
        result = estimate_non_expressing_cells(sc, gene_list)
        assert result.shape == (len(sc.cells), 1)
        assert result.dtypes.iloc[0] == bool

    def test_columns_match_gene_list_keys(self, tiny_sc_with_clusters):
        sc = tiny_sc_with_clusters
        gene_list = {"HB": [sc.genes[0]], "IG": [sc.genes[1]]}
        result = estimate_non_expressing_cells(sc, gene_list)
        assert set(result.columns) == {"HB", "IG"}

    def test_non_dict_raises(self, tiny_sc_with_clusters):
        sc = tiny_sc_with_clusters
        with pytest.raises(TypeError):
            estimate_non_expressing_cells(sc, [sc.genes[0]])

    def test_background_genes_mostly_passing(self, tiny_sc_with_clusters):
        """Background-only genes (10-24) should allow many cells to pass."""
        sc = tiny_sc_with_clusters
        bg_genes = list(sc.genes[10:15])
        gene_list = {"BG": bg_genes}
        result = estimate_non_expressing_cells(
            sc, gene_list, maximum_contamination=1.0, fdr=0.05
        )
        assert result["BG"].sum() > 0

    def test_cluster_level_propagation(self, tiny_sc_with_clusters):
        """All cells in a cluster should have the same pass/fail for each gene set."""
        sc = tiny_sc_with_clusters
        gene_list = {"BG": list(sc.genes[10:13])}
        result = estimate_non_expressing_cells(sc, gene_list)
        clusters = sc.meta_data["clusters"]
        for cl in clusters.unique():
            mask = clusters == cl
            assert result.loc[mask, "BG"].nunique() == 1


class TestCalculateContaminationFraction:
    def test_returns_rho(self, tiny_sc_with_clusters):
        sc = tiny_sc_with_clusters
        gene_list = {"BG": list(sc.genes[15:])}
        ute = estimate_non_expressing_cells(sc, gene_list)
        if ute["BG"].sum() == 0:
            pytest.skip("No non-expressing cells found with these fixtures")
        sc_fit = calculate_contamination_fraction(
            sc, gene_list, ute, verbose=False, force_accept=True
        )
        assert "rho" in sc_fit.meta_data.columns
        rho = sc_fit.meta_data["rho"].iloc[0]
        assert 0 < rho < 1

    def test_rho_low_high_set(self, tiny_sc_with_clusters):
        sc = tiny_sc_with_clusters
        gene_list = {"BG": list(sc.genes[15:])}
        ute = estimate_non_expressing_cells(sc, gene_list)
        if ute["BG"].sum() == 0:
            pytest.skip("No non-expressing cells found")
        sc_fit = calculate_contamination_fraction(
            sc, gene_list, ute, verbose=False, force_accept=True
        )
        assert "rhoLow" in sc_fit.meta_data.columns
        assert "rhoHigh" in sc_fit.meta_data.columns
        rho = sc_fit.meta_data["rho"].iloc[0]
        rho_low = sc_fit.meta_data["rhoLow"].iloc[0]
        rho_high = sc_fit.meta_data["rhoHigh"].iloc[0]
        assert rho_low <= rho <= rho_high

    def test_no_valid_cells_raises(self, tiny_sc_with_clusters):
        import pandas as pd
        sc = tiny_sc_with_clusters
        gene_list = {"BG": list(sc.genes[15:])}
        ute = estimate_non_expressing_cells(sc, gene_list)
        empty_ute = ute.copy()
        empty_ute.iloc[:] = False
        with pytest.raises(ValueError, match="No cells"):
            calculate_contamination_fraction(sc, gene_list, empty_ute, verbose=False)

    def test_inplace_false_returns_copy(self, tiny_sc_with_clusters):
        sc = tiny_sc_with_clusters
        gene_list = {"BG": list(sc.genes[15:])}
        ute = estimate_non_expressing_cells(sc, gene_list)
        if ute["BG"].sum() == 0:
            pytest.skip("No non-expressing cells found")
        sc_fit = calculate_contamination_fraction(
            sc, gene_list, ute, verbose=False, force_accept=True, inplace=False
        )
        assert sc_fit is not sc
        assert "rho" not in sc.meta_data.columns
