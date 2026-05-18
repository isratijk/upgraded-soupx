import numpy as np
import pandas as pd
import scipy.sparse
import pytest

from SoupX import SoupChannel, set_clusters, set_contamination_fraction


def _make_sc(n_genes=25, n_cells=80, n_drops=200, seed=0):
    rng = np.random.default_rng(seed)
    genes = pd.Index([f"Gene{i}" for i in range(n_genes)])
    cells = pd.Index([f"CELL{i:04d}" for i in range(n_cells)])
    drop_barcodes = [f"DROP{i:05d}" for i in range(n_drops)]

    # Cell counts: background everywhere + cluster-specific signal
    toc_dense = rng.poisson(0.3, size=(n_genes, n_cells)).astype(float)
    # Genes 0-4: highly expressed in cells 0-39 (cluster A)
    toc_dense[:5, :40] += rng.poisson(4.0, size=(5, 40))
    # Genes 5-9: highly expressed in cells 40-79 (cluster B)
    toc_dense[5:10, 40:] += rng.poisson(4.0, size=(5, 40))
    toc = scipy.sparse.csc_matrix(toc_dense)

    # Empty droplets: very low UMIs (1-5), reflecting background soup
    tod_dense = rng.poisson(0.05, size=(n_genes, n_drops)).astype(float)
    tod = scipy.sparse.csc_matrix(tod_dense)

    return SoupChannel(
        tod=tod,
        toc=toc,
        genes=genes,
        cells=cells,
        drop_barcodes=drop_barcodes,
        calc_soup_profile=True,
    )


@pytest.fixture
def tiny_sc():
    return _make_sc()


@pytest.fixture
def tiny_sc_with_clusters():
    sc = _make_sc()
    clusters = np.array(["A"] * 40 + ["B"] * 40)
    return set_clusters(sc, clusters)


@pytest.fixture
def tiny_sc_rho():
    sc = _make_sc()
    return set_contamination_fraction(sc, 0.05)


@pytest.fixture
def tiny_sc_rho_clustered():
    sc = _make_sc()
    clusters = np.array(["A"] * 40 + ["B"] * 40)
    sc = set_clusters(sc, clusters)
    return set_contamination_fraction(sc, 0.05)
