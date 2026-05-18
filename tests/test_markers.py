import numpy as np
import pandas as pd
import pytest

from SoupX.markers import quick_markers


EXPECTED_COLS = {
    "gene", "cluster", "geneFrequency", "geneFrequencyOutsideCluster",
    "geneFrequencySecondBest", "geneFrequencyGlobal", "secondBestClusterName",
    "tfidf", "idf", "qval",
}


def test_returns_dataframe(tiny_sc):
    clusters = np.array(["0"] * 40 + ["1"] * 40)
    df = quick_markers(tiny_sc.toc, clusters, genes=list(tiny_sc.genes))
    assert isinstance(df, pd.DataFrame)
    assert EXPECTED_COLS.issubset(df.columns)


def test_cluster_labels_in_output(tiny_sc):
    clusters = np.array(["A"] * 40 + ["B"] * 40)
    df = quick_markers(tiny_sc.toc, clusters, genes=list(tiny_sc.genes))
    assert set(df["cluster"].unique()).issubset({"A", "B"})


def test_marker_count_bounded_by_n(tiny_sc):
    clusters = np.array(["0"] * 40 + ["1"] * 40)
    n = 3
    df = quick_markers(tiny_sc.toc, clusters, genes=list(tiny_sc.genes), n=n)
    for cl in df["cluster"].unique():
        assert df[df["cluster"] == cl].shape[0] <= n


def test_cluster_specific_genes(tiny_sc):
    """Genes 0-4 are over-expressed in cells 0-39 → should be top markers for cluster 0."""
    clusters = np.array(["0"] * 40 + ["1"] * 40)
    df = quick_markers(tiny_sc.toc, clusters, genes=list(tiny_sc.genes), n=5)
    top_0 = df[df["cluster"] == "0"]["gene"].values
    gene_nums = [int(g.replace("Gene", "")) for g in top_0]
    assert any(g < 5 for g in gene_nums), f"No cluster-0 markers in top: {top_0}"


def test_frequencies_in_range(tiny_sc):
    clusters = np.array(["0"] * 40 + ["1"] * 40)
    df = quick_markers(tiny_sc.toc, clusters, genes=list(tiny_sc.genes))
    assert (df["geneFrequency"] >= 0).all()
    assert (df["geneFrequency"] <= 1).all()
    assert (df["geneFrequencyGlobal"] >= 0).all()
    assert (df["geneFrequencyGlobal"] <= 1).all()
    assert (df["qval"] >= 0).all()
    assert (df["qval"] <= 1).all()


def test_no_genes_arg(tiny_sc):
    clusters = np.array(["0"] * 40 + ["1"] * 40)
    df = quick_markers(tiny_sc.toc, clusters)
    assert len(df) >= 0
