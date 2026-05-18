"""
Tests for SoupX.downstream — uses synthetic data, no external datasets required.
"""

import warnings
import numpy as np
import pandas as pd
import scipy.sparse
import pytest

from SoupX.downstream import (
    normalize_log1p,
    run_pca,
    run_tsne,
    cluster_kmeans,
    differential_expression,
    score_cell_types,
    plot_embedding,
    plot_top_de_genes,
    run_downstream,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_GENES = 200
N_CELLS = 80
N_CLUSTERS = 4
RNG = np.random.default_rng(42)


@pytest.fixture(scope="module")
def synthetic_matrix():
    """Sparse genes × cells matrix with 4 distinct cluster blobs."""
    mat = np.zeros((N_GENES, N_CELLS))
    cells_per = N_CELLS // N_CLUSTERS
    for k in range(N_CLUSTERS):
        gene_start = k * (N_GENES // N_CLUSTERS)
        gene_end   = gene_start + (N_GENES // N_CLUSTERS)
        cell_start = k * cells_per
        cell_end   = cell_start + cells_per
        mat[gene_start:gene_end, cell_start:cell_end] = RNG.poisson(
            10, size=(gene_end - gene_start, cell_end - cell_start)
        )
    mat += RNG.poisson(0.5, size=mat.shape)   # sparse ambient noise
    return scipy.sparse.csc_matrix(mat.astype(int))


@pytest.fixture(scope="module")
def gene_names():
    return np.array([f"GENE{i:04d}" for i in range(N_GENES)])


@pytest.fixture(scope="module")
def true_labels():
    cells_per = N_CELLS // N_CLUSTERS
    return np.repeat(np.arange(N_CLUSTERS).astype(str), cells_per)


# ── normalize_log1p ───────────────────────────────────────────────────────────

class TestNormalizeLog1p:
    def test_output_shape(self, synthetic_matrix):
        out = normalize_log1p(synthetic_matrix)
        assert out.shape == (N_CELLS, N_GENES)

    def test_no_negatives(self, synthetic_matrix):
        out = normalize_log1p(synthetic_matrix)
        assert (out >= 0).all()

    def test_dense_input(self, synthetic_matrix):
        dense = synthetic_matrix.toarray()
        out = normalize_log1p(dense)
        assert out.shape == (N_CELLS, N_GENES)

    def test_zero_cell_handled(self, gene_names):
        mat = np.zeros((N_GENES, 5))
        mat[:, 1:] = 1
        out = normalize_log1p(mat)
        assert np.isfinite(out).all()


# ── run_pca ───────────────────────────────────────────────────────────────────

class TestRunPCA:
    def test_embedding_shape(self, synthetic_matrix, gene_names):
        result = run_pca(synthetic_matrix, gene_names, n_components=10)
        assert result['embedding'].shape == (N_CELLS, 10)

    def test_variance_ratio_sums_to_one(self, synthetic_matrix, gene_names):
        result = run_pca(synthetic_matrix, gene_names, n_components=10)
        assert result['variance_ratio'].sum() <= 1.0 + 1e-9

    def test_gene_names_returned(self, synthetic_matrix, gene_names):
        result = run_pca(synthetic_matrix, gene_names, n_components=5, n_top_genes=50)
        assert len(result['gene_names_used']) <= N_GENES

    def test_fewer_genes_than_top(self, synthetic_matrix, gene_names):
        result = run_pca(synthetic_matrix, gene_names, n_top_genes=N_GENES + 100)
        assert result['embedding'].shape[0] == N_CELLS


# ── run_tsne ──────────────────────────────────────────────────────────────────

class TestRunTSNE:
    def test_output_shape(self, synthetic_matrix, gene_names):
        pca = run_pca(synthetic_matrix, gene_names, n_components=10)
        emb = run_tsne(pca)
        assert emb.shape == (N_CELLS, 2)

    def test_accepts_array_directly(self, synthetic_matrix, gene_names):
        pca = run_pca(synthetic_matrix, gene_names, n_components=10)
        emb = run_tsne(pca['embedding'])
        assert emb.shape == (N_CELLS, 2)


# ── cluster_kmeans ────────────────────────────────────────────────────────────

class TestClusterKmeans:
    def test_returns_string_labels(self, synthetic_matrix, gene_names):
        pca = run_pca(synthetic_matrix, gene_names, n_components=10)
        labels = cluster_kmeans(pca, n_clusters=4)
        assert labels.dtype.kind in ('U', 'S', 'O')   # string dtype

    def test_correct_n_cells(self, synthetic_matrix, gene_names):
        pca = run_pca(synthetic_matrix, gene_names, n_components=10)
        labels = cluster_kmeans(pca, n_clusters=4)
        assert len(labels) == N_CELLS

    def test_n_clusters_capped_at_n_cells(self, synthetic_matrix, gene_names):
        pca = run_pca(synthetic_matrix, gene_names, n_components=10)
        labels = cluster_kmeans(pca, n_clusters=9999)
        assert len(np.unique(labels)) <= N_CELLS


# ── differential_expression ───────────────────────────────────────────────────

class TestDifferentialExpression:
    def test_returns_dataframe(self, synthetic_matrix, gene_names, true_labels):
        de = differential_expression(synthetic_matrix, gene_names, true_labels)
        assert isinstance(de, pd.DataFrame)

    def test_expected_columns(self, synthetic_matrix, gene_names, true_labels):
        de = differential_expression(synthetic_matrix, gene_names, true_labels)
        assert set(['cluster', 'gene', 'statistic', 'pvalue', 'log2fc', 'rank']).issubset(
            de.columns
        )

    def test_all_clusters_represented(self, synthetic_matrix, gene_names, true_labels):
        de = differential_expression(synthetic_matrix, gene_names, true_labels)
        assert set(de['cluster'].unique()) == set(true_labels)

    def test_top_genes_correct_markers(self, synthetic_matrix, gene_names, true_labels):
        de = differential_expression(synthetic_matrix, gene_names, true_labels)
        for k in range(N_CLUSTERS):
            top_genes = de[de['cluster'] == str(k)].nsmallest(5, 'rank')['gene'].tolist()
            gene_range = [f"GENE{i:04d}" for i in range(
                k * (N_GENES // N_CLUSTERS),
                (k + 1) * (N_GENES // N_CLUSTERS)
            )]
            assert any(g in gene_range for g in top_genes), (
                f"cluster {k} top genes {top_genes} not in expected range {gene_range[:3]}..."
            )

    def test_small_cluster_skipped(self, gene_names):
        mat = scipy.sparse.eye(N_GENES, N_CELLS, format='csc')
        labels = np.array(['A'] * 2 + ['B'] * (N_CELLS - 2))
        de = differential_expression(mat, gene_names, labels, min_cells=5)
        assert 'A' not in de['cluster'].values

    def test_empty_result_has_correct_columns(self, gene_names):
        mat = scipy.sparse.csc_matrix((N_GENES, 4))
        labels = np.array(['X'] * 4)
        de = differential_expression(mat, gene_names, labels, min_cells=5)
        assert list(de.columns) == ['cluster', 'gene', 'statistic', 'pvalue', 'log2fc', 'rank']


# ── score_cell_types ──────────────────────────────────────────────────────────

class TestScoreCellTypes:
    def test_returns_dataframe(self, synthetic_matrix, gene_names):
        marker_dict = {'T_cell': ['GENE0000', 'GENE0001'], 'B_cell': ['GENE0050']}
        scores = score_cell_types(synthetic_matrix, gene_names, marker_dict)
        assert isinstance(scores, pd.DataFrame)
        assert scores.shape == (N_CELLS, 2)

    def test_columns_match_marker_dict(self, synthetic_matrix, gene_names):
        marker_dict = {'TypeA': ['GENE0000'], 'TypeB': ['GENE0100']}
        scores = score_cell_types(synthetic_matrix, gene_names, marker_dict)
        assert set(scores.columns) == {'TypeA', 'TypeB'}

    def test_missing_markers_warns(self, synthetic_matrix, gene_names):
        marker_dict = {'Ghost': ['NOTEXIST1', 'NOTEXIST2']}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            score_cell_types(synthetic_matrix, gene_names, marker_dict)
            assert any('Ghost' in str(x.message) for x in w)

    def test_scores_nonnegative(self, synthetic_matrix, gene_names):
        marker_dict = {'T_cell': ['GENE0000', 'GENE0001']}
        scores = score_cell_types(synthetic_matrix, gene_names, marker_dict)
        assert (scores >= 0).all().all()


# ── plot_embedding ────────────────────────────────────────────────────────────

class TestPlotEmbedding:
    def test_returns_axes(self, synthetic_matrix, gene_names):
        import matplotlib
        matplotlib.use('Agg')
        pca = run_pca(synthetic_matrix, gene_names, n_components=10)
        emb = run_tsne(pca)
        ax = plot_embedding(emb)
        import matplotlib.pyplot as plt
        assert hasattr(ax, 'scatter')
        plt.close('all')

    def test_with_labels(self, synthetic_matrix, gene_names, true_labels):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        pca = run_pca(synthetic_matrix, gene_names, n_components=10)
        emb = run_tsne(pca)
        ax = plot_embedding(emb, labels=true_labels, title='test')
        assert ax.get_title() == 'test'
        plt.close('all')


# ── plot_top_de_genes ─────────────────────────────────────────────────────────

class TestPlotTopDeGenes:
    def test_returns_axes(self, synthetic_matrix, gene_names, true_labels):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        de = differential_expression(synthetic_matrix, gene_names, true_labels)
        ax = plot_top_de_genes(de, n_genes=5)
        assert ax is not None
        plt.close('all')

    def test_empty_df_warns(self):
        import matplotlib
        matplotlib.use('Agg')
        empty = pd.DataFrame(
            columns=['cluster', 'gene', 'statistic', 'pvalue', 'log2fc', 'rank']
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            plot_top_de_genes(empty)
            assert any('empty' in str(x.message).lower() for x in w)


# ── run_downstream ────────────────────────────────────────────────────────────

class TestRunDownstream:
    def test_full_pipeline_tsne_kmeans(self, synthetic_matrix, gene_names):
        result = run_downstream(
            synthetic_matrix, gene_names,
            n_pca=10, embedding='tsne', clustering='kmeans',
            n_clusters=4, run_de=True,
        )
        assert result['embedding'].shape == (N_CELLS, 2)
        assert len(result['cluster_labels']) == N_CELLS
        assert result['cluster_method'] == 'kmeans'
        assert isinstance(result['de_results'], pd.DataFrame)
        assert result['cell_type_scores'] is None

    def test_provided_labels_skip_clustering(self, synthetic_matrix, gene_names, true_labels):
        result = run_downstream(
            synthetic_matrix, gene_names,
            cluster_labels=true_labels,
            n_pca=10, embedding=None, clustering=None, run_de=False,
        )
        assert result['cluster_method'] == 'provided'
        assert result['embedding'] is None
        assert result['de_results'] is None

    def test_with_marker_dict(self, synthetic_matrix, gene_names):
        marker_dict = {'T_cell': ['GENE0000', 'GENE0001'], 'B_cell': ['GENE0050']}
        result = run_downstream(
            synthetic_matrix, gene_names,
            n_pca=10, embedding=None, clustering='kmeans',
            n_clusters=4, marker_dict=marker_dict, run_de=False,
        )
        assert result['cell_type_scores'] is not None
        assert result['cell_type_scores'].shape == (N_CELLS, 2)

    def test_no_embedding_no_clustering(self, synthetic_matrix, gene_names):
        result = run_downstream(
            synthetic_matrix, gene_names,
            n_pca=10, embedding=None, clustering=None, run_de=False,
        )
        assert result['embedding'] is None
        assert result['cluster_labels'] is None
        assert result['de_results'] is None

    def test_keys_always_present(self, synthetic_matrix, gene_names):
        result = run_downstream(
            synthetic_matrix, gene_names,
            n_pca=5, embedding=None, clustering=None, run_de=False,
        )
        for key in ('pca', 'embedding', 'embedding_method',
                    'cluster_labels', 'cluster_method',
                    'de_results', 'cell_type_scores'):
            assert key in result, f"missing key: {key}"
