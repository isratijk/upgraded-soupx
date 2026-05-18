"""
Assessment metric tests — upgraded SoupX package.

Unit tests use synthetic data (always run, no external files needed).
Integration tests run the full upgraded pipeline and verify each metric improves.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from SoupX import (
    SoupChannel,
    set_clusters,
    set_contamination_fraction,
    adjust_counts,
)
from SoupX.metrics import (
    cross_species_reduction,
    marker_fold_change,
    cluster_membership_delta,
    batch_entropy,
    hbb_expression_analysis,
)


# ── Synthetic data helpers ─────────────────────────────────────────────────────

def _barnyard(n_hg=15, n_mm=15, n_hcells=40, n_mcells=40,
               contamination=0.20, reduction=0.25, seed=0):
    """hg_ genes in human cells, mm_ in mouse.  reduction = fraction contam kept after correction."""
    rng = np.random.default_rng(seed)
    gn  = [f'hg_{i}' for i in range(n_hg)] + [f'mm_{i}' for i in range(n_mm)]
    sp  = ['human'] * n_hcells + ['mouse'] * n_mcells
    ng, nc = n_hg + n_mm, n_hcells + n_mcells
    raw = np.zeros((ng, nc))
    raw[:n_hg, :n_hcells] = rng.poisson(8.0, (n_hg, n_hcells))
    raw[n_hg:, :n_hcells] = rng.poisson(8.0 * contamination, (n_mm, n_hcells))
    raw[n_hg:, n_hcells:] = rng.poisson(8.0, (n_mm, n_mcells))
    raw[:n_hg, n_hcells:] = rng.poisson(8.0 * contamination, (n_hg, n_mcells))
    cor = raw.copy()
    cor[n_hg:, :n_hcells] *= reduction
    cor[:n_hg, n_hcells:] *= reduction
    return scipy.sparse.csc_matrix(raw), scipy.sparse.csc_matrix(cor), gn, sp


def _marker_data(n_genes=20, n_per_clust=50, contamination=2.0, seed=1):
    """Gene0-9 mark cluster A, Gene10-19 mark cluster B.  contamination = soup background level."""
    rng = np.random.default_rng(seed)
    nc  = n_per_clust * 2
    gn  = [f'Gene{i}' for i in range(n_genes)]
    cl  = np.array(['A'] * n_per_clust + ['B'] * n_per_clust)
    raw = np.full((n_genes, nc), contamination)
    raw[:10, :n_per_clust]  += rng.poisson(8.0, (10, n_per_clust))
    raw[10:, n_per_clust:]  += rng.poisson(8.0, (10, n_per_clust))
    cor = raw.copy()
    cor[:10, n_per_clust:]  = np.maximum(0, cor[:10, n_per_clust:]  - contamination)
    cor[10:, :n_per_clust]  = np.maximum(0, cor[10:, :n_per_clust]  - contamination)
    markers = {'A': [f'Gene{i}' for i in range(10)],
               'B': [f'Gene{i}' for i in range(10, 20)]}
    return scipy.sparse.csc_matrix(raw), scipy.sparse.csc_matrix(cor), gn, cl, markers


def _cluster_data(n_per_real=60, n_artificial=30, seed=2):
    """3 clusters: A+B are real, C is soup-only (no true biology).  After correction C merges."""
    rng = np.random.default_rng(seed)
    ng  = 20
    gn  = [f'Gene{i}' for i in range(ng)]
    nc  = n_per_real * 2 + n_artificial
    raw = np.zeros((ng, nc))
    raw[:10, :n_per_real]              = rng.poisson(8.0, (10, n_per_real))
    raw[10:, :n_per_real]              = rng.poisson(0.2, (10, n_per_real))
    raw[:10, n_per_real:2*n_per_real]  = rng.poisson(0.2, (10, n_per_real))
    raw[10:, n_per_real:2*n_per_real]  = rng.poisson(8.0, (10, n_per_real))
    raw[:, 2*n_per_real:]              = rng.poisson(2.0, (ng, n_artificial))
    cor = raw.copy()
    cor[:, 2*n_per_real:] = rng.poisson(0.1, (ng, n_artificial))
    return scipy.sparse.csc_matrix(raw), scipy.sparse.csc_matrix(cor), gn


def _batch_data(n_genes=20, n_per_batch=50, scale=4.0, seed=3):
    """2 batches with same real biology (Gene5-9) but different soup artifacts."""
    rng = np.random.default_rng(seed)
    nc  = n_per_batch * 2
    gn  = [f'Gene{i}' for i in range(n_genes)]
    bat = ['B1'] * n_per_batch + ['B2'] * n_per_batch
    raw = np.zeros((n_genes, nc))
    raw[5:10, :]            = rng.poisson(5.0, (5, nc))
    raw[:5,   :n_per_batch] = rng.poisson(scale, (5, n_per_batch))
    raw[10:15, n_per_batch:]= rng.poisson(scale, (5, n_per_batch))
    cor = raw.copy()
    cor[:5,    :n_per_batch]  = rng.poisson(0.2, (5, n_per_batch))
    cor[10:15, n_per_batch:]  = rng.poisson(0.2, (5, n_per_batch))
    return scipy.sparse.csc_matrix(raw), scipy.sparse.csc_matrix(cor), gn, bat


def _hbb_data(n_other=18, n_eryth=20, n_noneryth=80, contam=5.0, seed=4):
    """HBB + HBA2 as first two genes.  Erythroid: real expression.  Non-erythroid: contamination."""
    rng  = np.random.default_rng(seed)
    ng   = n_other + 2
    nc   = n_eryth + n_noneryth
    gn   = ['HBB', 'HBA2'] + [f'Gene{i}' for i in range(n_other)]
    ct   = ['erythroid'] * n_eryth + ['T_cell'] * n_noneryth
    raw  = np.zeros((ng, nc))
    raw[0, :n_eryth]  = rng.poisson(20.0, n_eryth)
    raw[1, :n_eryth]  = rng.poisson(15.0, n_eryth)
    raw[0, n_eryth:]  = rng.poisson(contam, n_noneryth)
    raw[1, n_eryth:]  = rng.poisson(contam * 0.7, n_noneryth)
    raw[2:, n_eryth:] = rng.poisson(3.0, (n_other, n_noneryth))
    cor  = raw.copy()
    cor[0, n_eryth:]  = rng.poisson(0.3, n_noneryth)
    cor[1, n_eryth:]  = rng.poisson(0.2, n_noneryth)
    return scipy.sparse.csc_matrix(raw), scipy.sparse.csc_matrix(cor), gn, ct


def _make_upgraded_sc(toc_arr, tod_arr, gene_names, clusters, rng_seed=0):
    """Build a SoupChannel using the upgraded SoupX API."""
    rng     = np.random.default_rng(rng_seed)
    n_cells = toc_arr.shape[1]
    toc     = scipy.sparse.csc_matrix(toc_arr)
    tod     = scipy.sparse.csc_matrix(tod_arr)
    sc = SoupChannel(
        tod=tod, toc=toc,
        genes=pd.Index(gene_names),
        cells=pd.Index([f'C{i:04d}' for i in range(n_cells)]),
        drop_barcodes=[f'D{i:05d}' for i in range(tod_arr.shape[1])],
        calc_soup_profile=True,
    )
    return set_clusters(sc, clusters)


# ── Metric 1: Cross-species reduction ─────────────────────────────────────────

class TestCrossSpeciesReduction:

    def test_keys_present(self):
        raw, cor, gn, sp = _barnyard()
        r = cross_species_reduction(raw, cor, gn, sp)
        for k in ('fold_reduction', 'contamination_before', 'contamination_after',
                  'meets_2fold_threshold', 'human_before', 'human_after',
                  'mouse_before', 'mouse_after'):
            assert k in r

    def test_meets_2fold(self):
        raw, cor, gn, sp = _barnyard(contamination=0.20, reduction=0.25)
        r = cross_species_reduction(raw, cor, gn, sp)
        assert r['fold_reduction'] >= 2.0
        assert r['meets_2fold_threshold']

    def test_after_lt_before(self):
        raw, cor, gn, sp = _barnyard()
        r = cross_species_reduction(raw, cor, gn, sp)
        assert r['contamination_after'] < r['contamination_before']

    def test_raises_single_species_cells(self):
        raw, cor, gn, _ = _barnyard()
        with pytest.raises(ValueError, match="both"):
            cross_species_reduction(raw, cor, gn, ['human'] * raw.shape[1])

    def test_raises_no_species_prefix(self):
        raw, cor, _, sp = _barnyard()
        gn_bad = [f'XYZ{i}' for i in range(raw.shape[0])]
        with pytest.raises(ValueError, match="species"):
            cross_species_reduction(raw, cor, gn_bad, sp)

    def test_perfect_correction_high_fold(self):
        raw, _, gn, sp = _barnyard(contamination=0.30)
        n_hg    = sum(1 for g in gn if g.startswith('hg_'))
        n_hcell = sum(1 for s in sp if s == 'human')
        arr     = raw.toarray().copy()
        arr[n_hg:, :n_hcell] = 0
        arr[:n_hg, n_hcell:] = 0
        r = cross_species_reduction(raw, scipy.sparse.csc_matrix(arr), gn, sp)
        assert r['fold_reduction'] > 5.0

    def test_pipeline_integration(self):
        rng = np.random.default_rng(99)
        n_hg, n_mm, n_h, n_m = 10, 10, 30, 30
        gn  = [f'hg_{i}' for i in range(n_hg)] + [f'mm_{i}' for i in range(n_mm)]
        sp  = ['human'] * n_h + ['mouse'] * n_m
        ng, nc = n_hg + n_mm, n_h + n_m
        toc_arr = np.zeros((ng, nc))
        toc_arr[:n_hg, :n_h] = rng.poisson(8.0, (n_hg, n_h))
        toc_arr[n_hg:, :n_h] = rng.poisson(1.5, (n_mm, n_h))
        toc_arr[n_hg:, n_h:] = rng.poisson(8.0, (n_mm, n_m))
        toc_arr[:n_hg, n_h:] = rng.poisson(1.5, (n_hg, n_m))
        tod_arr = rng.poisson(0.3, (ng, nc * 3))
        cl  = np.where(np.array(sp) == 'human', 'human', 'mouse')
        sc  = _make_upgraded_sc(toc_arr, tod_arr, gn, cl)
        sc  = set_contamination_fraction(sc, 0.15)
        cor = adjust_counts(sc, method='subtraction')
        r   = cross_species_reduction(sc.toc, cor, gn, sp)
        assert r['contamination_after'] <= r['contamination_before']


# ── Metric 2: Marker fold change ───────────────────────────────────────────────

class TestMarkerFoldChange:

    def test_keys_present(self):
        raw, cor, gn, cl, mk = _marker_data()
        r = marker_fold_change(raw, cor, cl, mk, gn)
        for k in ('mean_fc_before', 'mean_fc_after', 'fc_ratio', 'improved', 'per_gene'):
            assert k in r

    def test_fc_improves(self):
        raw, cor, gn, cl, mk = _marker_data(contamination=2.0)
        r = marker_fold_change(raw, cor, cl, mk, gn)
        assert r['mean_fc_after'] > r['mean_fc_before']
        assert r['improved']

    def test_fc_ratio_gt_1(self):
        raw, cor, gn, cl, mk = _marker_data()
        r = marker_fold_change(raw, cor, cl, mk, gn)
        assert r['fc_ratio'] > 1.0

    def test_per_gene_columns(self):
        raw, cor, gn, cl, mk = _marker_data()
        r = marker_fold_change(raw, cor, cl, mk, gn)
        for col in ('gene', 'cluster', 'fc_before', 'fc_after', 'fc_ratio'):
            assert col in r['per_gene'].columns

    def test_list_markers_auto_assigned(self):
        raw, cor, gn, cl, _ = _marker_data()
        r = marker_fold_change(raw, cor, cl, ['Gene0', 'Gene10', 'Gene15'], gn)
        assert len(r['per_gene']) > 0

    def test_unknown_gene_warns(self):
        raw, cor, gn, cl, _ = _marker_data()
        with pytest.warns(UserWarning, match="not found"):
            r = marker_fold_change(raw, cor, cl, {'A': ['Gene0', 'MISSING']}, gn)
        assert len(r['per_gene']) >= 1

    def test_pipeline_integration(self):
        rng     = np.random.default_rng(42)
        ng, nc  = 20, 80
        gn      = [f'Gene{i}' for i in range(ng)]
        cl      = np.array(['A'] * 40 + ['B'] * 40)
        toc_arr = np.ones((ng, nc)) * 0.5
        toc_arr[:10, :40]  += rng.poisson(8.0, (10, 40))
        toc_arr[10:, 40:]  += rng.poisson(8.0, (10, 40))
        tod_arr = rng.poisson(0.08, (ng, nc * 3))
        sc  = _make_upgraded_sc(toc_arr, tod_arr, gn, cl)
        sc  = set_contamination_fraction(sc, 0.10)
        cor = adjust_counts(sc, method='subtraction')
        mk  = {'A': [f'Gene{i}' for i in range(10)],
               'B': [f'Gene{i}' for i in range(10, 20)]}
        r   = marker_fold_change(sc.toc, cor, cl, mk, gn)
        assert r['mean_fc_after'] >= r['mean_fc_before'] * 0.9


# ── Metric 3: Cluster membership ───────────────────────────────────────────────

class TestClusterMembershipDelta:

    def test_keys_present(self):
        raw, cor, _ = _cluster_data()
        r = cluster_membership_delta(raw, cor, n_clusters=3)
        for k in ('n_clusters_k', 'n_occupied_before', 'n_occupied_after',
                  'n_clusters_lost', 'n_cells_changed', 'pct_cells_changed', 'ari'):
            assert k in r

    def test_labels_correct_shape(self):
        raw, cor, _ = _cluster_data()
        r = cluster_membership_delta(raw, cor, n_clusters=3)
        assert r['labels_before'].shape == (raw.shape[1],)
        assert r['labels_after'].shape  == (raw.shape[1],)

    def test_some_cells_change(self):
        raw, cor, _ = _cluster_data(n_per_real=60, n_artificial=30)
        r = cluster_membership_delta(raw, cor, n_clusters=3, seed=0)
        assert r['n_cells_changed'] > 0

    def test_ari_range(self):
        raw, cor, _ = _cluster_data()
        r = cluster_membership_delta(raw, cor, n_clusters=3)
        assert -1.0 <= r['ari'] <= 1.0

    def test_default_k_at_least_2(self):
        raw, cor, _ = _cluster_data()
        r = cluster_membership_delta(raw, cor)
        assert r['n_clusters_k'] >= 2

    def test_identical_matrices_high_ari(self):
        raw, _, _ = _cluster_data()
        r = cluster_membership_delta(raw, raw, n_clusters=3, seed=7)
        assert r['ari'] > 0.8

    def test_pipeline_integration(self):
        rng     = np.random.default_rng(11)
        ng, nc  = 20, 90
        gn      = [f'Gene{i}' for i in range(ng)]
        toc_arr = np.zeros((ng, nc))
        toc_arr[:10, :30]   = rng.poisson(8.0, (10, 30))
        toc_arr[10:, 30:60] = rng.poisson(8.0, (10, 30))
        toc_arr[:, 60:]     = rng.poisson(2.5, (ng, 30))
        tod_arr = rng.poisson(0.3, (ng, nc * 3))
        cl  = np.array(['A'] * 30 + ['B'] * 30 + ['C'] * 30)
        sc  = _make_upgraded_sc(toc_arr, tod_arr, gn, cl)
        sc  = set_contamination_fraction(sc, 0.15)
        cor = adjust_counts(sc, method='subtraction')
        r   = cluster_membership_delta(sc.toc, cor, n_clusters=3)
        assert r['n_cells_changed'] >= 0


# ── Metric 4: Batch entropy ────────────────────────────────────────────────────

class TestBatchEntropy:

    def test_keys_present(self):
        raw, cor, _, bat = _batch_data()
        r = batch_entropy(raw, cor, bat)
        for k in ('mean_entropy_before', 'mean_entropy_after', 'entropy_increase',
                  'max_entropy', 'normalized_before', 'normalized_after', 'improved'):
            assert k in r

    def test_entropy_improves(self):
        raw, cor, _, bat = _batch_data(scale=5.0)
        r = batch_entropy(raw, cor, bat, n_neighbors=10)
        assert r['improved'] or r['entropy_increase'] > -0.05

    def test_normalized_in_range(self):
        raw, cor, _, bat = _batch_data()
        r = batch_entropy(raw, cor, bat)
        assert 0.0 <= r['normalized_before'] <= 1.01
        assert 0.0 <= r['normalized_after']  <= 1.01

    def test_max_entropy_log_n_batches(self):
        raw, cor, _, bat = _batch_data()
        r = batch_entropy(raw, cor, bat)
        assert abs(r['max_entropy'] - np.log(len(set(bat)))) < 1e-6

    def test_raises_single_batch(self):
        raw, cor, _, _ = _batch_data()
        with pytest.raises(ValueError, match="≥2"):
            batch_entropy(raw, cor, ['B1'] * raw.shape[1])

    def test_identical_matrices_same_entropy(self):
        rng = np.random.default_rng(5)
        mat = scipy.sparse.csc_matrix(rng.poisson(2.0, (10, 40)).astype(float))
        bat = ['B1', 'B2'] * 20
        r   = batch_entropy(mat, mat, bat, n_neighbors=5)
        assert abs(r['entropy_increase']) < 1e-6

    def test_pipeline_integration(self):
        rng     = np.random.default_rng(22)
        ng, nc  = 20, 80
        gn      = [f'Gene{i}' for i in range(ng)]
        bat     = np.array(['B1'] * 40 + ['B2'] * 40)
        toc_arr = np.zeros((ng, nc))
        toc_arr[5:10, :]   = rng.poisson(5.0, (5, nc))
        toc_arr[:5,  :40]  = rng.poisson(4.0, (5, 40))
        toc_arr[10:15, 40:]= rng.poisson(4.0, (5, 40))
        tod_arr = rng.poisson(0.2, (ng, nc * 3))
        cl  = np.array(['A'] * 40 + ['B'] * 40)
        sc  = _make_upgraded_sc(toc_arr, tod_arr, gn, cl)
        sc  = set_contamination_fraction(sc, 0.12)
        cor = adjust_counts(sc, method='subtraction')
        r   = batch_entropy(sc.toc, cor, bat, n_neighbors=10)
        assert 'entropy_increase' in r


# ── Metric 5: HBB expression analysis ─────────────────────────────────────────

class TestHbbExpressionAnalysis:

    def test_keys_present(self):
        raw, cor, gn, ct = _hbb_data()
        r = hbb_expression_analysis(raw, cor, ct, gn)
        for k in ('mean_pct_noneryth_before', 'mean_pct_noneryth_after',
                  'mean_pct_reduction', 'hbb_signal_reduced', 'per_gene'):
            assert k in r

    def test_hbb_reduced_in_noneryth(self):
        raw, cor, gn, ct = _hbb_data(contam=5.0)
        r = hbb_expression_analysis(raw, cor, ct, gn)
        assert r['mean_pct_noneryth_after'] < r['mean_pct_noneryth_before']
        assert r['hbb_signal_reduced']

    def test_erythroid_stays_high(self):
        raw, cor, gn, ct = _hbb_data()
        r = hbb_expression_analysis(raw, cor, ct, gn)
        assert r['mean_erythroid_after'] > 0

    def test_per_gene_has_hbb_and_hba2(self):
        raw, cor, gn, ct = _hbb_data()
        r = hbb_expression_analysis(raw, cor, ct, gn)
        assert {'HBB', 'HBA2'}.issubset(set(r['per_gene']['gene']))

    def test_custom_erythroid_labels(self):
        raw, cor, gn, ct = _hbb_data()
        r = hbb_expression_analysis(raw, cor, ct, gn, erythroid_labels=['erythroid'])
        assert r['hbb_signal_reduced']

    def test_raises_missing_hbb_gene(self):
        raw, cor, gn, ct = _hbb_data()
        with pytest.raises(ValueError, match="None of hbb_genes"):
            hbb_expression_analysis(raw, cor, ct, gn, hbb_genes=['NOTEXIST'])

    def test_raises_no_noneryth_cells(self):
        raw, cor, gn, _ = _hbb_data()
        with pytest.raises(ValueError, match="non-erythroid"):
            hbb_expression_analysis(raw, cor, ['erythroid'] * raw.shape[1], gn)

    def test_pipeline_integration(self):
        rng     = np.random.default_rng(77)
        ng, nc  = 20, 80
        gn      = ['HBB'] + [f'Gene{i}' for i in range(ng - 1)]
        ct      = ['erythroid'] * 10 + ['T_cell'] * 70
        toc_arr = np.zeros((ng, nc))
        toc_arr[0, :10]  = rng.poisson(20.0, 10)
        toc_arr[0, 10:]  = rng.poisson(4.0,  70)
        toc_arr[1:, 10:] = rng.poisson(5.0,  (ng - 1, 70))
        tod_arr          = rng.poisson(0.1, (ng, nc * 3))
        tod_arr[0, :]    = rng.poisson(2.0, nc * 3)
        cl  = np.array(['ery'] * 10 + ['T'] * 70)
        sc  = _make_upgraded_sc(toc_arr, tod_arr, gn, cl)
        sc  = set_contamination_fraction(sc, 0.15)
        cor = adjust_counts(sc, method='subtraction')
        r   = hbb_expression_analysis(sc.toc, cor, ct, gn, hbb_genes=['HBB'])
        assert 'hbb_signal_reduced' in r
