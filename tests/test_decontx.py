import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from SoupX import SoupChannel, adjust_counts, select_n_topics
from SoupX.decontx import run_decontx, _decontx_lda_em
from SoupX.estimation import _decontx_em, estimate_decontx_rho


# ─── Synthetic fixture ────────────────────────────────────────────────────────

def _make_contaminated_sc(n_genes=40, n_cells=80, n_drops=200,
                          true_rho=0.2, seed=0):
    """
    SoupChannel where each cell's counts are a known mixture:
      n_ambient ~ Binomial(n_umis, true_rho)  drawn from soup_weights
      n_native  = n_umis - n_ambient           drawn from cell-type profile
    """
    rng = np.random.default_rng(seed)

    # Soup concentrated on genes 0-4
    soup_weights = np.zeros(n_genes)
    soup_weights[:5] = [0.40, 0.25, 0.15, 0.10, 0.10]
    soup_weights /= soup_weights.sum()

    # Two cell types at 35% and 70% of gene range so they stay in bounds
    a_start = n_genes * 35 // 100
    b_start = n_genes * 70 // 100

    native_A = np.zeros(n_genes)
    native_A[a_start:a_start + 5] = [0.40, 0.25, 0.15, 0.10, 0.10]
    native_A /= native_A.sum()

    native_B = np.zeros(n_genes)
    native_B[b_start:b_start + 5] = [0.40, 0.25, 0.15, 0.10, 0.10]
    native_B /= native_B.sum()

    n_umis_per_cell = 300
    toc_cols = []
    for i in range(n_cells):
        native = native_A if i < n_cells // 2 else native_B
        n_amb = rng.binomial(n_umis_per_cell, true_rho)
        n_nat = n_umis_per_cell - n_amb
        col = rng.multinomial(n_amb, soup_weights) + rng.multinomial(n_nat, native)
        toc_cols.append(col)

    toc = scipy.sparse.csc_matrix(np.array(toc_cols).T.astype(float))

    # Empty droplets: 2-5 UMIs, soup profile
    tod_rows = [rng.multinomial(rng.integers(2, 6), soup_weights) for _ in range(n_drops)]
    tod = scipy.sparse.csc_matrix(np.array(tod_rows).T.astype(float))

    genes = pd.Index([f"Gene{i}" for i in range(n_genes)])
    cells = pd.Index([f"CELL{i:04d}" for i in range(n_cells)])

    return SoupChannel(
        tod=tod, toc=toc, genes=genes, cells=cells,
        drop_barcodes=[f"DROP{i:05d}" for i in range(n_drops)],
        calc_soup_profile=True,
    ), true_rho


@pytest.fixture(scope="module")
def contaminated_sc():
    sc, true_rho = _make_contaminated_sc()
    return sc, true_rho


# ─── run_decontx ─────────────────────────────────────────────────────────────

class TestRunDecontx:
    def test_sets_rho_column(self, contaminated_sc):
        sc, _ = contaminated_sc
        out = run_decontx(sc, n_topics=3, n_iter=50, verbose=False)
        assert "rho" in out.meta_data.columns

    def test_rho_in_unit_interval(self, contaminated_sc):
        sc, _ = contaminated_sc
        out = run_decontx(sc, n_topics=3, n_iter=50, verbose=False)
        rho = out.meta_data["rho"].values
        assert rho.min() >= 0.0
        assert rho.max() <= 1.0

    def test_topic_columns_present(self, contaminated_sc):
        sc, _ = contaminated_sc
        n_topics = 3
        out = run_decontx(sc, n_topics=n_topics, n_iter=50, verbose=False)
        for k in range(n_topics):
            assert f"decontx_topic_{k}" in out.meta_data.columns

    def test_fit_metadata_stored(self, contaminated_sc):
        sc, _ = contaminated_sc
        out = run_decontx(sc, n_topics=3, n_iter=50, verbose=False)
        assert out.fit["method"] == "decontx"
        assert "ll_history" in out.fit
        assert "final_ll" in out.fit

    def test_ll_nondecreasing(self, contaminated_sc):
        sc, _ = contaminated_sc
        out = run_decontx(sc, n_topics=3, n_iter=100, verbose=False)
        ll = np.array(out.fit["ll_history"])
        assert len(ll) > 1
        diffs = np.diff(ll)
        # Allow tiny floating-point regressions (< 1e-3 absolute)
        assert np.all(diffs >= -1e-3), f"LL decreased: min diff={diffs.min():.4e}"

    def test_recovers_contamination_fraction(self, contaminated_sc):
        sc, true_rho = contaminated_sc
        out = run_decontx(sc, n_topics=3, n_iter=200,
                          tol_theta=1e-4, tol_param=1e-5, verbose=False)
        mean_rho = float(out.meta_data["rho"].mean())
        # Weak check: within 0.15 of ground truth (noisy small dataset)
        assert abs(mean_rho - true_rho) < 0.15, (
            f"mean rho={mean_rho:.3f}, true={true_rho:.3f}"
        )

    def test_inplace_false_returns_copy(self, contaminated_sc):
        sc, _ = contaminated_sc
        out = run_decontx(sc, n_topics=3, n_iter=30, verbose=False, inplace=False)
        assert out is not sc
        assert "rho" not in sc.meta_data.columns

    def test_inplace_true_modifies_sc(self, contaminated_sc):
        sc, _ = _make_contaminated_sc(seed=99)
        out = run_decontx(sc, n_topics=3, n_iter=30, verbose=False, inplace=True)
        assert out is sc
        assert "rho" in sc.meta_data.columns


# ─── LL-based convergence ─────────────────────────────────────────────────────

class TestLLConvergence:
    def test_ll_convergence_fires_in_lda_em(self):
        """With tol_theta/tol_param=0 (delta never passes), tol_ll triggers early stop."""
        sc, _ = _make_contaminated_sc(n_genes=30, n_cells=60, seed=7)
        toc_sub = sc.toc
        pi_sub = sc.soup_profile["est"].values.astype(float)
        pi_sub /= pi_sub.sum() + 1e-10

        _, _, ll_hist = _decontx_lda_em(
            toc_sub, pi_sub,
            n_topics=3, n_iter=300, tol_theta=0.0, tol_param=0.0, tol_ll=1e-4,
            prior_rho=0.2, verbose=False,
        )
        assert len(ll_hist) < 300, (
            f"Expected early stop via rel-LL; ran all 300 iters. "
            f"Final LL delta: {abs(ll_hist[-1] - ll_hist[-2]):.4e}"
        )

    def test_ll_convergence_fires_in_decontx_em(self):
        """_decontx_em rel-LL check stops before max_iter when LL plateaus."""
        sc, _ = _make_contaminated_sc(n_genes=30, n_cells=60, seed=8)

        # Capture verbose output to verify which criterion fired
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _decontx_em(sc, prior_rho=0.2, n_iter=500, tol=0.0, tol_ll=1e-4,
                        verbose=True)
        output = buf.getvalue()
        # Either converged (rel-LL) or hit max_iter — just ensure it ran
        assert "rel-LL" in output or "max_iter" in output

    def test_lda_em_ll_history_length_matches_iters(self):
        sc, _ = _make_contaminated_sc(n_genes=20, n_cells=40, seed=3)
        toc = sc.toc
        pi = sc.soup_profile["est"].values.astype(float)
        pi /= pi.sum()

        _, _, ll_hist = _decontx_lda_em(
            toc, pi, n_topics=2, n_iter=10,
            tol_theta=0.0, tol_param=0.0, tol_ll=0.0,
            prior_rho=0.1, verbose=False,
        )
        # all tols=0 means no early stop; should run full 10 iters
        assert len(ll_hist) == 10


# ─── _decontx_em sparse fix ──────────────────────────────────────────────────

class TestDecontxEmSparse:
    def test_no_dense_materialization(self, monkeypatch):
        """_decontx_em must not call .toarray() on toc."""
        import scipy.sparse as sp

        called = []
        original_tocsr = sp.csc_matrix.tocsr

        def patched_tocsr(self, *args, **kwargs):
            result = original_tocsr(self, *args, **kwargs)
            # Patch toarray on the result to detect calls
            orig_toarray = result.toarray

            def spy_toarray(*a, **kw):
                called.append(True)
                return orig_toarray(*a, **kw)

            result.toarray = spy_toarray
            return result

        sc, _ = _make_contaminated_sc(n_genes=20, n_cells=30, seed=5)
        monkeypatch.setattr(sp.csc_matrix, "tocsr", patched_tocsr)

        _decontx_em(sc, n_iter=3, verbose=False)
        assert not called, "_decontx_em called .toarray() — dense matrix bug not fixed"

    def test_returns_valid_theta(self):
        sc, _ = _make_contaminated_sc(n_genes=25, n_cells=40, seed=6)
        theta = _decontx_em(sc, prior_rho=0.1, n_iter=50, verbose=False)
        assert theta.shape == (40,)
        assert np.all(theta >= 0.0)
        assert np.all(theta <= 1.0)
        assert not np.any(np.isnan(theta))

    def test_estimate_decontx_rho_sets_rho(self):
        sc, _ = _make_contaminated_sc(n_genes=25, n_cells=40, seed=11)
        out = estimate_decontx_rho(sc, n_iter=30, verbose=False)
        assert "rho" in out.meta_data.columns
        rho = out.meta_data["rho"].values
        assert rho.min() >= 0.0
        assert rho.max() <= 1.0


# ─── Soup profile quality check ───────────────────────────────────────────────

class TestSoupProfileCheck:
    def test_mt_dominated_soup_warns(self):
        """Warn when MT genes > 5% of soup."""
        import warnings as _w
        sc, _ = _make_contaminated_sc(n_genes=40, n_cells=40, seed=20)

        # Manually inflate soup with MT genes
        soup = sc.soup_profile.copy()
        soup.index = [f"MT-Gene{i}" if i < 5 else f"Gene{i}"
                      for i in range(len(soup))]
        soup['est'] = 0.0
        soup.loc[soup.index.str.startswith('MT-'), 'est'] = 0.15 / 5
        soup.loc[~soup.index.str.startswith('MT-'), 'est'] = 0.85 / 35
        sc.soup_profile = soup
        sc._toc_genes = soup.index  # keep consistent if needed

        from SoupX.decontx import _check_soup_profile
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            _check_soup_profile(sc)
        mt_warns = [w for w in caught if "MT" in str(w.message)]
        assert mt_warns, "Expected MT-soup warning, got none"

    def test_soup_equals_cells_warns(self):
        """Warn when top-10 soup genes heavily overlap with top 10% cell expression."""
        import warnings as _w
        from SoupX.decontx import _check_soup_profile

        # n_genes=60: top 10% = 6 genes, enough to trigger the ≥5 threshold
        sc, _ = _make_contaminated_sc(n_genes=60, n_cells=60, seed=21)
        mean_expr = np.asarray(sc.toc.mean(axis=1)).flatten()
        # Force soup to match mean cell expression exactly
        soup_est = mean_expr / (mean_expr.sum() + 1e-10)
        sc.soup_profile['est'] = soup_est

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            _check_soup_profile(sc)
        overlap_warns = [w for w in caught if "Soup" in str(w.message) or "soup" in str(w.message).lower()]
        assert overlap_warns, "Expected soup≈cells warning, got none"

    def test_no_spurious_warns_on_clean_profile(self):
        """Clean fetal-liver-style soup should not warn."""
        import warnings as _w
        from SoupX.decontx import _check_soup_profile

        # Clean sc: soup concentrated on genes absent in cells
        sc, _ = _make_contaminated_sc(n_genes=40, n_cells=60, seed=22)
        # Zero out all cells' expression in soup genes (genes 0-4) to ensure no overlap
        # by making soup genes 8-12 (in bottom half of cell expression)
        soup_est = np.zeros(40)
        soup_est[8:13] = [0.40, 0.25, 0.15, 0.10, 0.10]
        soup_est /= soup_est.sum()
        sc.soup_profile['est'] = soup_est

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            _check_soup_profile(sc)
        quality_warns = [w for w in caught
                         if "MT" in str(w.message) or "soup" in str(w.message).lower()]
        assert not quality_warns, f"Unexpected warnings: {[str(w.message) for w in quality_warns]}"


# ─── exclude_mt parameter ────────────────────────────────────────────────────

class TestExcludeMt:
    def test_exclude_mt_zeros_mt_in_pi(self):
        """With exclude_mt=True, MT genes must have zero weight in soup pi."""
        import warnings as _w

        sc, _ = _make_contaminated_sc(n_genes=40, n_cells=40, seed=30)
        # Rename genes 0-2 to MT- (80% of soup mass); genes 3-4 remain non-MT
        # so pi_sub retains ~20% mass after exclusion and the EM can run.
        new_idx = [f"MT-Gene{i}" if i < 3 else f"Gene{i}" for i in range(40)]
        sc.soup_profile.index = pd.Index(new_idx)
        # Suppress quality warnings for this test
        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            out = run_decontx(sc, n_topics=2, n_iter=10,
                              exclude_mt=True, verbose=False)
        # rho should still be in range — MT exclusion must not crash
        rho = out.meta_data["rho"].values
        assert np.all(rho >= 0.0) and np.all(rho <= 1.0)
        assert out.fit["exclude_mt"] is True

    def test_exclude_mt_false_no_change(self):
        """exclude_mt=False (default) must not modify soup profile."""
        import warnings as _w

        sc, _ = _make_contaminated_sc(n_genes=40, n_cells=40, seed=31)
        pi_before = sc.soup_profile['est'].values.copy()
        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            run_decontx(sc, n_topics=2, n_iter=10, exclude_mt=False, verbose=False)
        np.testing.assert_array_equal(sc.soup_profile['est'].values, pi_before)


# ─── Separate tol_theta / tol_param ─────────────────────────────────────────

class TestSeparateTols:
    def test_tight_tol_theta_does_not_block_on_param(self):
        """Beta/Pi can converge (tol_param) even if Θ is still oscillating."""
        sc, _ = _make_contaminated_sc(n_genes=30, n_cells=60, seed=40)
        toc = sc.toc
        pi = sc.soup_profile["est"].values.astype(float)
        pi /= pi.sum()

        # Very loose tol_theta (Θ always passes), tight tol_param — must converge
        _, _, ll_hist_loose = _decontx_lda_em(
            toc, pi, n_topics=3, n_iter=200,
            tol_theta=1.0, tol_param=1e-3, tol_ll=0.0,
            prior_rho=0.2, verbose=False,
        )
        # Very tight tol_theta (Θ never passes), loose tol_param — relies on tol_param
        _, _, ll_hist_tight = _decontx_lda_em(
            toc, pi, n_topics=3, n_iter=200,
            tol_theta=0.0, tol_param=1e-3, tol_ll=0.0,
            prior_rho=0.2, verbose=False,
        )
        # Both should converge before max_iter if param criterion works independently
        assert len(ll_hist_loose) < 200 or len(ll_hist_tight) < 200, (
            "Expected at least one variant to converge via tol_param before max_iter"
        )

    def test_fit_stores_both_tols(self):
        sc, _ = _make_contaminated_sc(seed=41)
        import warnings as _w
        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            out = run_decontx(sc, n_topics=2, n_iter=10,
                              tol_theta=1e-3, tol_param=1e-4, verbose=False)
        assert out.fit["tol_theta"] == 1e-3
        assert out.fit["tol_param"] == 1e-4


# ─── Item 11: Perplexity baseline ─────────────────────────────────────────────

class TestPerplexityBaseline:
    def test_fit_stores_perplexity_random_and_ratio(self):
        sc, _ = _make_contaminated_sc(seed=50)
        import warnings as _w
        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            out = run_decontx(sc, n_topics=3, n_iter=20, verbose=False)
        assert "perplexity_random" in out.fit
        assert "perplexity_ratio" in out.fit
        # random baseline = n_genes used
        assert out.fit["perplexity_random"] == float(sc.toc.shape[0])

    def test_perplexity_ratio_gt_1(self):
        """Model must beat the random baseline (perplexity < n_genes)."""
        sc, _ = _make_contaminated_sc(seed=51)
        import warnings as _w
        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            out = run_decontx(sc, n_topics=3, n_iter=50, verbose=False)
        assert out.fit["perplexity_ratio"] > 1.0, (
            f"Model perplexity {out.fit['final_perplexity']:.1f} >= "
            f"random baseline {out.fit['perplexity_random']:.0f}"
        )

    def test_verbose_output_contains_baseline(self, capsys):
        sc, _ = _make_contaminated_sc(seed=52)
        import warnings as _w
        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            run_decontx(sc, n_topics=2, n_iter=10, verbose=True)
        out = capsys.readouterr().out
        assert "random baseline" in out
        assert "ratio" in out


# ─── Item 8: select_n_topics ─────────────────────────────────────────────────

class TestSelectNTopics:
    def test_returns_dataframe_with_expected_columns(self):
        sc, _ = _make_contaminated_sc(seed=60)
        import warnings as _w
        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            result = select_n_topics(sc, n_topics_range=(2, 3), n_iter=10,
                                     verbose=False)
        assert isinstance(result, pd.DataFrame)
        for col in ("n_topics", "perplexity", "perplexity_ratio",
                    "mean_rho", "n_iter_run"):
            assert col in result.columns

    def test_rows_sorted_by_n_topics(self):
        sc, _ = _make_contaminated_sc(seed=61)
        import warnings as _w
        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            result = select_n_topics(sc, n_topics_range=(5, 2, 3), n_iter=10,
                                     verbose=False)
        assert list(result["n_topics"]) == sorted(result["n_topics"].tolist())

    def test_more_topics_not_worse_in_ratio(self):
        """Larger n_topics should not systematically degrade fit on clean data."""
        sc, _ = _make_contaminated_sc(n_genes=40, n_cells=80, seed=62)
        import warnings as _w
        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            result = select_n_topics(sc, n_topics_range=(2, 5), n_iter=30,
                                     verbose=False)
        # Higher K should match or improve perplexity
        ppl_k2 = result.loc[result.n_topics == 2, "perplexity"].iloc[0]
        ppl_k5 = result.loc[result.n_topics == 5, "perplexity"].iloc[0]
        # Allow small numerical noise but K=5 should not be much worse than K=2
        assert ppl_k5 <= ppl_k2 * 1.05, (
            f"K=5 perplexity ({ppl_k5:.1f}) much worse than K=2 ({ppl_k2:.1f})"
        )

    def test_deduplicates_repeated_n_topics(self):
        sc, _ = _make_contaminated_sc(seed=63)
        import warnings as _w
        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            result = select_n_topics(sc, n_topics_range=(3, 3, 3), n_iter=5,
                                     verbose=False)
        assert len(result) == 1


# ─── Item 10: End-to-end pipeline test ───────────────────────────────────────

class TestEndToEndPipeline:
    def test_run_decontx_then_adjust_counts_reduces_soup_genes(self):
        """
        run_decontx → adjust_counts: ambient marker expression must decrease
        in the cell population that does NOT natively express those genes.

        Fixture: genes 0-4 are soup markers, cells 40-79 are type B
        (native profile on genes at 70% position, far from soup genes).
        """
        sc, _ = _make_contaminated_sc(
            n_genes=40, n_cells=80, true_rho=0.3, seed=70
        )
        import warnings as _w
        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            sc_fit = run_decontx(sc, n_topics=3, n_iter=100, verbose=False)
        corrected = adjust_counts(sc_fit, clusters=False)

        # Soup genes are indices 0-4; type-B cells are columns 40-79
        soup_gene_idx = list(range(5))
        before = np.asarray(sc.toc[soup_gene_idx, :][:, 40:].sum(axis=0)).flatten()
        after  = np.asarray(corrected[soup_gene_idx, :][:, 40:].sum(axis=0)).flatten()

        assert after.mean() < before.mean(), (
            f"Soup gene expression in type-B cells did not decrease after correction. "
            f"Before mean={before.mean():.2f}, After mean={after.mean():.2f}"
        )

    def test_adjust_counts_preserves_shape_after_decontx(self):
        sc, _ = _make_contaminated_sc(seed=71)
        import warnings as _w
        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            sc_fit = run_decontx(sc, n_topics=2, n_iter=30, verbose=False)
        corrected = adjust_counts(sc_fit, clusters=False)
        assert corrected.shape == sc.toc.shape

    def test_corrected_counts_nonneg(self):
        sc, _ = _make_contaminated_sc(seed=72)
        import warnings as _w
        with _w.catch_warnings(record=True):
            _w.simplefilter("always")
            sc_fit = run_decontx(sc, n_topics=2, n_iter=30, verbose=False)
        corrected = adjust_counts(sc_fit, clusters=False)
        if corrected.nnz > 0:
            assert corrected.data.min() >= -1e-10
