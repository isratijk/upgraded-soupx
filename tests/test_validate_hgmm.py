"""
HGMM species-mixing benchmark — automated pytest wrapper for validate_hgmm.py.

Dataset: 10x Genomics "1k 1:1 Mixture of Human HEK293T and Mouse NIH3T3 Cells".
Ground truth: per-cell contamination = minority-species UMIs / total UMIs.
This is the strongest possible DecontX benchmark (Yang et al. 2020).

Tests are automatically skipped when the dataset is absent.  To download:
    bash datasets/download_hgmm.sh

PASS criteria (from validate_hgmm.py):
    Pearson r(rho, ground_truth) > 0.50
    Mean absolute error          < 5 percentage points
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.io
import pytest


# ── Dataset location ─────────────────────────────────────────────────────────

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
BASE = os.path.join(_REPO_ROOT, 'datasets', 'hgmm_1k')
RAW_HG19  = os.path.join(BASE, 'raw_gene_bc_matrices',      'hg19')
RAW_MM10  = os.path.join(BASE, 'raw_gene_bc_matrices',      'mm10')
FILT_HG19 = os.path.join(BASE, 'filtered_gene_bc_matrices', 'hg19')
FILT_MM10 = os.path.join(BASE, 'filtered_gene_bc_matrices', 'mm10')

_DATA_PRESENT = all(os.path.isdir(d) for d in [RAW_HG19, RAW_MM10, FILT_HG19, FILT_MM10])

skip_if_no_data = pytest.mark.skipif(
    not _DATA_PRESENT,
    reason=(
        "hgmm_1k dataset not found. "
        "Run: bash datasets/download_hgmm.sh"
    ),
)

SPECIES_THRESHOLD = 0.90


# ── Dataset loading (module-scoped so the 737k-droplet matrix loads once) ────

def _load_v2_matrix(directory):
    mat = scipy.io.mmread(os.path.join(directory, 'matrix.mtx')).tocsc()
    barcodes = pd.read_csv(
        os.path.join(directory, 'barcodes.tsv'), header=None
    )[0].values
    genes_df = pd.read_csv(
        os.path.join(directory, 'genes.tsv'), header=None, sep='\t'
    )
    return mat, barcodes, genes_df[0].values, genes_df[1].values


@pytest.fixture(scope="module")
def hgmm_result():
    """
    Build the barnyard SoupChannel, run DecontX, and return
    (rho, ground_truth, human_cell_mask, mouse_cell_mask).
    Runs once per test session.
    """
    if not _DATA_PRESENT:
        pytest.skip("hgmm_1k dataset not present")

    raw_hg19,  raw_bc_hg19,  _, hg19_names = _load_v2_matrix(RAW_HG19)
    raw_mm10,  raw_bc_mm10,  _, mm10_names = _load_v2_matrix(RAW_MM10)
    filt_hg19, cell_bc_hg19, _, _          = _load_v2_matrix(FILT_HG19)
    filt_mm10, cell_bc_mm10, _, _          = _load_v2_matrix(FILT_MM10)

    assert np.array_equal(raw_bc_hg19, raw_bc_mm10), \
        "Raw barcodes differ between hg19 and mm10"

    n_hg19_genes = raw_hg19.shape[0]
    all_gene_names = np.concatenate([hg19_names, mm10_names])
    all_cell_bc    = np.concatenate([cell_bc_hg19, cell_bc_mm10])
    n_human = len(cell_bc_hg19)

    tod_combined = scipy.sparse.vstack([raw_hg19, raw_mm10], format='csc')

    raw_bc_to_idx = {bc: i for i, bc in enumerate(raw_bc_hg19)}
    cell_col_idx  = np.array([raw_bc_to_idx[bc] for bc in all_cell_bc])
    toc_combined  = tod_combined[:, cell_col_idx]

    human_gene_mask = np.zeros(len(all_gene_names), dtype=bool)
    human_gene_mask[:n_hg19_genes] = True
    mouse_gene_mask = ~human_gene_mask

    hg19_umi = np.array(toc_combined[human_gene_mask, :].sum(axis=0)).flatten()
    mm10_umi = np.array(toc_combined[mouse_gene_mask, :].sum(axis=0)).flatten()
    total_umi = hg19_umi + mm10_umi

    human_cell_mask = np.zeros(len(all_cell_bc), dtype=bool)
    human_cell_mask[:n_human] = True
    mouse_cell_mask = ~human_cell_mask

    ground_truth = np.zeros(len(all_cell_bc))
    ground_truth[human_cell_mask] = (
        mm10_umi[human_cell_mask] / np.maximum(total_umi[human_cell_mask], 1))
    ground_truth[mouse_cell_mask] = (
        hg19_umi[mouse_cell_mask] / np.maximum(total_umi[mouse_cell_mask], 1))

    sys.path.insert(0, _REPO_ROOT)
    from SoupX import SoupChannel, set_clusters, run_decontx

    sc = SoupChannel(
        tod=tod_combined,
        toc=toc_combined,
        genes=pd.Index(all_gene_names),
        cells=pd.Index(all_cell_bc),
        drop_barcodes=list(raw_bc_hg19),
        calc_soup_profile=True,
    )
    sc = set_clusters(sc, np.where(human_cell_mask, 'human', 'mouse'))

    sc_decontx = run_decontx(
        sc,
        n_topics=10,
        n_iter=300,
        tol_theta=1e-4,
        tol_param=1e-5,
        n_hvg=2000,
        soup_top_q=0.9,
        pca_init=True,
        verbose=False,
        inplace=False,
    )

    rho = sc_decontx.meta_data['rho'].values
    return rho, ground_truth, human_cell_mask, mouse_cell_mask


# ── Tests ─────────────────────────────────────────────────────────────────────

@skip_if_no_data
class TestHGMMValidation:

    def test_pearson_r_overall(self, hgmm_result):
        rho, ground_truth, _, _ = hgmm_result
        r_mean = rho.mean(); g_mean = ground_truth.mean()
        denom = (np.linalg.norm(rho - r_mean) * np.linalg.norm(ground_truth - g_mean))
        pearson_r = float(((rho - r_mean) * (ground_truth - g_mean)).sum() / (denom + 1e-12))
        assert pearson_r > 0.50, (
            f"Pearson r={pearson_r:.4f} < 0.50 — DecontX rho does not track "
            "ground-truth contamination on hgmm_1k."
        )

    def test_mae_overall(self, hgmm_result):
        rho, ground_truth, _, _ = hgmm_result
        mae = float(np.abs(rho - ground_truth).mean())
        assert mae < 0.05, (
            f"MAE={mae*100:.2f} pp > 5 pp — DecontX rho is too far from "
            "ground-truth contamination on hgmm_1k."
        )

    def test_human_cell_rho_plausible(self, hgmm_result):
        rho, ground_truth, human_mask, _ = hgmm_result
        mean_rho = rho[human_mask].mean()
        mean_gt  = ground_truth[human_mask].mean()
        assert mean_rho < 0.20, (
            f"Human cell mean rho={mean_rho*100:.2f}% seems too high (gt≈{mean_gt*100:.2f}%)"
        )

    def test_mouse_cell_rho_plausible(self, hgmm_result):
        rho, ground_truth, _, mouse_mask = hgmm_result
        mean_rho = rho[mouse_mask].mean()
        mean_gt  = ground_truth[mouse_mask].mean()
        assert mean_rho < 0.20, (
            f"Mouse cell mean rho={mean_rho*100:.2f}% seems too high (gt≈{mean_gt*100:.2f}%)"
        )

    def test_rho_range_valid(self, hgmm_result):
        rho, _, _, _ = hgmm_result
        assert rho.min() >= 0.0, "rho contains negative values"
        assert rho.max() <= 1.0, "rho > 1.0"

    def test_soup_profile_species_balance(self, hgmm_result):
        """Soup should be roughly 50/50 human/mouse for this barnyard dataset."""
        # We can't access sc here, but we can check the rho distribution is bimodal:
        # human cells and mouse cells should have similar mean rho.
        rho, _, human_mask, mouse_mask = hgmm_result
        human_mean = rho[human_mask].mean()
        mouse_mean  = rho[mouse_mask].mean()
        ratio = max(human_mean, mouse_mean) / (min(human_mean, mouse_mean) + 1e-6)
        assert ratio < 5.0, (
            f"Human/mouse mean rho ratio={ratio:.2f} — "
            f"human={human_mean*100:.2f}%, mouse={mouse_mean*100:.2f}%. "
            "Expected similar contamination levels for a 1:1 mix."
        )
