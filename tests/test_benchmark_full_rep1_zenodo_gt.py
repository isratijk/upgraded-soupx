from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse

import benchmarks.benchmark_full as bf
import benchmarks.rep1_zenodo_utils as rzu


def _fake_sc():
    tod = scipy.sparse.csc_matrix(
        [
            [10, 9, 1, 0, 0],
            [8, 7, 0, 1, 0],
            [0, 1, 9, 8, 7],
            [0, 0, 7, 9, 8],
        ],
        dtype=float,
    )
    toc = tod[:, :4]
    return SimpleNamespace(
        tod=tod,
        toc=toc,
        genes=pd.Index(["G1", "G2", "G3", "G4"]),
        cells=pd.Index(["c1-1", "c2-1", "c3-1", "c4-1"]),
        n_drop_umis=pd.Series([10, 10, 10, 10, 1], index=pd.Index(["c1-1", "c2-1", "c3-1", "c4-1", "d1"])),
    )


def test_benchmark_rep1_zenodo_gt_runs_with_stubbed_inputs(monkeypatch):
    monkeypatch.setattr(rzu, "load_rep1_zenodo_sample", lambda base_dir, verbose=False: _fake_sc())
    monkeypatch.setattr(
        rzu,
        "load_rep1_zenodo_gt_aligned",
        lambda base_dir, cells, gt_path=None: pd.DataFrame(
            {"rho_gt": [0.10, 0.20], "sample": ["rep1", "rep1"]},
            index=["c1", "c3"],
        ),
    )
    monkeypatch.setattr(bf, "_make_clusters_pca_kmeans", lambda *args, **kwargs: np.array(["0", "0", "1", "1"]))

    def _ok_pipe(tod, toc, gene_names, cell_names, clusters_series, **kwargs):
        rho = np.array([0.11, 0.15, 0.19, 0.18], dtype=float)
        return gene_names, toc, toc.copy(), rho

    def _ok_pipe_cell_filtered(tod, toc, gene_names, bc_raw, cell_names, clusters_series, **kwargs):
        rho = np.array([0.11, 0.15, 0.19, 0.18], dtype=float)
        return gene_names, toc, toc.copy(), rho

    monkeypatch.setattr(bf, "_pipe_baseline_from_mat", _ok_pipe)
    monkeypatch.setattr(bf, "_pipe_upg_auto_from_mat", _ok_pipe)
    monkeypatch.setattr(bf, "_pipe_upg_doublet", _ok_pipe_cell_filtered)
    monkeypatch.setattr(bf, "_pipe_upg_iterative", _ok_pipe_cell_filtered)
    monkeypatch.setattr(bf, "_pipe_upg_decontx", _ok_pipe_cell_filtered)
    monkeypatch.setattr(bf, "_pipe_upg_genehet", _ok_pipe_cell_filtered)
    monkeypatch.setattr(bf, "_run_m3", lambda *args, **kwargs: {"ari": 1.0, "n_clusters_lost": 0})
    monkeypatch.setattr(bf, "_run_m6", lambda *args, **kwargs: {"sil_delta": 0.0, "improved": True})
    monkeypatch.setattr(bf, "_run_m7", lambda *args, **kwargs: {"n_spurious": 0, "pct_spurious": 0.0})

    entries = bf.benchmark_rep1_zenodo_gt(skip_decontx=False)

    assert [entry.pipeline for entry in entries] == [
        "baseline",
        "upg-auto",
        "upg-doublet",
        "upg-iterative",
        "upg-decontx",
        "upg-genehet",
    ]
    assert all(entry.dataset == "rep1_zenodo_gt" for entry in entries)
    assert all(entry.gt_mae is not None for entry in entries)
