from pathlib import Path

import pandas as pd

from benchmarks.rep1_zenodo_utils import load_rep1_zenodo_ground_truth


def test_load_rep1_zenodo_ground_truth_prefers_exported_csv(tmp_path):
    (tmp_path / "raw_feature_bc_matrix.h5").touch()
    (tmp_path / "filtered_feature_bc_matrix.h5").touch()
    (tmp_path / "rep1_cast_gt.csv").write_text(
        "barcode,sample,rho_gt\n"
        "AAAC-1,rep1,0.1\n"
        "AAAG-1,rep1,0.2\n"
    )

    gt = load_rep1_zenodo_ground_truth(str(tmp_path))

    assert list(gt.index) == ["AAAC", "AAAG"]
    assert gt["rho_gt"].tolist() == [0.1, 0.2]
