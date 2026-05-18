"""
Upgraded-SoupX - ambient RNA contamination removal for droplet scRNA-seq.

Python port and extension of the R SoupX package (Young & Behjati, 2020).
Developed by Israt Jahan Khan (https://www.isratjahankhan.com).

Typical workflow
----------------
>>> from soupx import load_10x, set_clusters, auto_est_cont, adjust_counts
>>> sc = load_10x('path/to/cellranger/output')
>>> sc = set_clusters(sc, clusters)          # from Seurat/Scanpy clustering
>>> sc = auto_est_cont(sc)                   # fully automatic
>>> out = adjust_counts(sc)                  # returns corrected count matrix

Per-cell contamination refinement (optional, after auto_est_cont)
-----------------------------------------------------------------
>>> from soupx import estimate_cell_rho, estimate_decontx_rho
>>> sc = estimate_cell_rho(sc)              # empirical Bayes shrinkage
>>> # or
>>> sc = estimate_decontx_rho(sc)          # DecontX two-component EM per cell

Manual workflow (when you know which genes are non-expressed in some cells)
---------------------------------------------------------------------------
>>> from soupx import (load_10x, set_clusters,
...     estimate_non_expressing_cells, calculate_contamination_fraction,
...     adjust_counts)
>>> sc = load_10x('path/to/cellranger/output')
>>> sc = set_clusters(sc, clusters)
>>> gene_list = {'HB': ['HBB', 'HBA2']}
>>> use_to_est = estimate_non_expressing_cells(sc, gene_list)
>>> sc = calculate_contamination_fraction(sc, gene_list, use_to_est)
>>> out = adjust_counts(sc)
"""

from .soup_channel import SoupChannel
from .io import load_10x, read_10x, read_10x_h5, load_10x_h5
from .estimate_soup import estimate_soup
from .set_properties import (
    set_soup_profile,
    set_clusters,
    set_contamination_fraction,
    set_dr,
)
from .markers import quick_markers
from .estimation import (
    estimate_non_expressing_cells,
    calculate_contamination_fraction,
    auto_est_cont,
    estimate_cell_rho,
    estimate_decontx_rho,
)
from .correction import adjust_counts
from .decontx import run_decontx, select_n_topics
from .doublet import estimate_doublet_scores, auto_est_cont_doublet_aware
from .iterative import iterative_auto_est_cont
from .gene_het import compute_gene_enrichment, reweight_soup_profile, run_decontx_genehet
from .plot import (
    plot_soup_correlation,
    plot_marker_distribution,
    plot_marker_map,
    plot_change_map,
)

from .metrics import (
    cross_species_reduction,
    marker_fold_change,
    cluster_membership_delta,
    batch_entropy,
    hbb_expression_analysis,
    cluster_silhouette,
    spurious_de_reduction,
    marker_enrichment_score,
)

from .downstream import (
    normalize_log1p,
    run_pca,
    run_umap,
    run_tsne,
    cluster_leiden,
    cluster_kmeans,
    differential_expression,
    score_cell_types,
    plot_embedding,
    plot_top_de_genes,
    run_downstream,
)

__version__ = '1.7.0'
__author__ = 'Israt Jahan Khan'
__email__ = 'isratjahankhanijk@gmail.com'
__url__ = 'https://github.com/IsratIJK/Upgraded-soupX'
__all__ = [
    'cross_species_reduction',
    'marker_fold_change',
    'cluster_membership_delta',
    'batch_entropy',
    'hbb_expression_analysis',
    'cluster_silhouette',
    'spurious_de_reduction',
    'marker_enrichment_score',
    'SoupChannel',
    'load_10x',
    'read_10x',
    'read_10x_h5',
    'load_10x_h5',
    'estimate_soup',
    'set_soup_profile',
    'set_clusters',
    'set_contamination_fraction',
    'set_dr',
    'quick_markers',
    'estimate_non_expressing_cells',
    'calculate_contamination_fraction',
    'auto_est_cont',
    'estimate_cell_rho',
    'estimate_decontx_rho',
    'adjust_counts',
    'run_decontx',
    'select_n_topics',
    'estimate_doublet_scores',
    'auto_est_cont_doublet_aware',
    'iterative_auto_est_cont',
    'compute_gene_enrichment',
    'reweight_soup_profile',
    'run_decontx_genehet',
    'plot_soup_correlation',
    'plot_marker_distribution',
    'plot_marker_map',
    'plot_change_map',
    'normalize_log1p',
    'run_pca',
    'run_umap',
    'run_tsne',
    'cluster_leiden',
    'cluster_kmeans',
    'differential_expression',
    'score_cell_types',
    'plot_embedding',
    'plot_top_de_genes',
    'run_downstream',
]
