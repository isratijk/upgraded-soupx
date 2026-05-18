# Assessment Metrics

All metric functions accept `(toc_raw, toc_corrected, ...)` sparse matrices and return a dict, making results easy to tabulate.

## `cross_species_reduction`

Measure cross-species contamination fold reduction in a barnyard experiment.

:param toc_raw: Raw count matrix (genes × cells).
:type toc_raw: sparse matrix

:param toc_corrected: Corrected count matrix (genes × cells).
:type toc_corrected: sparse matrix

:param gene_names: Gene names with species prefixes (``hg19_*``, ``mm10_*``, etc.).
:type gene_names: array-like

:param cell_species: Species label for each cell (``'human'`` or ``'mouse'``).
:type cell_species: array-like

:return: Dict with keys human_before, human_after, mouse_before, mouse_after, fold_reduction, meets_2fold_threshold.
:rtype: dict

---

## `marker_fold_change`

Measure fold change in marker gene expression before and after correction.

:param toc_raw: Raw count matrix.
:type toc_raw: sparse matrix

:param toc_corrected: Corrected count matrix.
:type toc_corrected: sparse matrix

:param gene_names: Gene names.
:type gene_names: array-like

:param cluster_labels: Cluster assignment for each cell.
:type cluster_labels: array-like

:param marker_genes: Dict mapping cluster names to known marker gene lists.
:type marker_genes: dict

:return: Dict with per-cluster fold changes and summary statistics.
:rtype: dict

---

## `cluster_membership_delta`

Measure shift in cluster composition after correction.

:param toc_raw: Raw count matrix.
:type toc_raw: sparse matrix

:param toc_corrected: Corrected count matrix.
:type toc_corrected: sparse matrix

:param gene_names: Gene names.
:type gene_names: array-like

:return: Dict with cluster stability and membership change statistics.
:rtype: dict

---

## `batch_entropy`

Local neighbourhood batch-mixing entropy. Higher = better mixing between batches.

:param toc_corrected: Corrected count matrix.
:type toc_corrected: sparse matrix

:param batch_labels: Batch assignment for each cell.
:type batch_labels: array-like

:param k: Number of nearest neighbours for entropy computation.
:type k: int

:return: Dict with mean batch entropy before and after correction.
:rtype: dict

---

## `hbb_expression_analysis`

Measure HBB/HBA contamination removal in non-erythroid cells.

:param toc_raw: Raw count matrix.
:type toc_raw: sparse matrix

:param toc_corrected: Corrected count matrix.
:type toc_corrected: sparse matrix

:param gene_names: Gene names.
:type gene_names: array-like

:param cell_types: Cell type labels.
:type cell_types: array-like

:return: Dict with HBB expression before/after in erythroid and non-erythroid cells.
:rtype: dict

---

## `cluster_silhouette`

Silhouette score of clusters in the corrected count space.

:param toc_corrected: Corrected count matrix.
:type toc_corrected: sparse matrix

:param cluster_labels: Cluster assignment for each cell.
:type cluster_labels: array-like

:return: Dict with silhouette score.
:rtype: dict

---

## `spurious_de_reduction`

Measure reduction in spurious differentially expressed genes between clusters.

:param toc_raw: Raw count matrix.
:type toc_raw: sparse matrix

:param toc_corrected: Corrected count matrix.
:type toc_corrected: sparse matrix

:param gene_names: Gene names.
:type gene_names: array-like

:param cluster_labels: Cluster assignment for each cell.
:type cluster_labels: array-like

:return: Dict with number of spurious DE genes before and after correction.
:rtype: dict

---

## `marker_enrichment_score`

Enrichment of known cell-type markers post-correction.

:param toc_corrected: Corrected count matrix.
:type toc_corrected: sparse matrix

:param gene_names: Gene names.
:type gene_names: array-like

:param cluster_labels: Cluster assignment for each cell.
:type cluster_labels: array-like

:param marker_dict: Dict mapping cell-type names to marker gene lists.
:type marker_dict: dict

:return: Dict with marker enrichment scores per cell type.
:rtype: dict
