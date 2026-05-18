---
sidebar_position: 5
---

# Assessment Metrics

All metric functions accept `(toc_raw, toc_corrected, ...)` sparse matrices and return a `dict`, making results easy to tabulate.

## `cross_species_reduction`

Measure cross-species contamination fold reduction in a barnyard experiment.

**Parameters:** `toc_raw`, `toc_corrected`, `gene_names` (with species prefixes `hg19_*`, `mm10_*`), `cell_species`

**Returns:** `dict` with `human_before`, `human_after`, `mouse_before`, `mouse_after`, `fold_reduction`, `meets_2fold_threshold`

---

## `marker_fold_change`

Measure fold change in marker gene expression before and after correction.

**Parameters:** `toc_raw`, `toc_corrected`, `gene_names`, `cluster_labels`, `marker_genes` (dict mapping cluster names to gene lists)

**Returns:** `dict` with per-cluster fold changes and summary statistics

---

## `cluster_membership_delta`

Measure shift in cluster composition after correction.

**Parameters:** `toc_raw`, `toc_corrected`, `gene_names`

**Returns:** `dict` with cluster stability and membership change statistics

---

## `batch_entropy`

Local neighbourhood batch-mixing entropy. Higher = better mixing between batches.

**Parameters:** `toc_corrected`, `batch_labels`, `k` (number of nearest neighbours)

**Returns:** `dict` with mean batch entropy before and after correction

---

## `hbb_expression_analysis`

Measure HBB/HBA contamination removal in non-erythroid cells.

**Parameters:** `toc_raw`, `toc_corrected`, `gene_names`, `cell_types`

**Returns:** `dict` with HBB expression before/after in erythroid and non-erythroid cells

---

## `cluster_silhouette`

Silhouette score of clusters in the corrected count space.

**Parameters:** `toc_corrected`, `cluster_labels`

**Returns:** `dict` with silhouette score

---

## `spurious_de_reduction`

Measure reduction in spurious differentially expressed genes between clusters.

**Parameters:** `toc_raw`, `toc_corrected`, `gene_names`, `cluster_labels`

**Returns:** `dict` with number of spurious DE genes before and after correction

---

## `marker_enrichment_score`

Enrichment of known cell-type markers post-correction.

**Parameters:** `toc_corrected`, `gene_names`, `cluster_labels`, `marker_dict`

**Returns:** `dict` with marker enrichment scores per cell type

---

## Running all metrics at once

```python
from SoupX.metrics import (
    cross_species_reduction,
    marker_fold_change,
    cluster_membership_delta,
    hbb_expression_analysis,
    cluster_silhouette,
    spurious_de_reduction,
    marker_enrichment_score,
)

results = {
    'm2_marker_fc':     marker_fold_change(sc.toc, corrected, sc.genes, labels, markers),
    'm3_cluster_delta': cluster_membership_delta(sc.toc, corrected, sc.genes),
    'm6_silhouette':    cluster_silhouette(corrected, labels),
    'm7_spurious_de':   spurious_de_reduction(sc.toc, corrected, sc.genes, labels),
    'm8_enrichment':    marker_enrichment_score(corrected, sc.genes, labels, marker_dict),
}
```
