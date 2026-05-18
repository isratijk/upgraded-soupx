"""
Human-Mouse Species Mixing (hgmm_1k) — DecontX validation.

Dataset: 10x Genomics "1k 1:1 Mixture of Human HEK293T and Mouse NIH3T3 Cells" (v2.1.0)
         CellRanger outputs SEPARATE hg19 (human) and mm10 (mouse) matrices.
         This script merges them into a combined barnyard matrix for DecontX.

Ground truth (per cell, computed directly from counts — no annotation needed):
  Human cell: ground_truth_rho = mm10_UMIs / (hg19_UMIs + mm10_UMIs)
  Mouse cell: ground_truth_rho = hg19_UMIs / (hg19_UMIs + mm10_UMIs)

Why this is the strongest possible DecontX benchmark:
  1. Identifiability PERFECT: human cells express ZERO mouse genes natively.
     Any mouse transcript in a human cell is 100% ambient contamination.
  2. Ground truth is exact per-cell math, not a biological assumption.
  3. Real empty-droplet soup from raw_gene_bc_matrices (737k droplets).
  4. Soup ≠ cells guaranteed: soup is ~50/50 human+mouse RNA.
  5. This is the dataset the original DecontX paper (Yang et al. 2020) used.

Matrix structure after merging (barnyard):
  tod: (hg19_genes + mm10_genes) × 737,280 all-droplets  [~60k × 737k sparse]
  toc: (hg19_genes + mm10_genes) × 1,020 cells           [~60k × 1020 sparse]
  Human cells (504): hg19 rows HIGH, mm10 rows LOW (contamination)
  Mouse cells (516): mm10 rows HIGH, hg19 rows LOW (contamination)
"""

import sys
import os
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.io

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

BASE = os.path.join(REPO_ROOT, 'datasets', 'hgmm_1k')
RAW_HG19  = os.path.join(BASE, 'raw_gene_bc_matrices',      'hg19')
RAW_MM10  = os.path.join(BASE, 'raw_gene_bc_matrices',      'mm10')
FILT_HG19 = os.path.join(BASE, 'filtered_gene_bc_matrices', 'hg19')
FILT_MM10 = os.path.join(BASE, 'filtered_gene_bc_matrices', 'mm10')

SPECIES_THRESHOLD = 0.90


def load_v2_matrix(directory):
    """Load CellRanger v2 (non-gzipped) sparse matrix."""
    mat = scipy.io.mmread(os.path.join(directory, 'matrix.mtx')).tocsc()
    barcodes  = pd.read_csv(os.path.join(directory, 'barcodes.tsv'),
                            header=None)[0].values
    genes_df  = pd.read_csv(os.path.join(directory, 'genes.tsv'),
                            header=None, sep='\t')
    return mat, barcodes, genes_df[0].values, genes_df[1].values


print("=" * 60)
print("hgmm_1k — DecontX Species-Mixing Validation")
print("=" * 60)

for d in [RAW_HG19, RAW_MM10, FILT_HG19, FILT_MM10]:
    if not os.path.isdir(d):
        print(f"\n  ERROR: directory not found: {d}")
        print("  Run: bash datasets/download_hgmm.sh")
        sys.exit(1)

# ── Load all four matrices ────────────────────────────────────
print("\n[1/7] Loading hg19 and mm10 matrices...")
raw_hg19,  raw_bc_hg19,  _, hg19_names = load_v2_matrix(RAW_HG19)
raw_mm10,  raw_bc_mm10,  _, mm10_names = load_v2_matrix(RAW_MM10)
filt_hg19, cell_bc_hg19, _, _          = load_v2_matrix(FILT_HG19)
filt_mm10, cell_bc_mm10, _, _          = load_v2_matrix(FILT_MM10)

assert np.array_equal(raw_bc_hg19, raw_bc_mm10), \
    "Raw barcodes differ between hg19 and mm10 — unexpected CellRanger output"

n_hg19_genes = raw_hg19.shape[0]
n_mm10_genes = raw_mm10.shape[0]
n_raw_bc     = len(raw_bc_hg19)
n_human      = len(cell_bc_hg19)
n_mouse      = len(cell_bc_mm10)
print(f"  hg19: {n_hg19_genes:,} genes  |  raw {n_raw_bc:,} barcodes  |  filtered {n_human} cells")
print(f"  mm10: {n_mm10_genes:,} genes  |  raw {n_raw_bc:,} barcodes  |  filtered {n_mouse} cells")

# ── Merge into combined barnyard matrices ─────────────────────
print("\n[2/7] Merging into combined barnyard matrix...")
all_gene_names = np.concatenate([hg19_names, mm10_names])
all_cell_bc    = np.concatenate([cell_bc_hg19, cell_bc_mm10])  # 504 + 516 = 1,020

# Both raw matrices share the same 737,280 barcodes in the same order.
# Stack genes vertically → combined barnyard raw matrix: 60,736 × 737,280.
tod_combined = scipy.sparse.vstack([raw_hg19, raw_mm10], format='csc')
print(f"  Combined tod: {tod_combined.shape[0]:,} genes × {tod_combined.shape[1]:,} droplets  "
      f"(nnz={tod_combined.nnz:,})")

# Extract cell columns from combined raw matrix to get toc.
# This correctly captures cross-species counts for each cell.
raw_bc_to_idx = {bc: i for i, bc in enumerate(raw_bc_hg19)}
cell_col_idx  = np.array([raw_bc_to_idx[bc] for bc in all_cell_bc])
toc_combined  = tod_combined[:, cell_col_idx]
print(f"  Combined toc: {toc_combined.shape[0]:,} genes × {toc_combined.shape[1]:,} cells")

genes_idx = pd.Index(all_gene_names)

# ── Species masks on genes and cells ─────────────────────────
human_gene_mask = np.zeros(len(all_gene_names), dtype=bool)
human_gene_mask[:n_hg19_genes] = True
mouse_gene_mask = ~human_gene_mask

# Per-cell: UMI totals by species
hg19_umi = np.array(toc_combined[human_gene_mask, :].sum(axis=0)).flatten()
mm10_umi = np.array(toc_combined[mouse_gene_mask, :].sum(axis=0)).flatten()
total_umi = hg19_umi + mm10_umi

human_frac = hg19_umi / np.maximum(total_umi, 1)
mouse_frac  = mm10_umi / np.maximum(total_umi, 1)

# Cell species labels from index: first n_human are human, rest are mouse
human_cell_mask = np.zeros(len(all_cell_bc), dtype=bool)
human_cell_mask[:n_human] = True
mouse_cell_mask = ~human_cell_mask

# ── Ground truth rho ─────────────────────────────────────────
print("\n[3/7] Computing per-cell ground truth contamination...")
ground_truth = np.zeros(len(all_cell_bc))
ground_truth[human_cell_mask] = (
    mm10_umi[human_cell_mask] / np.maximum(total_umi[human_cell_mask], 1))
ground_truth[mouse_cell_mask] = (
    hg19_umi[mouse_cell_mask] / np.maximum(total_umi[mouse_cell_mask], 1))

print(f"  Human cells (n={n_human}):  "
      f"mean gt rho = {ground_truth[human_cell_mask].mean()*100:.3f}%  "
      f"range [{ground_truth[human_cell_mask].min()*100:.3f}%, "
      f"{ground_truth[human_cell_mask].max()*100:.3f}%]")
print(f"  Mouse cells (n={n_mouse}):  "
      f"mean gt rho = {ground_truth[mouse_cell_mask].mean()*100:.3f}%  "
      f"range [{ground_truth[mouse_cell_mask].min()*100:.3f}%, "
      f"{ground_truth[mouse_cell_mask].max()*100:.3f}%]")

# ── Build SoupChannel ─────────────────────────────────────────
print("\n[4/7] Building SoupChannel...")
from SoupX import SoupChannel, set_clusters

sc = SoupChannel(
    tod=tod_combined,
    toc=toc_combined,
    genes=genes_idx,
    cells=pd.Index(all_cell_bc),
    drop_barcodes=list(raw_bc_hg19),
    calc_soup_profile=True,
)
n_empty = n_raw_bc - len(all_cell_bc)
print(f"  Soup estimated from {n_empty:,} empty droplets")
print(f"  Cell nUMIs: mean={sc.meta_data['nUMIs'].mean():.0f}, "
      f"median={sc.meta_data['nUMIs'].median():.0f}")

soup_hg19_frac = sc.soup_profile.loc[human_gene_mask, 'est'].sum()
soup_mm10_frac = sc.soup_profile.loc[mouse_gene_mask, 'est'].sum()
print(f"  Soup species: {soup_hg19_frac*100:.1f}% human / {soup_mm10_frac*100:.1f}% mouse")

# Cluster labels = species (seeded for LDA topic alignment)
species_labels = np.where(human_cell_mask, 'human', 'mouse')
sc = set_clusters(sc, species_labels)

# ── Top soup genes ────────────────────────────────────────────
print("\n[5/7] Top 10 soup genes:")
top_soup = sc.soup_profile.nlargest(10, 'est')
for gene, row in top_soup.iterrows():
    species = 'human' if gene.startswith('hg19') else 'mouse'
    print(f"  {gene:<30s}  {row['est']:.4f}  [{species}]")

# ── Run DecontX ───────────────────────────────────────────────
print("\n[6/7] Running DecontX (n_topics=10, n_iter=300, n_hvg=2000)...")
from SoupX import run_decontx

sc_decontx = run_decontx(
    sc,
    n_topics=10,
    n_iter=300,
    tol_theta=1e-4,
    tol_param=1e-5,
    n_hvg=2000,
    soup_top_q=0.9,
    pca_init=True,
    verbose=True,
    inplace=False,
)

# ── Results ───────────────────────────────────────────────────
rho = sc_decontx.meta_data['rho'].values

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Overall: mean rho={rho.mean()*100:.2f}%  "
      f"std={rho.std()*100:.2f}%  "
      f"range=[{rho.min()*100:.2f}%, {rho.max()*100:.2f}%]")

print("\nBy species:")
for label, mask in [('Human (HEK293T)', human_cell_mask),
                    ('Mouse (NIH3T3)',  mouse_cell_mask)]:
    r = rho[mask]
    g = ground_truth[mask]
    err = np.abs(r - g)
    print(f"  {label:<16s}  n={mask.sum():4d}  "
          f"mean_rho={r.mean()*100:5.3f}%  "
          f"ground_truth={g.mean()*100:5.3f}%  "
          f"MAE={err.mean()*100:4.3f}%  "
          f"max_err={err.max()*100:4.2f}%")

# Pearson r between rho and ground truth
from numpy.linalg import norm
r_mean = rho.mean(); g_mean = ground_truth.mean()
pearson_r = (
    ((rho - r_mean) * (ground_truth - g_mean)).sum()
    / (norm(rho - r_mean) * norm(ground_truth - g_mean) + 1e-12)
)
mae = np.abs(rho - ground_truth).mean()

print(f"\nPearson r(rho, ground_truth): {pearson_r:.4f}")
print(f"Mean absolute error:          {mae*100:.3f} pp")

print("\n" + "=" * 60)
print("VALIDATION VERDICT")
print("=" * 60)
if pearson_r > 0.5 and mae < 0.05:
    print(f"  PASS  — rho tracks ground-truth contamination "
          f"(r={pearson_r:.3f}, MAE={mae*100:.2f} pp)")
elif pearson_r > 0.25:
    print(f"  PARTIAL — weak correlation (r={pearson_r:.3f}), "
          f"MAE={mae*100:.2f} pp")
else:
    print(f"  FAIL  — rho does not track ground truth "
          f"(r={pearson_r:.3f}, MAE={mae*100:.2f} pp)")

print("\n[7/7] Three-dataset summary:")
print("  PBMC 10k v3     (negative control):  ~0.29% mean rho  [very low, as expected]")
print(f"  hgmm_1k        (EXACT ground truth): {rho.mean()*100:.2f}% mean rho  "
      f"[Pearson r={pearson_r:.3f} vs per-cell math ground truth]")
