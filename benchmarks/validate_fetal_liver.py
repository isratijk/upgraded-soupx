"""
Fetal liver validation for DecontX.

Dataset: E-MTAB-7407 (FCAImmP7352195) — fetal liver scRNA-seq
Ground truth: HBB/HBA2/HBA1 are known ambient contaminants.
Expected: non-erythroid cells show high rho (~10-20%).
Erythroid cells genuinely express HBB, so DecontX should assign
their hemoglobin counts as native, not contamination.

No raw empty-droplet matrix available — use aggregate cell counts
as soup proxy (standard fallback).
"""

import sys
import os
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.io

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

DATA_DIR = os.path.join(REPO_ROOT, 'datasets', 'E-MTAB-7407_fetal_liver', 'FCAImmP7352195')
MATRIX_DIR = os.path.join(DATA_DIR, 'GRCh38')

print("=" * 60)
print("Fetal Liver DecontX Validation")
print("=" * 60)

# ── Load matrix ───────────────────────────────────────────────
print("\n[1/5] Loading count matrix...")
mat = scipy.io.mmread(os.path.join(MATRIX_DIR, 'matrix.mtx')).tocsc()
print(f"  Shape: {mat.shape[0]} genes × {mat.shape[1]} cells")

barcodes = pd.read_csv(os.path.join(MATRIX_DIR, 'barcodes.tsv'),
                       header=None)[0].str.replace('-1', '').values

genes_df = pd.read_csv(os.path.join(MATRIX_DIR, 'genes.tsv'),
                       header=None, sep='\t')
gene_ids   = genes_df[0].values
gene_names = genes_df[1].values
genes = pd.Index(gene_names)

# ── Load cell type labels ─────────────────────────────────────
print("[2/5] Loading cell type metadata...")
meta = pd.read_csv(os.path.join(DATA_DIR, 'FCAImmP7352195.csv'))
meta['Barcodes'] = meta['Barcodes'].str.strip('"')
meta['Cell.Labels'] = meta['Cell.Labels'].str.strip('"').str.strip()
meta = meta.set_index('Barcodes')

label_map = meta.reindex(barcodes)['Cell.Labels'].fillna('Unknown').values
print(f"  Cell types: {sorted(set(label_map))}")

# ── Build SoupChannel ─────────────────────────────────────────
print("[3/5] Building SoupChannel (cells as soup proxy)...")
from SoupX import SoupChannel, set_clusters, set_soup_profile

cells_idx = pd.Index(barcodes)

# Build SoupChannel without soup estimation (no raw empty-droplet matrix)
sc = SoupChannel(
    tod=mat,
    toc=mat,
    genes=genes,
    cells=cells_idx,
    drop_barcodes=list(barcodes),
    calc_soup_profile=False,
)

# Estimate soup from aggregate cell expression (standard fallback)
agg_counts = np.array(mat.sum(axis=1)).flatten().astype(float)
total = agg_counts.sum()
soup_est = agg_counts / (total + 1e-10)
soup_df = pd.DataFrame(
    {'counts': agg_counts, 'est': soup_est},
    index=genes
)
sc = set_soup_profile(sc, soup_df)

# Add cluster labels
sc = set_clusters(sc, label_map)
print(f"  nUMIs: mean={sc.meta_data['nUMIs'].mean():.0f}, "
      f"median={sc.meta_data['nUMIs'].median():.0f}")

# ── Check soup profile ────────────────────────────────────────
print("\n[4/5] Top 10 soup genes:")
top_soup = sc.soup_profile.nlargest(10, 'est')
for gene, row in top_soup.iterrows():
    print(f"  {gene:<15s}  {row['est']:.4f}")

hb_genes = ['HBB', 'HBA2', 'HBA1', 'HBD', 'HBG1', 'HBG2']
print("\n  Hemoglobin genes in soup profile:")
for g in hb_genes:
    if g in sc.soup_profile.index:
        pct = sc.soup_profile.loc[g, 'est']
        rank = (sc.soup_profile['est'] > pct).sum() + 1
        print(f"  {g:<8s}  est={pct:.4f}  rank={rank}")

# ── Run DecontX ───────────────────────────────────────────────
print("\n[5/5] Running DecontX (n_topics=20, n_iter=500)...")
from SoupX import run_decontx

sc_decontx = run_decontx(
    sc,
    n_topics=20,
    n_iter=500,
    tol_theta=1e-4,
    tol_param=1e-5,
    n_hvg=3000,
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
print(f"Overall: mean rho={rho.mean():.4f} ({rho.mean()*100:.2f}%),  "
      f"std={rho.std():.4f},  "
      f"range=[{rho.min():.4f}, {rho.max():.4f}]")

# Contamination by cell type
print("\nContamination by cell type:")
ct_rho = pd.DataFrame({'rho': rho, 'cell_type': label_map})
summary = ct_rho.groupby('cell_type')['rho'].agg(['mean', 'median', 'count'])
summary = summary.sort_values('mean', ascending=False)
print(f"  {'Cell Type':<30s}  {'Mean rho':>8s}  {'Median':>8s}  {'N':>5s}")
print("  " + "-" * 60)
for ct, row in summary.iterrows():
    marker = " ← erythroid" if 'Erythroid' in str(ct) or 'erythroid' in str(ct) else ""
    print(f"  {str(ct):<30s}  {row['mean']*100:>7.2f}%  "
          f"{row['median']*100:>7.2f}%  {int(row['count']):>5d}{marker}")

# Threshold analysis
thresholds = [0.01, 0.05, 0.10, 0.20]
print("\nCells above contamination thresholds:")
for t in thresholds:
    n = (rho > t).sum()
    print(f"  >{t*100:.0f}%:  {n} cells ({n/len(rho)*100:.1f}%)")

print("\nConclusion:")
non_ery = ct_rho[~ct_rho['cell_type'].str.contains('Erythroid|erythroid', na=False)]['rho']
ery = ct_rho[ct_rho['cell_type'].str.contains('Erythroid|erythroid', na=False)]['rho']
if len(non_ery) > 0:
    print(f"  Non-erythroid mean rho: {non_ery.mean()*100:.2f}%")
if len(ery) > 0:
    print(f"  Erythroid mean rho:     {ery.mean()*100:.2f}%  "
          f"(should be LOW — genuine HBB/HBA2 expression)")
print(f"  HBB in top-10 soup genes: "
      f"{'YES' if 'HBB' in top_soup.index else 'NO'}")
