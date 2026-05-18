#!/usr/bin/env python3
"""
benchmark.py — SoupX unified benchmark runner.

Evaluates all available datasets and reports metrics in benchmark-report style.
Missing datasets are skipped gracefully (download scripts noted per dataset).

Usage
-----
    python benchmark.py                         # all available datasets
    python benchmark.py --quick                 # Toy PBMC only (fast, always present)
    python benchmark.py --datasets hgmm toy_pbmc  # specific subset
    python benchmark.py --list                  # availability check, no runs

Datasets
--------
  hgmm          10X hgmm_1k barnyard — exact per-cell ground truth
  fetal_liver   E-MTAB-7407 fetal liver — cell-type contamination profile
  toy_pbmc      PBMC toy data (in-repo) — pinned golden regression
"""

import argparse
import gzip
import io
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse

# ── Repository root & import path ────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from SoupX import SoupChannel, set_clusters                    # noqa: E402 — path insert needed first
from SoupX import __version__ as _SOUPX_VERSION               # noqa: E402


# ── Dataset path detection ────────────────────────────────────────────────────

def _find_dir(*candidates: str) -> Optional[str]:
    """Return first existing directory from candidates, else None."""
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None


DATASETS = os.path.join(REPO_ROOT, 'datasets')

DATA: Dict[str, Optional[str]] = {
    'hgmm':        _find_dir(os.path.join(DATASETS, 'hgmm_1k')),
    'fetal_liver': _find_dir(os.path.join(DATASETS, 'E-MTAB-7407_fetal_liver', 'FCAImmP7352195')),
    'nsclc':       _find_dir(os.path.join(DATASETS, 'nsclc_tumor')),
    'toy_pbmc':    os.path.join(DATASETS, 'toyData'),
}

# Download instructions shown when a dataset is absent
_DOWNLOAD_HINTS: Dict[str, str] = {
    'hgmm':
        'bash datasets/download_hgmm.sh',
    'fetal_liver':
        'bash datasets/download_datasets.sh',
    'nsclc':
        'bash datasets/download_nsclc.sh',
    'toy_pbmc':
        '(always present — committed to repo)',
}


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    name:         str
    dataset_key:  str
    status:       str          # 'pass' | 'partial' | 'fail' | 'skip' | 'error'
    n_cells:      int   = 0
    n_genes:      int   = 0
    mean_rho_pct: float = float('nan')
    std_rho_pct:  float = float('nan')
    key_label:    str   = ''
    key_value:    str   = ''
    verdict_note: str   = ''
    elapsed_s:    float = 0.0
    skip_reason:  str   = ''
    details:      Dict[str, Any] = field(default_factory=dict)


# ── Display helpers ───────────────────────────────────────────────────────────

_W = 76   # total line width

def _rule(char: str = '=') -> None:
    print(char * _W)

def _blank() -> None:
    print()

def _banner() -> None:
    import datetime
    _blank()
    _rule()
    title = f'SoupX Python  |  Benchmark Report  |  v{_SOUPX_VERSION}'
    print(f'  {title}')
    print(f'  Date: {datetime.date.today().isoformat()}')
    _rule()

def _sec_header(idx: int, total: int, title: str) -> None:
    label = f'[{idx}/{total}]  {title}'
    _blank()
    _rule('-')
    print(f'  {label}')
    _rule('-')

def _row(label: str, value: str, indent: int = 2, lw: int = 28) -> None:
    print(f'{" " * indent}{label:<{lw}} {value}')

def _sub_header(title: str) -> None:
    print(f'\n  {"─" * 4}  {title}')

def _verdict_str(status: str) -> str:
    icons = {'pass': '✓ PASS', 'partial': '~ PARTIAL',
             'fail': '✗ FAIL',  'skip': '— SKIP', 'error': '✗ ERROR'}
    return icons.get(status, status.upper())

def _pct(v: float) -> str:
    return f'{v:.2f}%' if np.isfinite(v) else 'n/a'


# ── MEX matrix loaders ────────────────────────────────────────────────────────

def _load_mex_v2(directory: str):
    """Load CellRanger v2 (uncompressed) MEX format."""
    mat = scipy.io.mmread(os.path.join(directory, 'matrix.mtx')).tocsc().astype(float)
    bc  = pd.read_csv(os.path.join(directory, 'barcodes.tsv'), header=None)[0].values
    gdf = pd.read_csv(os.path.join(directory, 'genes.tsv'),    header=None, sep='\t')
    return mat, bc, gdf[0].values, gdf[1].values   # mat, barcodes, gene_ids, gene_names


def _load_mex_v3(directory: str):
    """Load CellRanger v3 (gzipped) MEX format."""
    def _gz(name):
        return os.path.join(directory, name)
    with gzip.open(_gz('matrix.mtx.gz'), 'rb') as f:
        mat = scipy.io.mmread(io.BytesIO(f.read())).tocsc().astype(float)
    with gzip.open(_gz('barcodes.tsv.gz'), 'rt') as f:
        bc = [line.strip() for line in f]
    with gzip.open(_gz('features.tsv.gz'), 'rt') as f:
        rows = [line.strip().split('\t') for line in f]
    return mat, bc, [r[0] for r in rows], [r[1] for r in rows]


# ── Dataset runners ───────────────────────────────────────────────────────────

def run_hgmm(base: str, idx: int, total: int) -> BenchmarkResult:
    """
    HGMM barnyard (hgmm_1k).

    Ground truth: per-cell math — minority-species UMIs / total UMIs.
    Human cell GT = mm10_UMIs / total.  Mouse cell GT = hg19_UMIs / total.
    Pass criteria: Pearson r > 0.50 AND MAE < 5 pp.
    """
    t0 = time.time()
    name = 'HGMM Barnyard (hgmm_1k)'

    raw_hg19  = os.path.join(base, 'raw_gene_bc_matrices',      'hg19')
    raw_mm10  = os.path.join(base, 'raw_gene_bc_matrices',      'mm10')
    filt_hg19 = os.path.join(base, 'filtered_gene_bc_matrices', 'hg19')
    filt_mm10 = os.path.join(base, 'filtered_gene_bc_matrices', 'mm10')

    _sec_header(idx, total, name)
    _row('Ground truth', 'per-cell math — exact, no annotation needed')
    _row('Benchmark type', 'STRONGEST — minority species = guaranteed contamination')

    # Load
    _row('Loading', '4 MEX v2 directories ...')
    rh, bc_raw,  _, hg_names = _load_mex_v2(raw_hg19)
    rm, bc_raw2, _, mm_names = _load_mex_v2(raw_mm10)
    fh, bc_ch,   _, _        = _load_mex_v2(filt_hg19)
    fm, bc_cm,   _, _        = _load_mex_v2(filt_mm10)

    assert np.array_equal(bc_raw, bc_raw2), 'hg19/mm10 raw barcodes differ'

    n_hg       = rh.shape[0]
    all_genes  = np.concatenate([hg_names, mm_names])
    all_cells  = np.concatenate([bc_ch, bc_cm])
    n_human    = len(bc_ch)
    n_mouse    = len(bc_cm)
    n_empty    = len(bc_raw) - len(all_cells)

    tod = scipy.sparse.vstack([rh, rm], format='csc')
    bc_to_idx = {b: i for i, b in enumerate(bc_raw)}
    toc = tod[:, np.array([bc_to_idx[b] for b in all_cells])]

    human_gene_mask = np.zeros(len(all_genes), dtype=bool)
    human_gene_mask[:n_hg] = True
    human_cell_mask = np.zeros(len(all_cells), dtype=bool)
    human_cell_mask[:n_human] = True
    mouse_cell_mask = ~human_cell_mask

    hg_umi  = np.array(toc[human_gene_mask,  :].sum(axis=0)).flatten()
    mm_umi  = np.array(toc[~human_gene_mask, :].sum(axis=0)).flatten()
    tot_umi = hg_umi + mm_umi

    gt = np.zeros(len(all_cells))
    gt[human_cell_mask] = mm_umi[human_cell_mask] / np.maximum(tot_umi[human_cell_mask], 1)
    gt[mouse_cell_mask] = hg_umi[mouse_cell_mask] / np.maximum(tot_umi[mouse_cell_mask], 1)

    _row('Cells',          f'{len(all_cells):,}  ({n_human} human HEK293T, {n_mouse} mouse NIH3T3)')
    _row('Genes',          f'{len(all_genes):,}  (hg19 + mm10 combined barnyard)')
    _row('Empty droplets', f'{n_empty:,}')
    _row('GT human',       f'mean={gt[human_cell_mask].mean()*100:.3f}%  '
                           f'range=[{gt[human_cell_mask].min()*100:.3f}%, '
                           f'{gt[human_cell_mask].max()*100:.3f}%]')
    _row('GT mouse',       f'mean={gt[mouse_cell_mask].mean()*100:.3f}%  '
                           f'range=[{gt[mouse_cell_mask].min()*100:.3f}%, '
                           f'{gt[mouse_cell_mask].max()*100:.3f}%]')

    # Build SoupChannel + run DecontX
    _row('Running', 'DecontX  n_topics=10, n_iter=300, n_hvg=2000 ...')
    from SoupX import run_decontx
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sc = SoupChannel(tod=tod, toc=toc,
                         genes=pd.Index(all_genes), cells=pd.Index(all_cells),
                         drop_barcodes=list(bc_raw), calc_soup_profile=True)
        sc = set_clusters(sc, np.where(human_cell_mask, 'human', 'mouse'))

        soup_hg = float(sc.soup_profile.loc[human_gene_mask, 'est'].sum())
        _row('Soup mix', f'{soup_hg*100:.1f}% human  /  {(1-soup_hg)*100:.1f}% mouse')

        sc_out = run_decontx(sc, n_topics=10, n_iter=300, tol_theta=1e-4, tol_param=1e-5,
                             n_hvg=2000, soup_top_q=0.9, pca_init=True, verbose=False)

    rho = sc_out.meta_data['rho'].values

    mae_h = float(np.abs(rho[human_cell_mask] - gt[human_cell_mask]).mean())
    mae_m = float(np.abs(rho[mouse_cell_mask] - gt[mouse_cell_mask]).mean())
    mae   = float(np.abs(rho - gt).mean())

    r_mean = rho.mean(); g_mean = gt.mean()
    denom  = np.linalg.norm(rho - r_mean) * np.linalg.norm(gt - g_mean)
    pearson_r = float(((rho - r_mean) * (gt - g_mean)).sum() / (denom + 1e-12))

    _sub_header('Results')
    hdr = f'  {"Metric":<28}  {"Human":>10}  {"Mouse":>10}  {"Overall":>10}'
    sep = '  ' + '─' * (len(hdr) - 2)
    print(hdr); print(sep)
    _row('Mean rho',
         f'{rho[human_cell_mask].mean()*100:>9.3f}%'
         f'  {rho[mouse_cell_mask].mean()*100:>9.3f}%'
         f'  {rho.mean()*100:>9.3f}%')
    _row('Ground truth mean',
         f'{gt[human_cell_mask].mean()*100:>9.3f}%'
         f'  {gt[mouse_cell_mask].mean()*100:>9.3f}%'
         f'  {gt.mean()*100:>9.3f}%')
    _row('MAE',
         f'{mae_h*100:>8.2f} pp'
         f'  {mae_m*100:>8.2f} pp'
         f'  {mae*100:>8.2f} pp')
    print(sep)
    _row('Pearson r(rho, GT)', f'{pearson_r:.4f}')
    _row('Std rho',            f'{rho.std()*100:.3f}%')

    # Pass criteria
    status = ('pass'    if pearson_r > 0.50 and mae < 0.05 else
              'partial' if pearson_r > 0.25                 else 'fail')
    elapsed = time.time() - t0
    _sub_header('Verdict')
    _row(_verdict_str(status),
         f'r={pearson_r:.3f} (need >0.50)   MAE={mae*100:.2f} pp (need <5 pp)')
    _row('Elapsed', f'{elapsed:.1f} s')

    return BenchmarkResult(
        name=name, dataset_key='hgmm', status=status,
        n_cells=len(all_cells), n_genes=len(all_genes),
        mean_rho_pct=float(rho.mean() * 100),
        std_rho_pct=float(rho.std() * 100),
        key_label='Pearson r', key_value=f'{pearson_r:.4f}',
        verdict_note=f'MAE={mae*100:.2f} pp',
        elapsed_s=elapsed,
        details={'pearson_r': pearson_r, 'mae': mae, 'mae_human': mae_h,
                 'mae_mouse': mae_m, 'gt_mean': float(gt.mean())},
    )


def run_fetal_liver(base: str, idx: int, total: int) -> BenchmarkResult:
    """
    Fetal Liver E-MTAB-7407.

    Ground truth: HBB/HBA2/HBA1 are known contaminants for non-erythroid cells.
    Expected: non-erythroid rho > erythroid rho (genuine HBB expression).
    Pass criteria: non_ery_rho > ery_rho AND HBB in top-10 soup genes.
    """
    t0 = time.time()
    name = 'Fetal Liver (E-MTAB-7407)'

    matrix_dir = os.path.join(base, 'GRCh38')
    meta_csv   = os.path.join(base, 'FCAImmP7352195.csv')

    _sec_header(idx, total, name)
    _row('Ground truth', 'HBB/HBA2/HBA1 high in non-erythroid = contamination signal')
    _row('Benchmark type', 'CELL-TYPE PROFILE — erythroid vs non-erythroid rho contrast')

    mat      = scipy.io.mmread(os.path.join(matrix_dir, 'matrix.mtx')).tocsc().astype(float)
    barcodes = pd.read_csv(os.path.join(matrix_dir, 'barcodes.tsv'),
                           header=None)[0].str.replace('-1', '').values
    gdf      = pd.read_csv(os.path.join(matrix_dir, 'genes.tsv'), header=None, sep='\t')
    genes    = pd.Index(gdf[1].values)

    meta      = pd.read_csv(meta_csv)
    meta['Barcodes']    = meta['Barcodes'].str.strip('"')
    meta['Cell.Labels'] = meta['Cell.Labels'].str.strip('"').str.strip()
    label_map = meta.set_index('Barcodes').reindex(barcodes)['Cell.Labels'].fillna('Unknown').values

    n_types = pd.Series(label_map).nunique()
    _row('Cells',          f'{mat.shape[1]:,}')
    _row('Genes',          f'{mat.shape[0]:,}')
    _row('Cell types',     f'{n_types}')
    _row('Note',           'No raw empty-droplet matrix — using aggregate cell counts as soup proxy')

    from SoupX import set_soup_profile
    sc = SoupChannel(tod=mat, toc=mat, genes=genes, cells=pd.Index(barcodes),
                     drop_barcodes=list(barcodes), calc_soup_profile=False)
    agg = np.array(mat.sum(axis=1)).flatten().astype(float)
    tot = agg.sum()
    sc = set_soup_profile(sc, pd.DataFrame({'counts': agg, 'est': agg / (tot + 1e-10)}, index=genes))
    sc = set_clusters(sc, label_map)

    top_soup = sc.soup_profile.nlargest(10, 'est')
    hb_genes = ['HBB', 'HBA2', 'HBA1', 'HBD', 'HBG1', 'HBG2']
    hb_in_top10 = [g for g in hb_genes if g in top_soup.index]
    hb_ranks = {g: int((sc.soup_profile['est'] > sc.soup_profile.loc[g, 'est']).sum()) + 1
                for g in hb_genes if g in sc.soup_profile.index}

    _row('Top-5 soup genes', ', '.join(top_soup.index[:5].tolist()) + ', ...')
    _row('HB in top-10',     ', '.join(hb_in_top10) if hb_in_top10 else 'none')
    for g, rank in sorted(hb_ranks.items(), key=lambda x: x[1]):
        est = float(sc.soup_profile.loc[g, 'est']) if g in sc.soup_profile.index else float('nan')
        _row(f'  {g} rank / est', f'{rank}  /  {est:.4f}')

    _row('Running', 'DecontX  n_topics=20, n_iter=500, n_hvg=3000 ...')
    from SoupX import run_decontx
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sc_out = run_decontx(sc, n_topics=20, n_iter=500, tol_theta=1e-4, tol_param=1e-5,
                             n_hvg=3000, soup_top_q=0.9, pca_init=True, verbose=False)

    rho   = sc_out.meta_data['rho'].values
    ct_df = pd.DataFrame({'rho': rho, 'ct': label_map})
    ct_summary = ct_df.groupby('ct')['rho'].agg(['mean', 'median', 'count'])
    ct_summary = ct_summary.sort_values('mean', ascending=False)

    ery_mask = pd.Series(label_map).str.contains('Erythroid|erythroid', na=False).values
    non_ery_rho = rho[~ery_mask].mean() if (~ery_mask).sum() > 0 else float('nan')
    ery_rho     = rho[ery_mask].mean()  if ery_mask.sum()  > 0 else float('nan')

    _sub_header('Results')
    _row('Mean rho (overall)', _pct(rho.mean() * 100))
    _row('Non-erythroid rho',  _pct(non_ery_rho * 100) + '  ← should be HIGH (HBB = contaminant)')
    _row('Erythroid rho',      _pct(ery_rho * 100)     + '  ← should be LOW  (HBB = genuine)')
    _row('HBB in top-10 soup', 'YES' if 'HBB' in hb_in_top10 else 'NO')

    print(f'\n  {"Cell type":<32}  {"Mean rho":>9}  {"Median":>9}  {"N":>6}')
    print('  ' + '─' * 62)
    for ct, row_s in ct_summary.head(8).iterrows():
        tag = '  ← erythroid' if 'Erythroid' in str(ct) or 'erythroid' in str(ct) else ''
        print(f'  {str(ct)[:32]:<32}  {row_s["mean"]*100:>8.2f}%  '
              f'{row_s["median"]*100:>8.2f}%  {int(row_s["count"]):>6}{tag}')
    if len(ct_summary) > 8:
        print(f'  ... {len(ct_summary)-8} more types')

    thresholds = [0.01, 0.05, 0.10, 0.20]
    _sub_header('Contamination thresholds')
    for t in thresholds:
        n = int((rho > t).sum())
        print(f'  >{t*100:.0f}%:  {n:5d} cells  ({n/len(rho)*100:.1f}%)')

    passes = (
        not np.isnan(non_ery_rho) and not np.isnan(ery_rho) and non_ery_rho > ery_rho
        and 'HBB' in hb_in_top10
    )
    status = ('pass'    if passes                          else
              'partial' if 'HBB' in hb_in_top10            else 'fail')
    elapsed = time.time() - t0
    _sub_header('Verdict')
    _row(_verdict_str(status),
         f'non-ery={_pct(non_ery_rho*100)} > ery={_pct(ery_rho*100)}   '
         f'HBB top-10: {"YES" if "HBB" in hb_in_top10 else "NO"}')
    _row('Elapsed', f'{elapsed:.1f} s')

    return BenchmarkResult(
        name=name, dataset_key='fetal_liver', status=status,
        n_cells=mat.shape[1], n_genes=mat.shape[0],
        mean_rho_pct=float(rho.mean() * 100),
        std_rho_pct=float(rho.std() * 100),
        key_label='non-ery / ery rho',
        key_value=f'{_pct(non_ery_rho*100)} / {_pct(ery_rho*100)}',
        verdict_note=f'HBB top-10: {"YES" if "HBB" in hb_in_top10 else "NO"}',
        elapsed_s=elapsed,
        details={'non_ery_rho': float(non_ery_rho) if np.isfinite(non_ery_rho) else None,
                 'ery_rho': float(ery_rho) if np.isfinite(ery_rho) else None,
                 'hb_in_top10': hb_in_top10, 'hb_ranks': hb_ranks},
    )


def run_nsclc(base: str, idx: int, total: int) -> BenchmarkResult:
    """
    NSCLC Tumor (vdj_v1_hs_nsclc_5gex, Cell Ranger 2.2.0).

    No external ground truth.
    Smoke test: checks that tumor-tissue contamination (3–35%) exceeds
    blood-level contamination, and that lung/epithelial markers appear in soup.
    Pass criteria: 0.03 < mean_rho < 0.35  AND  >=1 epithelial soup marker found.
    """
    t0 = time.time()
    name = 'NSCLC Tumor (vdj_v1_hs_nsclc_5gex)'

    raw_dir  = os.path.join(base, 'raw_gene_bc_matrices',      'GRCh38')
    filt_dir = os.path.join(base, 'filtered_gene_bc_matrices', 'GRCh38')

    _sec_header(idx, total, name)
    _row('Ground truth',   'none  (smoke test — tumor microenvironment, no barnyard)')
    _row('Benchmark type', 'SMOKE TEST — rho plausibility + epithelial soup markers')
    _row('Why useful',     'Tumor lysis → EPCAM/KRT7/SFTPB in soup; immune cells carry highest rho')

    _row('Loading', 'CellRanger v2 uncompressed MEX ...')
    tod, drop_bc, _, gnames = _load_mex_v2(raw_dir)
    toc, cell_bc, _, _      = _load_mex_v2(filt_dir)
    genes = pd.Index(gnames)

    n_empty = tod.shape[1] - toc.shape[1]
    _row('Cells',          f'{toc.shape[1]:,}')
    _row('Genes',          f'{toc.shape[0]:,}')
    _row('Empty droplets', f'{n_empty:,}')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sc = SoupChannel(tod=tod, toc=toc, genes=genes, cells=pd.Index(cell_bc),
                         drop_barcodes=list(drop_bc), calc_soup_profile=True)

    # Proxy clusters: 5 nUMI-quantile buckets (no cluster file with this dataset)
    numi = sc.meta_data['nUMIs'].values
    q    = np.percentile(numi, [20, 40, 60, 80])
    sc   = set_clusters(sc, np.digitize(numi, q).astype(str))

    top_soup = sc.soup_profile.nlargest(10, 'est')
    _row('Top-5 soup genes', ', '.join(top_soup.index[:5].tolist()) + ', ...')

    # Check lung/epithelial markers in soup
    epi_markers  = ['EPCAM', 'KRT7', 'KRT19', 'KRT8', 'KRT18']
    lung_markers = ['SFTPB', 'SFTPC', 'SFTPD', 'SCGB1A1']
    found_epi  = [(g, int((sc.soup_profile['est'] > sc.soup_profile.loc[g, 'est']).sum()) + 1)
                  for g in epi_markers  if g in sc.soup_profile.index]
    found_lung = [(g, int((sc.soup_profile['est'] > sc.soup_profile.loc[g, 'est']).sum()) + 1)
                  for g in lung_markers if g in sc.soup_profile.index]
    if found_epi:
        _row('Epithelial soup',
             '  '.join(f'{g}(rank={r})' for g, r in sorted(found_epi,  key=lambda x: x[1])[:4]))
    if found_lung:
        _row('Lung soup',
             '  '.join(f'{g}(rank={r})' for g, r in sorted(found_lung, key=lambda x: x[1])[:4]))

    _row('Running', 'DecontX  n_topics=20, n_iter=500, n_hvg=3000, prior_rho=0.10 ...')
    from SoupX import run_decontx
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sc_out = run_decontx(sc, n_topics=20, n_iter=500, tol_theta=1e-4, tol_param=1e-5,
                             n_hvg=3000, prior_rho=0.10, soup_top_q=0.9,
                             pca_init=True, exclude_mt=True, verbose=False)

    rho   = sc_out.meta_data['rho'].values
    pctls = {p: float(np.percentile(rho, p)) for p in [10, 25, 50, 75, 90, 95, 99]}

    _sub_header('Results')
    _row('Mean rho',   f'{rho.mean()*100:.2f}%')
    _row('Std rho',    f'{rho.std()*100:.2f}%')
    _row('Median rho', f'{pctls[50]*100:.2f}%')
    _row('Percentiles',
         f'p10={pctls[10]*100:.2f}%  p25={pctls[25]*100:.2f}%  p50={pctls[50]*100:.2f}%  '
         f'p75={pctls[75]*100:.2f}%  p90={pctls[90]*100:.2f}%  p95={pctls[95]*100:.2f}%')

    _sub_header('Cells above contamination thresholds')
    for t in [0.01, 0.05, 0.10, 0.20, 0.30]:
        n = int((rho > t).sum())
        print(f'  >{t*100:4.0f}%:  {n:6,} cells  ({n/len(rho)*100:5.1f}%)')

    n_epi_found = len(found_epi) + len(found_lung)
    status = ('pass'    if 0.03 <= rho.mean() <= 0.35 and n_epi_found > 0 else
              'partial' if 0.03 <= rho.mean() <= 0.35                      else 'fail')
    elapsed = time.time() - t0
    _sub_header('Verdict')
    _row(_verdict_str(status),
         f'mean rho={rho.mean()*100:.2f}% (expected 3–35%)  '
         f'epi/lung soup markers found: {n_epi_found}')
    _row('Elapsed', f'{elapsed:.1f} s')

    return BenchmarkResult(
        name=name, dataset_key='nsclc', status=status,
        n_cells=toc.shape[1], n_genes=toc.shape[0],
        mean_rho_pct=float(rho.mean() * 100),
        std_rho_pct=float(rho.std() * 100),
        key_label='epi markers in soup', key_value=str(n_epi_found),
        verdict_note=f'expected 3–35% rho',
        elapsed_s=elapsed,
        details={'pctiles': pctls, 'found_epi': found_epi, 'found_lung': found_lung},
    )


def run_toy_pbmc(base: str, idx: int, total: int) -> BenchmarkResult:
    """
    Toy PBMC (in-repo, always present).

    Method: SoupX pipeline (auto_est_cont, not DecontX).
    Pass criteria: rho within 5% of pinned golden (0.068), top-1 soup gene = LTB.
    """
    t0 = time.time()
    name = 'Toy PBMC (in-repo)'

    # Pinned values from test_regression.py — update both places if algorithm changes
    GOLDEN = {
        'rho_global':      0.068,
        'soup_top1_gene':  'LTB',
        'soup_top5_genes': ['LTB', 'LDHB', 'IL32', 'CD3D', 'CD3E'],
        'n_cells': 62, 'n_genes': 226,
    }

    _sec_header(idx, total, name)
    _row('Ground truth', 'pinned golden values (Python self-consistency, not R validation)')
    _row('Benchmark type', 'REGRESSION — detects algorithm drift vs. previously accepted output')
    _row('Note', 'Validates SoupX pipeline (auto_est_cont); see hgmm for external ground truth')

    from SoupX.io import load_10x
    from SoupX import auto_est_cont

    _row('Loading', f'{base}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sc = load_10x(base, verbose=False)

    meta_path = os.path.join(base, 'metaData.tsv')
    meta = pd.read_csv(meta_path, sep='\t', index_col=0)
    clusters = meta['res.1'].astype(str).reindex(sc.cells).fillna('0')
    sc = set_clusters(sc, clusters)

    _row('Cells',    f'{len(sc.cells)}  (golden: {GOLDEN["n_cells"]})')
    _row('Genes',    f'{len(sc.genes)}  (golden: {GOLDEN["n_genes"]})')
    _row('Clusters', f'{sc.meta_data["clusters"].nunique()} unique')
    _row('Running',  'auto_est_cont (SoupX pipeline) ...')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sc_fit = auto_est_cont(sc, do_plot=False, verbose=False, force_accept=True)

    rho_val    = float(sc_fit.meta_data['rho'].iloc[0])
    top5       = sc_fit.soup_profile.nlargest(5, 'est')
    n_usable   = int(sc_fit.fit['dd']['useEst'].sum())
    rho_err    = abs(rho_val - GOLDEN['rho_global']) / GOLDEN['rho_global'] * 100
    top1_match = top5.index[0] == GOLDEN['soup_top1_gene']
    top5_match = top5.index.tolist() == GOLDEN['soup_top5_genes']

    _sub_header('Results vs. golden')
    print(f'  {"Metric":<28}  {"Got":>10}  {"Golden":>10}  {"Delta / Match"}')
    print('  ' + '─' * 70)
    _row('rho (global)',
         f'{rho_val:.4f}  '
         f'    {GOLDEN["rho_global"]:.4f}  '
         f'    {rho_val - GOLDEN["rho_global"]:+.4f}  ({rho_err:.1f}%)')
    _row('Top-1 soup gene',
         f'{top5.index[0]:<10}  '
         f'    {GOLDEN["soup_top1_gene"]:<10}  '
         f'    {"✓ match" if top1_match else "✗ mismatch"}')
    _row('Top-5 soup genes', ', '.join(top5.index.tolist()))
    _row('  vs. golden',     ', '.join(GOLDEN['soup_top5_genes']) +
         ('  ✓' if top5_match else '  ✗'))
    _row('n_usable_estimates', str(n_usable))
    _row('Top-5 est values',
         '  '.join(f'{v:.4f}' for v in top5['est'].values))

    status = ('pass'    if rho_err < 5.0  and top1_match else
              'partial' if rho_err < 20.0              else 'fail')
    elapsed = time.time() - t0
    _sub_header('Verdict')
    _row(_verdict_str(status),
         f'rho err={rho_err:.1f}% (need <5%)   top-1 gene: {"✓" if top1_match else "✗"}')
    _row('Elapsed', f'{elapsed:.1f} s')

    return BenchmarkResult(
        name=name, dataset_key='toy_pbmc', status=status,
        n_cells=len(sc.cells), n_genes=len(sc.genes),
        mean_rho_pct=rho_val * 100,
        std_rho_pct=0.0,
        key_label='rho err vs golden', key_value=f'{rho_err:.1f}%',
        verdict_note=f'top-1 gene: {top5.index[0]}',
        elapsed_s=elapsed,
        details={'rho': rho_val, 'rho_err_pct': rho_err, 'n_usable': n_usable,
                 'top5_genes': top5.index.tolist(), 'top5_est': top5['est'].tolist()},
    )


def _make_skip(name: str, key: str, reason: str) -> BenchmarkResult:
    """
    Print a SKIPPED block and return a skip BenchmarkResult.

    :param name: Human-readable dataset name.
    :type name: str
    :param key: Dataset key (e.g. 'hgmm').
    :type key: str
    :param reason: Reason for skipping (e.g. dataset not found).
    :type reason: str
    :return: BenchmarkResult with status='skip'.
    :rtype: BenchmarkResult
    """
    _blank()
    _rule('-')
    print(f'  SKIPPED  {name}')
    print(f'  Reason : {reason}')
    _rule('-')
    return BenchmarkResult(name=name, dataset_key=key, status='skip', skip_reason=reason)


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary(results: List[BenchmarkResult]) -> None:
    """
    Print a formatted summary table of all benchmark results.

    :param results: List of BenchmarkResult objects from each dataset run.
    :type results: list[BenchmarkResult]
    """
    _blank()
    _rule()
    print('  BENCHMARK SUMMARY')
    _rule()

    c_ds = 32; c_n = 9; c_rho = 10; c_km = 28; c_v = 12

    print(f'  {"Dataset":<{c_ds}}  {"N_cells":>{c_n}}  {"Mean_rho":>{c_rho}}'
          f'  {"Key metric":<{c_km}}  {"Verdict":<{c_v}}')
    print('  ' + '─' * (c_ds + c_n + c_rho + c_km + c_v + 8))

    counts: Dict[str, int] = {'pass': 0, 'partial': 0, 'fail': 0, 'skip': 0, 'error': 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        n_str   = f'{r.n_cells:,}' if r.n_cells > 0 else 'n/a'
        rho_str = _pct(r.mean_rho_pct) if np.isfinite(r.mean_rho_pct) else 'n/a'
        key_str = (f'{r.key_label}={r.key_value}' if r.key_label
                   else r.skip_reason[:c_km])[:c_km]
        print(f'  {r.name:<{c_ds}}'
              f'  {n_str:>{c_n}}'
              f'  {rho_str:>{c_rho}}'
              f'  {key_str:<{c_km}}'
              f'  {_verdict_str(r.status):<{c_v}}')

    print('  ' + '─' * (c_ds + c_n + c_rho + c_km + c_v + 8))

    n_run  = counts['pass'] + counts['partial'] + counts['fail'] + counts.get('error', 0)
    n_skip = counts['skip']
    total_elapsed = sum(r.elapsed_s for r in results)

    _blank()
    print(f'  Datasets   : {n_run} run  |  {counts["pass"]} PASS  |  '
          f'{counts["partial"]} PARTIAL  |  {counts["fail"]} FAIL  |  {n_skip} SKIP')
    print(f'  Total time : {total_elapsed:.1f} s  ({total_elapsed/60:.1f} min)')
    _blank()

    if counts['fail'] > 0 or counts.get('error', 0) > 0:
        overall = '✗  BENCHMARK FAILED'
    elif counts['partial'] > 0:
        overall = f'~  BENCHMARK PARTIAL  ({counts["partial"]} dataset(s) show partial agreement)'
    elif n_run == 0:
        overall = '—  No datasets available — download external data to run full benchmark'
    else:
        overall = f'✓  ALL {n_run} AVAILABLE DATASETS PASSED'

    print(f'  {overall}')
    _blank()
    _rule()
    _blank()


# ── CLI entry point ───────────────────────────────────────────────────────────

_RUNNERS = {
    'hgmm':        run_hgmm,
    'fetal_liver': run_fetal_liver,
    'nsclc':       run_nsclc,
    'toy_pbmc':    run_toy_pbmc,
}

_DEFAULT_ORDER = ['hgmm', 'fetal_liver', 'nsclc', 'toy_pbmc']


def main() -> None:
    parser = argparse.ArgumentParser(
        description='SoupX unified benchmark runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Toy PBMC only — fast, always available, no DecontX',
    )
    parser.add_argument(
        '--list', action='store_true',
        help='Show dataset availability and exit',
    )
    parser.add_argument(
        '--datasets', nargs='+',
        choices=_DEFAULT_ORDER,
        metavar='DATASET',
        help='Run specific datasets (hgmm fetal_liver nsclc toy_pbmc)',
    )
    args = parser.parse_args()

    _banner()

    # Availability report
    avail = {k: (v is not None and os.path.isdir(v)) for k, v in DATA.items()}
    print('  Dataset availability:')
    for k in _DEFAULT_ORDER:
        icon = '✓' if avail[k] else '✗'
        loc  = DATA[k] if DATA[k] else '(not found)'
        print(f'    {icon}  {k:<15}  {loc}')
    _blank()

    if args.list:
        print('  Download hints for missing datasets:')
        for k in _DEFAULT_ORDER:
            if not avail[k]:
                print(f'    {k:<15}  {_DOWNLOAD_HINTS[k]}')
        _blank()
        return

    # Decide which datasets to run
    if args.quick:
        to_run = ['toy_pbmc']
    elif args.datasets:
        to_run = args.datasets
    else:
        to_run = _DEFAULT_ORDER

    total = len(to_run)
    results: List[BenchmarkResult] = []

    for i, key in enumerate(to_run, 1):
        name = {
            'hgmm':        'HGMM Barnyard (hgmm_1k)',
            'fetal_liver': 'Fetal Liver (E-MTAB-7407)',
            'nsclc':       'NSCLC Tumor (vdj_v1_hs_nsclc_5gex)',
            'toy_pbmc':    'Toy PBMC (in-repo)',
        }[key]

        if not avail.get(key, False):
            hint = _DOWNLOAD_HINTS.get(key, '')
            results.append(_make_skip(name, key, hint or f'{key} data not found'))
            continue

        try:
            results.append(_RUNNERS[key](DATA[key], i, total))
        except Exception as exc:
            import traceback
            print(f'\n  ERROR running {name}:')
            traceback.print_exc()
            results.append(BenchmarkResult(
                name=name, dataset_key=key, status='error',
                skip_reason=str(exc),
            ))

    _print_summary(results)


if __name__ == '__main__':
    main()
