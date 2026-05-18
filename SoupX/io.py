import os
import gzip
import io
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse

from .soup_channel import SoupChannel


def read_10x_h5(h5_path, genome=None):
    """
    Read a 10X HDF5 (.h5) file produced by cellranger.

    5-10x faster than MEX format for large datasets.
    Supports both v2 (genome-named groups) and v3 ('matrix' group) layouts.

    Parameters
    ----------
    h5_path : str
        Path to .h5 file (e.g. raw_feature_bc_matrix.h5).
    genome : str, optional
        Genome group name for v2 multi-genome files. If None uses the first group.

    Returns
    -------
    mat : scipy.sparse.csc_matrix
        (n_genes x n_barcodes) count matrix.
    gene_names : pd.Index
    barcodes : pd.Index
    feature_types : list of str
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py required to read .h5 files. "
            "Install it into this interpreter with: python3 -m pip install --user h5py"
        )

    def _decode(arr):
        return [x.decode('utf-8') if isinstance(x, bytes) else x for x in arr]

    with h5py.File(h5_path, 'r') as f:
        if 'matrix' in f:
            # cellranger v3+ layout
            grp = f['matrix']
            data = grp['data'][:]
            indices = grp['indices'][:]
            indptr = grp['indptr'][:]
            shape = tuple(grp['shape'][:])           # [n_features, n_barcodes]
            barcodes = pd.Index(_decode(grp['barcodes'][:]))
            feat = grp['features']
            gene_names = pd.Index(_decode(feat['name'][:]))
            if 'feature_type' in feat:
                feature_types = _decode(feat['feature_type'][:])
            else:
                feature_types = ['Gene Expression'] * shape[0]
        else:
            # cellranger v2 layout: /{genome}/data, indices, indptr, barcodes, gene_names
            if genome is None:
                genome = list(f.keys())[0]
            grp = f[genome]
            data = grp['data'][:]
            indices = grp['indices'][:]
            indptr = grp['indptr'][:]
            shape = tuple(grp['shape'][:])
            barcodes = pd.Index(_decode(grp['barcodes'][:]))
            gene_names = pd.Index(_decode(grp['gene_names'][:]))
            feature_types = ['Gene Expression'] * shape[0]

    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape).astype(float)
    return mat, gene_names, barcodes, feature_types


def load_10x_h5(raw_h5, filtered_h5=None, cell_ids=None, channel_name=None,
                include_features=('Gene Expression',), verbose=True, **kwargs):
    """
    Build a SoupChannel from 10X HDF5 files.

    Parameters
    ----------
    raw_h5 : str
        Path to raw_feature_bc_matrix.h5 (all droplets).
    filtered_h5 : str, optional
        Path to filtered_feature_bc_matrix.h5. Required when cell_ids is None.
    cell_ids : array-like, optional
        Cell barcodes. Overrides filtered_h5 when supplied.
    channel_name : str, optional
        Defaults to raw_h5 path.
    include_features : tuple
        Feature types to retain for multi-modal v3 files.
    verbose : bool
    **kwargs
        Forwarded to SoupChannel constructor.

    Returns
    -------
    SoupChannel
    """
    if verbose:
        print("Loading raw count data from H5")
    tod, genes, drop_barcodes, feature_types = read_10x_h5(raw_h5)

    if include_features and not all(ft == 'Gene Expression' for ft in feature_types):
        keep = np.array([ft in include_features for ft in feature_types])
        tod = tod[keep, :]
        genes = genes[keep]

    if verbose:
        print("Loading cell-only count data from H5")

    if cell_ids is not None:
        cell_ids = list(cell_ids)
        drop_idx = {b: i for i, b in enumerate(drop_barcodes)}
        missing = [c for c in cell_ids if c not in drop_idx]
        if missing:
            raise ValueError(f"{len(missing)} supplied cellIDs not found in raw data.")
        col_idx = [drop_idx[c] for c in cell_ids]
        toc = tod[:, col_idx]
        cell_barcodes = pd.Index(cell_ids)
    elif filtered_h5 is not None:
        toc, _, cell_barcodes, filt_ft = read_10x_h5(filtered_h5)
        if include_features and not all(ft == 'Gene Expression' for ft in filt_ft):
            keep = np.array([ft in include_features for ft in filt_ft])
            toc = toc[keep, :]
    else:
        raise ValueError("Provide either filtered_h5 or cell_ids.")

    if channel_name is None:
        channel_name = raw_h5

    return SoupChannel(
        tod=tod,
        toc=toc,
        genes=genes,
        cells=cell_barcodes,
        drop_barcodes=list(drop_barcodes),
        channel_name=channel_name,
        calc_soup_profile=True,
        **kwargs
    )


def _open_file(path, mode='rt'):
    """Open a possibly gzipped file."""
    if path.endswith('.gz'):
        return gzip.open(path, mode)
    return open(path, mode)


def read_10x(data_dir):
    """
    Read a 10X MEX-format directory (genes/features + barcodes + matrix).

    Supports both v2 (genes.tsv, uncompressed) and v3 (features.tsv.gz, compressed).

    Parameters
    ----------
    data_dir : str
        Directory containing matrix.mtx(.gz), barcodes.tsv(.gz),
        and genes.tsv / features.tsv (.gz).

    Returns
    -------
    mat : scipy.sparse.csc_matrix
        (n_genes x n_barcodes) count matrix.
    gene_names : pd.Index
        Gene names (second column of genes/features file).
    barcodes : pd.Index
        Cell/droplet barcodes.
    """
    gz = os.path.exists(os.path.join(data_dir, 'matrix.mtx.gz'))

    def _p(name):
        return os.path.join(data_dir, name + ('.gz' if gz else ''))

    matrix_path = _p('matrix.mtx')
    barcodes_path = _p('barcodes.tsv')

    # Prefer features.tsv (v3) over genes.tsv (v2)
    feat_gz = os.path.exists(os.path.join(data_dir, 'features.tsv.gz'))
    feat_nogz = os.path.exists(os.path.join(data_dir, 'features.tsv'))
    if feat_gz or feat_nogz:
        features_path = os.path.join(data_dir, 'features.tsv' + ('.gz' if feat_gz else ''))
    else:
        features_path = _p('genes.tsv')

    # Read matrix
    with _open_file(matrix_path, 'rb' if matrix_path.endswith('.gz') else 'r') as f:
        if matrix_path.endswith('.gz'):
            raw = f.read()
            mat = scipy.io.mmread(io.BytesIO(raw)).tocsc().astype(float)
        else:
            mat = scipy.io.mmread(f).tocsc().astype(float)
    # mat shape: (n_genes x n_barcodes) — 10X mtx stores features as rows

    with _open_file(barcodes_path) as f:
        barcodes = pd.Index([line.strip() for line in f if line.strip()])

    with _open_file(features_path) as f:
        gene_names = []
        feature_types = []
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue  # skip blank/trailing lines; avoids IndexError on parts[0]
            parts = stripped.split('\t')
            gene_names.append(parts[1] if len(parts) > 1 else parts[0])
            feature_types.append(parts[2] if len(parts) > 2 else 'Gene Expression')

    gene_names = pd.Index(gene_names)

    if mat.shape != (len(gene_names), len(barcodes)):
        raise ValueError(
            f"Matrix shape {mat.shape} does not match "
            f"genes ({len(gene_names)}) x barcodes ({len(barcodes)})."
        )

    return mat, gene_names, barcodes, feature_types


def _load_10x_dir(data_dir, include_features=('Gene Expression',)):
    """Load one 10X cellranger output directory, returning (mat, genes, barcodes)."""
    mat, genes, barcodes, feature_types = read_10x(data_dir)
    feature_types = np.array(feature_types)

    # Filter to requested feature types (v3 multi-modal support)
    if include_features and not all(ft == 'Gene Expression' for ft in feature_types):
        keep = np.array([ft in include_features for ft in feature_types])
        mat = mat[keep, :]
        genes = genes[keep]

    return mat, genes, barcodes


def load_10x(data_dir, cell_ids=None, channel_name=None,
              include_features=('Gene Expression',), verbose=True, **kwargs):
    """
    Load a 10X cellranger output directory and build a SoupChannel.

    Mirrors R SoupX load10X(). Automatically detects v2/v3 cellranger format,
    loads cluster annotations and tSNE/UMAP projections when available.

    Parameters
    ----------
    data_dir : str
        Top-level cellranger output directory (contains raw_gene_bc_matrices/
        or raw_feature_bc_matrix/).
    cell_ids : array-like, optional
        Barcodes of droplets that are cells. If None, uses the filtered matrix.
    channel_name : str, optional
        Name for this channel. Defaults to data_dir.
    include_features : tuple
        Feature types to keep when multiple are present (v3 multi-modal).
    verbose : bool
    **kwargs
        Forwarded to SoupChannel constructor.

    Returns
    -------
    SoupChannel
    """
    is_v3 = os.path.isdir(os.path.join(data_dir, 'raw_feature_bc_matrix'))

    def _first(*paths):
        """Return the first path that exists, or None."""
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    raw_dir = os.path.join(data_dir, 'raw_feature_bc_matrix' if is_v3 else 'raw_gene_bc_matrices')
    if not is_v3:
        subdirs = [d for d in os.listdir(raw_dir)
                   if os.path.isdir(os.path.join(raw_dir, d))]
        raw_dir = os.path.join(raw_dir, subdirs[0]) if subdirs else raw_dir

    if verbose:
        print("Loading raw count data")
    tod, genes, drop_barcodes = _load_10x_dir(raw_dir, include_features=include_features)

    if verbose:
        print("Loading cell-only count data")

    if cell_ids is not None:
        cell_ids = list(cell_ids)
        drop_idx = {b: i for i, b in enumerate(drop_barcodes)}
        missing = [c for c in cell_ids if c not in drop_idx]
        if missing:
            raise ValueError(f"{len(missing)} supplied cellIDs not found in raw data.")
        col_idx = [drop_idx[c] for c in cell_ids]
        toc = tod[:, col_idx]
        cell_barcodes = pd.Index(cell_ids)
    else:
        filt_dir = os.path.join(data_dir,
                                'filtered_feature_bc_matrix' if is_v3 else 'filtered_gene_bc_matrices')
        if not is_v3:
            subdirs = [d for d in os.listdir(filt_dir)
                       if os.path.isdir(os.path.join(filt_dir, d))]
            filt_dir = os.path.join(filt_dir, subdirs[0]) if subdirs else filt_dir
        toc, _, cell_barcodes = _load_10x_dir(filt_dir, include_features=include_features)

    if verbose:
        print("Loading extra analysis data where available")

    meta_data = None
    DR = None
    _a = os.path.join(data_dir, 'analysis')

    # ── Cluster annotations ───────────────────────────────────────────────────
    # Try all known cellranger layout variants in priority order:
    #   multi output (v6+)  →  v7 GEX-prefixed  →  standard (v3-v6, v8+)
    clust_path = _first(
        os.path.join(_a, 'clustering', 'gex', 'graphclust', 'clusters.csv'),
        os.path.join(_a, 'clustering', 'gene_expression_graphclust', 'clusters.csv'),
        os.path.join(_a, 'clustering', 'graphclust', 'clusters.csv'),
    )
    if clust_path:
        clusters_df = pd.read_csv(clust_path)
        meta_data = pd.DataFrame({'clusters': clusters_df['Cluster'].values},
                                  index=pd.Index(clusters_df['Barcode'].values))

    # ── Fine-grained clusters (kmeans) ────────────────────────────────────────
    fine_path = _first(
        os.path.join(_a, 'clustering', 'gex', 'kmeans_10_clusters', 'clusters.csv'),
        os.path.join(_a, 'clustering', 'gene_expression_kmeans_10_clusters', 'clusters.csv'),
        os.path.join(_a, 'clustering', 'kmeans_10_clusters', 'clusters.csv'),
    )
    if fine_path:
        fine_df = pd.read_csv(fine_path)
        fine_series = pd.Series(fine_df['Cluster'].values,
                                index=pd.Index(fine_df['Barcode'].values),
                                name='clustersFine')
        if meta_data is None:
            meta_data = fine_series.to_frame()
        else:
            meta_data = meta_data.join(fine_series, how='left')

    # ── tSNE projection ───────────────────────────────────────────────────────
    tsne_path = _first(
        os.path.join(_a, 'dimensionality_reduction', 'gex', 'tsne_projection.csv'),
        os.path.join(_a, 'tsne', 'gene_expression_2_components', 'projection.csv'),
        os.path.join(_a, 'tsne', '2_components', 'projection.csv'),
    )
    if tsne_path:
        tsne_df = pd.read_csv(tsne_path)
        tsne_part = pd.DataFrame(
            {'tSNE1': tsne_df.iloc[:, 1].values, 'tSNE2': tsne_df.iloc[:, 2].values},
            index=pd.Index(tsne_df.iloc[:, 0].values)
        )
        if meta_data is None:
            meta_data = tsne_part
        else:
            meta_data = meta_data.join(tsne_part, how='left')
        DR = ['tSNE1', 'tSNE2']

    # ── UMAP projection ───────────────────────────────────────────────────────
    umap_path = _first(
        os.path.join(_a, 'dimensionality_reduction', 'gex', 'umap_projection.csv'),
        os.path.join(_a, 'umap', 'gene_expression_2_components', 'projection.csv'),
        os.path.join(_a, 'umap', '2_components', 'projection.csv'),
    )
    if umap_path:
        umap_df = pd.read_csv(umap_path)
        umap_part = pd.DataFrame(
            {'UMAP1': umap_df.iloc[:, 1].values, 'UMAP2': umap_df.iloc[:, 2].values},
            index=pd.Index(umap_df.iloc[:, 0].values)
        )
        if meta_data is None:
            meta_data = umap_part
        else:
            meta_data = meta_data.join(umap_part, how='left')
        DR = ['UMAP1', 'UMAP2']   # prefer UMAP as primary DR when available

    # Align meta_data index to cell barcodes (strip -1 suffix if needed)
    if meta_data is not None:
        if not all(b in meta_data.index for b in cell_barcodes):
            stripped = pd.Index([b.rstrip('-1').rstrip('-') for b in meta_data.index])
            if all(b in stripped for b in cell_barcodes):
                meta_data.index = stripped
        meta_data = meta_data.reindex(cell_barcodes)

    if channel_name is None:
        channel_name = data_dir

    sc = SoupChannel(
        tod=tod,
        toc=toc,
        genes=genes,
        cells=cell_barcodes,
        meta_data=meta_data,
        drop_barcodes=list(drop_barcodes),
        channel_name=channel_name,
        data_dir=data_dir,
        data_type='10X',
        is_v3=is_v3,
        calc_soup_profile=True,
        **kwargs
    )
    if DR is not None:
        sc.DR = DR

    return sc
