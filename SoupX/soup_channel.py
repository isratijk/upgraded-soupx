import numpy as np
import pandas as pd
import scipy.sparse
import warnings


class SoupChannel:
    """
    Container for all data and results relating to a single droplet sequencing channel.

    Attributes
    ----------
    tod : scipy.sparse.csc_matrix or None
        Table of droplets (genes x droplets). Set to None after estimate_soup
        unless keep_droplets=True.
    toc : scipy.sparse.csc_matrix
        Table of counts (genes x cells). Filtered barcodes only.
    genes : pd.Index
        Gene names (row labels for toc/tod).
    cells : pd.Index
        Cell barcodes (column labels for toc).
    meta_data : pd.DataFrame
        Per-cell metadata. Index = cell barcodes. Always contains 'nUMIs'.
        Populated with 'clusters', 'rho', DR columns as pipeline progresses.
    n_drop_umis : pd.Series or None
        Per-droplet UMI counts (index = droplet barcodes).
    soup_profile : pd.DataFrame or None
        Per-gene soup expression profile. Index = gene names.
        Columns: 'est' (fraction), 'counts'.
    DR : list or None
        Names of columns in meta_data that hold the dimension reduction coords.
    fit : object or None
        Stored fit result from contamination estimation.
    """

    def __init__(self, tod, toc, genes=None, cells=None, meta_data=None,
                 calc_soup_profile=True, **kwargs):
        """
        Parameters
        ----------
        tod : scipy.sparse matrix
            Table of droplets (genes x droplets).
        toc : scipy.sparse matrix
            Table of counts (genes x cells).
        genes : array-like, optional
            Gene names. If None, uses integer indices.
        cells : array-like, optional
            Cell barcodes. If None, uses integer indices.
        meta_data : pd.DataFrame, optional
            Additional per-cell metadata. Index must match cells.
        calc_soup_profile : bool
            Whether to call estimate_soup automatically.
        **kwargs
            Additional attributes stored on the object (channel_name, data_dir, etc.).
        """
        if tod.shape[0] != toc.shape[0]:
            raise ValueError(
                "tod and toc have different numbers of genes. "
                f"tod: {tod.shape[0]}, toc: {toc.shape[0]}"
            )

        self.tod = scipy.sparse.csc_matrix(tod)
        toc = scipy.sparse.csc_matrix(toc)

        # Detect and remove zero-UMI cells before any downstream use.
        # Leaving them in causes division-by-zero in normalization and EM steps.
        _cell_umis = np.array(toc.sum(axis=0)).flatten()
        _zero_mask = _cell_umis == 0
        if _zero_mask.any():
            n_zero = int(_zero_mask.sum())
            warnings.warn(
                f"{n_zero} cell(s) have zero total UMI counts and have been removed. "
                "Filter zero-UMI cells before constructing SoupChannel to suppress this warning.",
                UserWarning, stacklevel=2,
            )
            keep = ~_zero_mask
            toc = toc[:, keep]
            if cells is not None:
                cells = np.asarray(cells)[keep]
            if meta_data is not None and hasattr(meta_data, 'iloc'):
                meta_data = meta_data.iloc[keep]

        self.toc = toc
        n_genes, n_cells = toc.shape
        n_drop = tod.shape[1]

        self.genes = pd.Index(genes) if genes is not None else pd.RangeIndex(n_genes)
        self.cells = pd.Index(cells) if cells is not None else pd.RangeIndex(n_cells)

        if genes is not None and len(genes) != n_genes:
            raise ValueError("genes length must equal number of rows in toc/tod.")
        if cells is not None and len(cells) != n_cells:
            raise ValueError("cells length must equal number of columns in toc.")

        # Build core metadata
        self.meta_data = pd.DataFrame(
            {'nUMIs': np.array(toc.sum(axis=0)).flatten()},
            index=self.cells
        )
        if meta_data is not None:
            meta_data = meta_data.drop(columns=['nUMIs'], errors='ignore')
            # Align on index
            self.meta_data = self.meta_data.join(meta_data, how='left')

        # Per-droplet UMI counts
        drop_labels = kwargs.pop('drop_barcodes', None)
        drop_umis = np.array(tod.sum(axis=0)).flatten()
        self.n_drop_umis = pd.Series(
            drop_umis,
            index=pd.Index(drop_labels) if drop_labels is not None else pd.RangeIndex(n_drop)
        )

        self.soup_profile = None
        self.DR = None
        self.fit = None

        # Store extra kwargs as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

        if calc_soup_profile:
            from .estimate_soup import estimate_soup
            estimate_soup(self, inplace=True)

    def __repr__(self):
        n_genes = len(self.genes)
        n_cells = len(self.cells)
        rho_info = ""
        if self.meta_data is not None and 'rho' in self.meta_data.columns:
            rho_info = f", rho={self.meta_data['rho'].mean():.3f}"
        return f"SoupChannel with {n_genes} genes and {n_cells} cells{rho_info}"

    def copy(self):
        import copy
        return copy.deepcopy(self)

    # ── AnnData interoperability ───────────────────────────────────────────────

    def to_anndata(self, corrected=None):
        """
        Convert SoupChannel to AnnData (cells × genes convention).

        Parameters
        ----------
        corrected : scipy.sparse matrix, optional
            Corrected matrix from adjust_counts(). Stored as
            adata.layers['corrected'] and set as adata.X when provided.

        Returns
        -------
        anndata.AnnData
        """
        try:
            import anndata
        except ImportError:
            raise ImportError("anndata required: pip install anndata")

        X = self.toc.T.tocsr().astype(float)
        obs = self.meta_data.copy() if self.meta_data is not None else pd.DataFrame(index=self.cells)
        var = pd.DataFrame(index=self.genes)
        if self.soup_profile is not None:
            var = var.join(self.soup_profile)

        adata = anndata.AnnData(X=X, obs=obs, var=var)
        adata.layers['counts'] = X.copy()

        if corrected is not None:
            adata.layers['corrected'] = corrected.T.tocsr().astype(float)
            adata.X = adata.layers['corrected']

        # Store DR coords in obsm for scanpy compatibility
        if self.DR is not None and self.meta_data is not None:
            dr_cols = [c for c in self.DR if c in self.meta_data.columns]
            if dr_cols:
                arr = self.meta_data[dr_cols].fillna(0).values.astype(float)
                key = 'X_umap' if any('UMAP' in c.upper() for c in dr_cols) else 'X_tsne'
                adata.obsm[key] = arr

        adata.uns['soupx_channel_name'] = getattr(self, 'channel_name', '')
        if self.soup_profile is not None:
            adata.uns['soup_profile'] = self.soup_profile.to_dict()

        return adata

    @classmethod
    def from_anndata(cls, adata, tod, layer='counts', soup_profile=None, **kwargs):
        """
        Build a SoupChannel from an AnnData object.

        Parameters
        ----------
        adata : anndata.AnnData
            (cells × genes) AnnData. toc is taken from adata.layers[layer]
            or adata.X when layer is absent.
        tod : scipy.sparse matrix
            (genes × droplets) table of all droplets for soup estimation.
        layer : str
            Layer to use as the cell count matrix (default 'counts').
        soup_profile : pd.DataFrame, optional
            Pre-computed soup profile; skips automatic estimation when supplied.
        **kwargs
            Forwarded to SoupChannel constructor.

        Returns
        -------
        SoupChannel
        """
        if layer in adata.layers:
            toc = scipy.sparse.csc_matrix(adata.layers[layer]).T
        else:
            toc = scipy.sparse.csc_matrix(adata.X).T

        genes = pd.Index(adata.var_names)
        cells = pd.Index(adata.obs_names)
        meta_data = adata.obs.copy()

        # Pull dimensionality reductions from obsm into meta_data columns
        DR = None
        for obsm_key, arr in adata.obsm.items():
            if obsm_key.startswith('X_') and arr.shape[1] >= 2:
                prefix = obsm_key[2:].upper()
                col1, col2 = f'{prefix}1', f'{prefix}2'
                meta_data[col1] = arr[:, 0]
                meta_data[col2] = arr[:, 1]
                DR = [col1, col2]

        sc = cls(
            tod=tod,
            toc=toc,
            genes=genes,
            cells=cells,
            meta_data=meta_data,
            calc_soup_profile=(soup_profile is None),
            **kwargs
        )
        if soup_profile is not None:
            sc.soup_profile = soup_profile
        if DR is not None:
            sc.DR = DR

        return sc

    # ── Serialization ──────────────────────────────────────────────────────────

    def save(self, path):
        """
        Save SoupChannel to disk using pickle.

        Parameters
        ----------
        path : str
            Output file path (e.g. 'channel.pkl').
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """
        Load a SoupChannel saved with save().

        Parameters
        ----------
        path : str
            Path to saved .pkl file.

        Returns
        -------
        SoupChannel
        """
        import pickle
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)!r}, expected SoupChannel.")
        return obj
