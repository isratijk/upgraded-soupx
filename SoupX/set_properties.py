import numpy as np
import pandas as pd


def set_soup_profile(sc, soup_profile, inplace=False):
    """
    Manually set or update the soup profile.

    Parameters
    ----------
    sc : SoupChannel
    soup_profile : pd.DataFrame
        Must have index matching sc.genes and columns 'est' and 'counts'.
    inplace : bool

    Returns
    -------
    SoupChannel
    """
    if not inplace:
        sc = sc.copy()

    if 'est' not in soup_profile.columns:
        raise ValueError("soup_profile must have an 'est' column.")
    if 'counts' not in soup_profile.columns:
        raise ValueError("soup_profile must have a 'counts' column.")
    if soup_profile.index.has_duplicates:
        raise ValueError("soup_profile index contains duplicate gene names.")
    if not soup_profile.index.isin(sc.genes).all():
        raise ValueError("Not all genes in soup_profile found in sc.genes.")

    missing_genes = sc.genes.difference(soup_profile.index)
    if len(missing_genes) > 0:
        raise ValueError(
            "soup_profile must contain every gene in sc.genes. "
            f"Missing {len(missing_genes)} gene(s)."
        )

    profile = soup_profile.reindex(sc.genes)
    if profile[['est', 'counts']].isna().any().any():
        raise ValueError("soup_profile contains missing values after alignment.")
    if not np.isfinite(profile[['est', 'counts']].to_numpy(dtype=float)).all():
        raise ValueError("soup_profile must contain only finite numeric values.")
    if (profile['est'] < 0).any() or (profile['counts'] < 0).any():
        raise ValueError("soup_profile cannot contain negative values.")

    sc.soup_profile = profile
    return sc


def set_clusters(sc, clusters, inplace=False):
    """
    Add or update cluster labels in sc.meta_data.

    Parameters
    ----------
    sc : SoupChannel
    clusters : array-like or pd.Series
        Cluster ID for each cell. If a Series, index must map to cell barcodes.
        If an array, order must match sc.cells.
    inplace : bool

    Returns
    -------
    SoupChannel
    """
    if not inplace:
        sc = sc.copy()

    if isinstance(clusters, pd.Series) and clusters.index.isin(sc.cells).all():
        sc.meta_data['clusters'] = clusters.reindex(sc.cells).astype(str).values
    else:
        clusters = np.asarray(clusters)
        if len(clusters) != len(sc.cells):
            raise ValueError(
                "clusters must be either a named Series mapping to cell barcodes, "
                "or an array with length equal to the number of cells."
            )
        sc.meta_data['clusters'] = clusters.astype(str)

    if sc.meta_data['clusters'].isna().any():
        raise ValueError("NAs found in cluster names after assignment.")

    return sc


def set_contamination_fraction(sc, cont_frac, force_accept=False, inplace=False):
    """
    Set the contamination fraction (rho) for each cell.

    Parameters
    ----------
    sc : SoupChannel
    cont_frac : float or array-like
        Contamination fraction. Scalar applies to all cells; array must
        be length n_cells or a named Series with cell barcode index.
    force_accept : bool
        If True, demote errors about extreme values to warnings.
    inplace : bool

    Returns
    -------
    SoupChannel
    """
    if not inplace:
        sc = sc.copy()

    if isinstance(cont_frac, pd.Series):
        if cont_frac.index.has_duplicates:
            raise ValueError("cont_frac Series index contains duplicate cell barcodes.")

        missing_cells = sc.cells.difference(cont_frac.index)
        extra_cells = cont_frac.index.difference(sc.cells)
        if len(missing_cells) > 0 or len(extra_cells) > 0:
            bits = []
            if len(missing_cells) > 0:
                bits.append(f"missing {len(missing_cells)} cell(s)")
            if len(extra_cells) > 0:
                bits.append(f"{len(extra_cells)} unknown cell(s)")
            raise ValueError(
                "cont_frac Series must cover exactly the cells in sc.cells: "
                + ", ".join(bits) + "."
            )

        cont_frac_aligned = cont_frac.reindex(sc.cells)
        if cont_frac_aligned.isna().any():
            raise ValueError("cont_frac Series contains missing values after alignment.")

        cont_frac_arr = cont_frac_aligned.to_numpy(dtype=float)
    else:
        cont_frac_arr = np.asarray(cont_frac, dtype=float)

    if not np.isfinite(cont_frac_arr).all():
        raise ValueError("cont_frac must contain only finite numeric values.")
    if np.any(cont_frac_arr < 0.0):
        raise ValueError("Contamination fraction < 0 detected. This is impossible.")

    if np.any(cont_frac_arr > 1.0):
        raise ValueError(
            "Contamination fraction > 1 detected. This is impossible and likely "
            "represents a failure in the estimation procedure."
        )

    def _warn(msg):
        import warnings
        warnings.warn(msg)

    def _error(msg):
        if force_accept:
            print(f"[forceAccept] {msg}")
        else:
            raise ValueError(msg)

    if np.any(cont_frac_arr > 0.5):
        _error(
            f"Extremely high contamination estimated ({cont_frac_arr.max():.2g}). "
            "This likely represents a failure in estimating the contamination fraction. "
            "Set force_accept=True to proceed."
        )
    elif np.any(cont_frac_arr > 0.3):
        _warn(f"Estimated contamination is very high ({cont_frac_arr.max():.2g}).")

    if isinstance(cont_frac, pd.Series):
        sc.meta_data['rho'] = cont_frac_arr
    elif cont_frac_arr.ndim == 0 or len(cont_frac_arr) == 1:
        sc.meta_data['rho'] = float(cont_frac_arr.flat[0])
    else:
        if len(cont_frac_arr) != len(sc.cells):
            raise ValueError(
                "cont_frac must be a scalar or have length equal to the number of cells."
            )
        sc.meta_data['rho'] = cont_frac_arr

    return sc


def set_dr(sc, dr, reduct_name=None, inplace=False):
    """
    Add a dimension reduction embedding to sc.meta_data.

    Parameters
    ----------
    sc : SoupChannel
    dr : pd.DataFrame
        Two-column DataFrame. Index must match sc.cells or have same length.
    reduct_name : str, optional
        Prefix for column names (e.g. 'UMAP' → ['UMAP_1', 'UMAP_2']).
    inplace : bool

    Returns
    -------
    SoupChannel
    """
    if not inplace:
        sc = sc.copy()

    dr = pd.DataFrame(dr)
    if dr.shape[1] > 2:
        import warnings
        warnings.warn(f"DR has {dr.shape[1]} columns, using first two.")
        dr = dr.iloc[:, :2]

    # Try to align by index
    m = dr.index.isin(sc.cells)
    if m.all():
        dr = dr.reindex(sc.cells)
    elif len(dr) == len(sc.cells):
        dr.index = sc.cells
    else:
        raise ValueError(
            f"DR index not found in sc.cells and row count differs "
            f"({len(dr)} vs {len(sc.cells)})."
        )

    if reduct_name is not None:
        dr.columns = [f'{reduct_name}_1', f'{reduct_name}_2']

    for col in dr.columns:
        sc.meta_data[col] = dr[col].values

    sc.DR = list(dr.columns)
    return sc
