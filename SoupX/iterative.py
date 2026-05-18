"""
Iterative soup-profile refinement for improved contamination estimation.

After one round of correction, genes remaining highly expressed in corrected
cells are likely truly cellular rather than ambient.  Reweighting the soup
profile away from such genes and re-running auto_est_cont converges to a
more accurate contamination estimate — especially in datasets where ambient
RNA overlaps substantially with cellular expression (e.g. PBMC, fetal liver).

Each iteration:
  1. Run auto_est_cont on current soup profile -> rho.
  2. Adjust counts -> corrected matrix.
  3. Down-weight soup genes whose corrected cell share > soup share.
  4. Repeat up to n_iter rounds or until rho converges.
"""

import copy
import warnings

import numpy as np
import scipy.sparse


def iterative_auto_est_cont(sc, n_iter=2, shrink_factor=5.0, tol=1e-3,
                             inplace=False, **aec_kwargs):
    """
    Iteratively refine the soup profile and re-estimate contamination.

    Parameters
    ----------
    sc : SoupChannel
        Must have soup_profile set and clusters in meta_data.
    n_iter : int
        Number of refinement iterations (default 2; 1-3 is typical).
    shrink_factor : float
        Controls how aggressively cellular genes are penalized in the
        soup profile.  Higher = more penalization (default 5.0).
    tol : float
        Mean absolute change in rho to declare convergence (default 1e-3).
    inplace : bool
    **aec_kwargs : forwarded to auto_est_cont

    Returns
    -------
    SoupChannel with refined rho and soup_profile
    """
    from .estimation import auto_est_cont
    from .correction import adjust_counts

    if not inplace:
        sc = sc.copy()

    sc_curr = auto_est_cont(sc, **aec_kwargs)

    for i in range(1, n_iter):
        prev_rho = sc_curr.meta_data['rho'].values.copy().astype(float)

        corrected = adjust_counts(sc_curr, method='subtraction', verbose=0)
        corrected = scipy.sparse.csc_matrix(corrected)
        corrected.data[corrected.data < 0] = 0.0
        corrected.eliminate_zeros()

        refined_soup_df = _shrink_cellular_soup(
            sc.soup_profile, corrected, shrink_factor=shrink_factor
        )

        # Re-start from original counts with refined soup profile
        sc_next = sc.copy()
        sc_next.soup_profile = refined_soup_df
        if 'clusters' in sc_curr.meta_data.columns:
            sc_next.meta_data['clusters'] = sc_curr.meta_data['clusters'].copy()

        sc_curr = auto_est_cont(sc_next, **aec_kwargs)

        delta = float(np.abs(sc_curr.meta_data['rho'].values - prev_rho).mean())
        if delta < tol:
            break

    return sc_curr


def _shrink_cellular_soup(soup_profile_df, corrected_toc, shrink_factor=5.0,
                           min_weight=0.3):
    """
    Return a new soup profile with reduced weight for genes expressed in corrected cells.

    Weight_g = max(1 / (1 + shrink_factor * cell_share_g / soup_share_g), min_weight)

    min_weight=0.3 floors the suppression so no gene drops below 30% of its
    original soup weight.  This prevents cross-species genes from being
    eliminated in barnyard experiments where corrected cells have zero
    foreign-species expression.
    """
    soup = soup_profile_df['est'].values.astype(float)
    soup_norm = soup / (soup.sum() + 1e-10)

    cell_mean = np.asarray(corrected_toc.mean(axis=1)).flatten().astype(float)
    cell_norm = cell_mean / (cell_mean.sum() + 1e-10)

    ratio  = cell_norm / (soup_norm + 1e-10)
    weight = 1.0 / (1.0 + shrink_factor * ratio)
    weight = np.maximum(weight, min_weight)

    new_soup = soup * weight
    new_soup = np.maximum(new_soup, 0)
    new_soup /= (new_soup.sum() + 1e-10)

    new_df = soup_profile_df.copy()
    new_df['est'] = new_soup
    return new_df
